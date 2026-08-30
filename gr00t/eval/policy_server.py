"""Lightweight zmq + msgpack policy server/client (no torch on the wire).

Lifted from RLDX-1's `rldx/policy/server_client.py` — same MsgSerializer +
PolicyServer pattern, with ModalityConfig/BasePolicy bindings stripped so
this module has no torch-tensor coupling on the client side. The server can
still feed it `torch.Tensor` outputs; they're up-cast to numpy before being
shipped over the wire.
"""

from dataclasses import dataclass
import io
from typing import Any, Callable, Dict

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """msgpack-based codec. Converts torch tensors → numpy on the way out, and
    numpy arrays survive a round trip via in-memory `np.save` / `np.load`.
    Pure msgpack on the client side means we never need `import torch`."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        # torch only imported lazily on the server side — keep the client free
        # of any torch dependency.
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                t = obj.detach().cpu()
                if t.dtype in (torch.bfloat16, torch.float16):
                    t = t.float()
                obj = t.numpy()
        except ImportError:
            pass
        # numpy scalars (np.float32 / np.int64 / etc.) are not msgpack-native.
        # Wrap as 0-d ndarray so the receiver still gets a numpy object with
        # `.shape` / `.dtype`, matching what the policy expects.
        if isinstance(obj, np.generic):
            obj = np.asarray(obj)
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class PolicyServer:
    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: Dict[str, EndpointHandler] = {}
        self.register_endpoint("ping", lambda: {"status": "ok"}, requires_input=False)
        self.register_endpoint("kill", self._kill, requires_input=False)

    def _kill(self):
        self.running = False

    def register_endpoint(self, name: str, handler: Callable, requires_input: bool = True):
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}", flush=True)
        while self.running:
            try:
                request = MsgSerializer.from_bytes(self.socket.recv())
                endpoint = request.get("endpoint", "get_action")
                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                handler = self._endpoints[endpoint]
                if handler.requires_input:
                    result = handler.handler(**request.get("data", {}))
                else:
                    result = handler.handler()
                self.socket.send(MsgSerializer.to_bytes(result))
            except Exception as e:
                import traceback
                print(f"Error in server: {e}\n{traceback.format_exc()}", flush=True)
                self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))


class PolicyClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{host}:{port}")

    def call_endpoint(self, endpoint: str, data: Any = None, requires_input: bool = True):
        msg = {"endpoint": endpoint}
        if requires_input:
            msg["data"] = data or {}
        self.socket.send(MsgSerializer.to_bytes(msg))
        return MsgSerializer.from_bytes(self.socket.recv())

    def ping(self):
        return self.call_endpoint("ping", requires_input=False)
