"""Prepare and label RoboCasa D1 v2 windows with resumable raw probabilities."""
import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq

from robocasa_d1_policy import CATEGORIES, VERSION, select_ratio, validate_result
from robocasa_d1_reader import Reader, digest

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = REPO/"vlm_gate/prompts/robocasa_d1_v2.txt"
DEFAULT_CAPS = REPO/"vlm_gate/analysis/robocasa_task_ceilings.json"


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    tmp.replace(path)


def prepare(args):
    ds = Reader(args.dataset)
    caps = json.loads(Path(args.caps).read_text())
    tasks = {}
    for ep, meta in sorted(ds.episodes.items()):
        task = meta["tasks"][1]
        if task not in caps or meta["length"] <= 16:
            raise ValueError(f"missing cap or too short: {ep}, {task}")
        tasks.setdefault(task, []).append(ep)
    rows = []
    if args.pilot:
        for task, episodes in sorted(tasks.items()):
            chosen = [episodes[0], episodes[len(episodes)//2]]
            for ep in chosen:
                top = ds.episodes[ep]["length"]-17
                for fraction in (.05, .35, .65, .90):
                    frame = round(fraction*top)
                    rows.append([ep, frame])
        rows = sorted(set(map(tuple, rows)))
    manifest = dict(schema="robocasa_d1_manifest_v2", dataset_fingerprint=ds.fingerprint,
                    mode="pilot" if args.pilot else "full", stride=args.stride,
                    rows=rows if args.pilot else None, tasks=sorted(tasks),
                    episodes=len(ds.episodes), tail=16, observation_offsets=[0, 8, 16],
                    eligible_rows=len(rows) if args.pilot else sum(len(range(0, m["length"]-16, args.stride)) for m in ds.episodes.values()),
                    cap_sha256=digest(args.caps))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists() and json.loads(Path(args.out).read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError("manifest already exists with different inputs")
    json_write(args.out, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2))


class Legacy:
    def __init__(self, path, episodes):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "fixed"])
        self.ep = table["episode_index"].to_numpy()
        self.frame = table["frame_index"].to_numpy()
        self.fixed = table["fixed"].to_numpy()
        if not np.isin(self.fixed, [0, 1]).all():
            raise ValueError("bad legacy fixed values")
        offset = 0
        self.offsets = {}
        for ep, meta in sorted(episodes.items()):
            n = meta["length"]-4
            if not np.all(self.ep[offset:offset+n] == ep) or not np.array_equal(self.frame[offset:offset+n], np.arange(n)):
                raise ValueError(f"legacy keys mismatch ep{ep}")
            self.offsets[ep] = offset
            offset += n
        if offset != len(self.ep):
            raise ValueError("unexpected legacy rows")

    def get(self, ep, frame):
        return bool(self.fixed[self.offsets[ep]+frame])


def request(url, items, prompt, timeout):
    payload = dict(batch=items, guidance="", question=prompt, n_ask=4, n_grade=0,
                   category_tokens=CATEGORIES, mode="text", max_new_tokens=96)
    req = urllib.request.Request(url.rstrip("/")+"/judge", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read())
    results = value.get("results")
    if not isinstance(results, list) or len(results) != len(items):
        raise ValueError(f"wrong result count: {value}")
    return results


def complete_keys(path):
    """Recover only a physically incomplete final record; never rewind batches."""
    done = set()
    if not path.exists():
        return done
    with path.open("rb+") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n") or f.read(1):
                    raise ValueError("corrupt complete/interior JSONL record")
                f.seek(start)
                f.truncate()
                with path.with_suffix(".recovery.jsonl").open("a") as audit:
                    audit.write(json.dumps(dict(time=time.time(), truncated_offset=start, bytes=len(line)))+"\n")
                break
            key = (row["episode_index"], row["frame_index"])
            if key in done:
                raise ValueError(f"duplicate output key: {key}")
            done.add(key)
    return done


def label(args):
    ds = Reader(args.dataset)
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["dataset_fingerprint"] != ds.fingerprint:
        raise ValueError("dataset/manifest mismatch")
    caps = json.loads(Path(args.caps).read_text())
    if digest(args.caps) != manifest["cap_sha256"]:
        raise ValueError("caps changed since manifest")
    prompt = Path(args.prompt).read_text()
    legacy = Legacy(args.legacy, ds.episodes)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fingerprint = dict(schema="robocasa_d1_raw_v2", policy_version=VERSION,
                       manifest_sha256=digest(args.manifest), dataset_fingerprint=ds.fingerprint,
                       prompt_sha256=digest(args.prompt), cap_sha256=digest(args.caps),
                       legacy_sha256=digest(args.legacy), model_revision=args.model_revision,
                       category_order=CATEGORIES, layout="3times_3views_768x840_v1",
                       reader_sha256=digest(Path(__file__).with_name("robocasa_d1_reader.py")),
                       policy_sha256=digest(Path(__file__).with_name("robocasa_d1_policy.py")),
                       server_sha256=digest(Path(__file__).with_name("vlm_gate_cosmos.py")),
                       labeler_sha256=digest(__file__), shard=args.shard, num_shards=args.num_shards)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != fingerprint:
            raise ValueError("resume semantic fingerprint mismatch")
    elif path.exists():
        raise ValueError("raw output without metadata")
    else:
        json_write(meta_path, fingerprint)
    done = complete_keys(path)
    def keys():
        if manifest["rows"] is not None:
            yield from map(tuple, manifest["rows"])
        else:
            for ep, meta in sorted(ds.episodes.items()):
                for frame in range(0, meta["length"]-16, manifest["stride"]):
                    yield ep, frame
    allowed = {key for key in keys() if key[0] % args.num_shards == args.shard}
    if not done <= allowed:
        raise ValueError("resume contains unexpected keys")
    pending = iter(sorted(allowed-done))
    total = len(allowed)
    started, written = time.monotonic(), 0
    with path.open("a") as output, path.with_suffix(".errors.jsonl").open("a") as errors:
        while True:
            batch = []
            for _ in range(args.batch_size):
                key = next(pending, None)
                if key is None:
                    break
                batch.append(key)
            if not batch or (args.limit_rows and written >= args.limit_rows):
                break
            items, facts = [], []
            for ep, frame in batch:
                x, text = ds.facts(ep, frame)
                image = ds.image(ep, frame)
                if args.save_images:
                    image_path = Path(args.save_images)/f"ep{ep:06d}_f{frame:06d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                items.append(dict(images_b64=[base64.b64encode(stream.getvalue()).decode()],
                                  instruction=ds.episodes[ep]["tasks"][0]+"\n"+text))
                facts.append((x, int(ds.indices[frame])))
            try:
                results = request(args.url, items, prompt, args.timeout)
            except Exception as exc:
                results = [dict(error=repr(exc)) for _ in batch]
            for i, ((ep, frame), item, (x, global_index)) in enumerate(zip(batch, items, facts)):
                result = results[i]
                for attempt in range(3):
                    try:
                        picks, probs = validate_result(result)
                        break
                    except Exception as exc:
                        errors.write(json.dumps(dict(episode_index=ep, frame_index=frame,
                                                     attempt=attempt, error=repr(exc), result=result))+"\n")
                        errors.flush()
                        if attempt == 2:
                            raise RuntimeError(f"technical failure ep{ep} frame{frame}") from exc
                        try:
                            result = request(args.url, [item], prompt, args.timeout)[0]
                        except Exception as retry_exc:
                            result = dict(error=repr(retry_exc))
                task = ds.episodes[ep]["tasks"][1]
                row = dict(episode_index=ep, frame_index=frame, index=global_index, task=task,
                           instruction=ds.episodes[ep]["tasks"][0],
                           observation_indices=[frame, frame+8, frame+16],
                           category_probs=probs, generated_text=result["text"],
                           n_input_tokens=result.get("n_input_tokens"), image_grid_thw=result.get("image_grid_thw"),
                           **dict(zip("ABCD", picks)), command_facts=x,
                           legacy_guard_source="pinned_contact_release_including_proxy_fallback",
                           **select_ratio(probs, picks, caps[task][1], legacy.get(ep, frame)))
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
                written += 1
            output.flush()
            os.fsync(output.fileno())
            elapsed = time.monotonic()-started
            status = dict(completed=len(done)+written, total=total, new_rows=written,
                          elapsed_seconds=elapsed, rows_per_second=written/max(elapsed, 1e-9),
                          last_key=batch[-1], time=time.time())
            json_write(path.with_suffix(".status.json"), status)
            print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--caps", default=str(DEFAULT_CAPS))
    prep.add_argument("--out", required=True)
    prep.add_argument("--pilot", action="store_true")
    prep.add_argument("--stride", type=int, default=1)
    lab = sub.add_parser("label")
    for name in ("dataset", "manifest", "legacy", "out", "url", "model-revision"):
        lab.add_argument("--"+name, required=True)
    lab.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    lab.add_argument("--caps", default=str(DEFAULT_CAPS))
    lab.add_argument("--batch-size", type=int, default=4)
    lab.add_argument("--num-shards", type=int, default=1)
    lab.add_argument("--shard", type=int, default=0)
    lab.add_argument("--timeout", type=float, default=600)
    lab.add_argument("--limit-rows", type=int, default=0)
    lab.add_argument("--save-images")
    args = p.parse_args()
    if args.command == "prepare":
        if args.stride < 1: p.error("stride must be positive")
        prepare(args)
    else:
        if args.batch_size < 1 or not 0 <= args.shard < args.num_shards:
            p.error("invalid batching/shard")
        label(args)


if __name__ == "__main__":
    main()
