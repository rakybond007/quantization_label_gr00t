"""배속 라벨 jsonl 을 parquet 으로. 통째로 안 올리고 흘려 쓴다.

`ratio_label.py` 는 parquet 을 만들 때 200만 행을 한 DataFrame 으로 올린다.
등급까지 담으면 열이 열넷이라, 이미 메모리에 있는 원본 행과 겹쳐 로그인 노드에서
OOM 으로 죽는다(exit 137). jsonl 은 그 전에 이미 온전히 쓰였으므로 여기서
이어받아 조각으로 변환한다.

    python ratio_jsonl_to_parquet.py <in.jsonl> [out.parquet] [--chunk 200000]
"""
import argparse, json, os
import pyarrow as pa
import pyarrow.parquet as pq

# jsonl 키 -> 배포 열 이름. `apply_ratio_labels.py` 가 읽는 이름에 맞춘다.
REN = {"ep": "episode_index", "f": "frame_index", "guard": "fixed"}
INT8 = ("A", "B", "C", "D", "E", "fixed")


def schema_from(keys):
    f = []
    for k in keys:
        if k in ("episode_index", "frame_index"):
            f.append(pa.field(k, pa.int32()))
        elif k in INT8:
            f.append(pa.field(k, pa.int8()))
        elif k == "task":
            f.append(pa.field(k, pa.string()))
        else:
            f.append(pa.field(k, pa.float32()))
    return pa.schema(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default="")
    ap.add_argument("--chunk", type=int, default=200_000)
    a = ap.parse_args()
    dst = a.dst or a.src.replace(".jsonl", ".parquet")

    # 첫 줄만 보면 열을 놓친다: `guard` 는 접촉이 걸린 행에만 붙는 경우가 있어
    # 행마다 키 집합이 다르다. 앞부분을 훑어 열을 모으고, 없는 행은 기본값으로 채운다.
    src_keys = []
    with open(a.src) as f:
        for i, ln in enumerate(f):
            for k in json.loads(ln):
                if k not in src_keys:
                    src_keys.append(k)
            if i >= 50_000:
                break
    keys = [REN.get(k, k) for k in src_keys]
    back = {REN.get(k, k): k for k in src_keys}      # 배포 열 -> jsonl 키
    # 접촉 없는 판에는 guard 키가 아예 없다. 그래도 fixed 열을 0 으로 낸다 --
    # 두 판을 나란히 놓는 것이 목적이라 열이 갈리면 대조가 안 된다.
    if "fixed" not in keys:
        keys.append("fixed")
        back["fixed"] = "guard"
    sch = schema_from(keys)

    w = None
    buf = {k: [] for k in keys}
    n = 0
    try:
        with open(a.src) as f:
            for ln in f:
                d = json.loads(ln)
                for k in keys:
                    v = d.get(back[k])
                    if v is None:
                        v = "" if k == "task" else 0
                    buf[k].append(v)
                n += 1
                if n % a.chunk == 0:
                    t = pa.table({k: pa.array(v, type=sch.field(k).type)
                                  for k, v in buf.items()}, schema=sch)
                    w = w or pq.ParquetWriter(dst, sch, compression="zstd")
                    w.write_table(t)
                    buf = {k: [] for k in keys}
        if any(buf.values()):
            t = pa.table({k: pa.array(v, type=sch.field(k).type)
                          for k, v in buf.items()}, schema=sch)
            w = w or pq.ParquetWriter(dst, sch, compression="zstd")
            w.write_table(t)
    finally:
        if w:
            w.close()
    print(f"{n:,}행 -> {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
    print(f"열: {keys}")


if __name__ == "__main__":
    main()
