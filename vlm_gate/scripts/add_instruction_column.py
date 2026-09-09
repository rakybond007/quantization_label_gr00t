"""이미 만들어 둔 배속 라벨 parquet 에 **지시문 열**을 붙인다.

    python vlm_gate/scripts/add_instruction_column.py robocasa \
        old.parquet new.parquet

라벨링을 다시 돌리지 않는다 -- 비싼 것은 판정기를 돌린 부분이고, 지시문은
`episodes.jsonl` 에 이미 있다. 에피소드 번호로 붙이기만 하면 된다.

**행 묶음 단위로 흘려 쓴다.** 204만 행을 한 번에 DataFrame 으로 세우면
메모리에서 죽는다 -- 만들 때 한 번 겪었다.

지시문을 라벨 파일에 넣는 이유는 `ratio_label.ep_to_instruction` 에 적어 두었다.
요약하면 이 라벨을 쓰는 머리가 받는 것이 태스크 이름이 아니라 이 문장이고,
배포본만 받은 사람에게는 원본 데이터셋을 되짚을 길이 없기 때문이다.
"""
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from ratio_label import ep_to_instruction


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 1
    bench, src, dst = sys.argv[1:4]

    ep2i = ep_to_instruction(bench)
    print(f"지시문 {len(set(ep2i.values()))}종 · 에피소드 {len(ep2i)}개")

    f = pq.ParquetFile(src)
    if "instruction" in f.schema_arrow.names:
        print("[!] 이미 instruction 열이 있다. 아무것도 안 한다.")
        return 1

    w, sch = None, None
    n, miss = 0, set()
    for batch in f.iter_batches(batch_size=200_000):
        t = pa.Table.from_batches([batch])
        eps = t.column("episode_index").to_pylist()
        for e in eps:
            if e not in ep2i:
                miss.add(e)
        col = pa.array([ep2i.get(e) for e in eps], type=pa.string())
        # `task` 바로 뒤에 넣는다 -- 둘이 붙어 있어야 읽는 사람이 안 헷갈린다.
        at = t.schema.get_field_index("task")
        t = t.add_column(at + 1 if at >= 0 else t.num_columns, "instruction", col)
        if w is None:
            sch = t.schema
            w = pq.ParquetWriter(dst, sch, compression="zstd")
        w.write_table(t.cast(sch))
        n += t.num_rows
        print(f"  {n:,}행", flush=True)
    if w is not None:
        w.close()

    out = {"rows": n, "instructions": len(set(ep2i.values())),
           "episodes_missing_instruction": len(miss)}
    if miss:
        # 조용히 None 을 넣고 넘어가면 학습 쪽에서 찾기 어렵다.
        out["example_missing"] = sorted(miss)[:5]
        print(f"\n[!] 지시문을 못 찾은 에피소드 {len(miss)}개 -- 그 행은 None 이다")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
