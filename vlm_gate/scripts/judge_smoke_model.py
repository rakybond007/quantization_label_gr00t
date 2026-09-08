"""새 판정 모델이 이 파이프라인에 쓸 수 있는지 **네 단계로** 확인한다.

    python vlm_gate/scripts/judge_smoke_model.py --model Qwen/Qwen3.8-27B

클래스가 있다고 그 체크포인트가 도는 것이 아니다. 막히는 자리를 미리 갈라 둔다 --
어디서 막혔는지 알아야 환경을 새로 만들지 코드를 고칠지 정할 수 있다.

    1  설정      config 를 읽고 어느 클래스로 갈지 본다
    2  프로세서  AutoProcessor 가 뜨고 이미지를 받는가
    3  가중치    모델이 GPU 에 올라가는가
    4  답변      이미지 셋 + 5문항 프롬프트에 형식대로 답하는가

**4단계가 본체다.** 1~3 이 되어도 형식대로 못 답하면 못 쓴다. Cosmos 는
`'A) 3\\nB) 1\\nC) 1\\nD) 4\\nE) 3'` 처럼 군더더기 없이 답한다.
"""
import argparse
import sys
import traceback

STEP = []


def step(n, name):
    def deco(fn):
        STEP.append((n, name, fn))
        return fn
    return deco


@step(1, "설정")
def s1(a, st):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust)
    arch = getattr(cfg, "architectures", None)
    print(f"    model_type   {getattr(cfg, 'model_type', '?')}")
    print(f"    architectures {arch}")
    tc = getattr(cfg, "text_config", None)
    if tc is not None:
        print(f"    text  layers {getattr(tc, 'num_hidden_layers', '?')} "
              f"hidden {getattr(tc, 'hidden_size', '?')}")
    vc = getattr(cfg, "vision_config", None)
    if vc is not None:
        ks = [k for k in ("depth", "hidden_size", "deepstack_visual_indexes")
              if hasattr(vc, k)]
        print(f"    vision {' '.join(f'{k}={getattr(vc, k)}' for k in ks)}")
    st["cfg"] = cfg
    return True


@step(2, "프로세서")
def s2(a, st):
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(a.model, trust_remote_code=a.trust)
    print(f"    {type(p).__name__}")
    print(f"    tokenizer {type(p.tokenizer).__name__} "
          f"vocab {len(p.tokenizer)}")
    # 등급 토큰이 한 글자로 잡히는지 -- 여러 토큰이면 분포를 못 읽는다
    ids = [p.tokenizer.encode(str(g), add_special_tokens=False) for g in range(1, 6)]
    print(f"    등급 1~5 토큰 {ids}"
          + ("" if all(len(i) == 1 for i in ids) else "   [!] 한 토큰이 아니다"))
    st["proc"] = p
    return all(len(i) == 1 for i in ids)


@step(3, "가중치")
def s3(a, st):
    import torch
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=a.trust).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"    {type(m).__name__}  파라미터 {n/1e9:.1f}B")
    print(f"    device_map {getattr(m, 'hf_device_map', {}) and '있음' or '없음'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"    GPU{i} {torch.cuda.memory_allocated(i)/1e9:.1f}GB / "
                  f"{torch.cuda.get_device_properties(i).total_memory/1e9:.0f}GB")
    st["model"] = m
    return True


@step(4, "답변")
def s4(a, st):
    import numpy as np
    import torch
    from PIL import Image
    m, p = st["model"], st["proc"]
    # 회색 판 셋. 내용이 아니라 **형식**을 보는 것이므로 그림은 아무거나 된다.
    imgs = [Image.fromarray(np.full((256, 256, 3), v, np.uint8))
            for v in (90, 120, 150)]
    ask = ("Answer each check on its own line as \"A) 3\", in order, nothing "
           "else -- one digit from 1 to 5 per check.\n"
           "A) Is the arm near an object?\nB) Is the gripper open?\n"
           "C) Is the arm moving fast?\nD) Is the scene cluttered?\n"
           "E) Is the arm idle?\nAnswer:")
    msg = [{"role": "user", "content":
            [{"type": "image", "image": im} for im in imgs]
            + [{"type": "text", "text": ask}]}]
    enc = p.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                return_dict=True, return_tensors="pt")
    enc = {k: (v.to(m.device) if hasattr(v, "to") else v) for k, v in enc.items()}
    with torch.inference_mode():
        out = m.generate(**enc, max_new_tokens=64, do_sample=False)
    txt = p.tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip()
    print(f"    원문 {txt!r}")
    lines = [l for l in txt.splitlines() if ")" in l]
    good = len(lines) == 5 and all(
        l.split(")")[1].strip()[:1].isdigit() for l in lines if ")" in l)
    print(f"    5칸 형식 {'맞음' if good else '어긋남'} (줄 {len(lines)})")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--trust", action="store_true",
                    help="trust_remote_code. 저장소 코드를 실행하므로 필요할 때만")
    ap.add_argument("--upto", type=int, default=4)
    a = ap.parse_args()

    st, failed = {}, None
    for n, name, fn in STEP:
        if n > a.upto:
            break
        print(f"[{n}] {name}")
        try:
            ok = fn(a, st)
        except Exception:
            traceback.print_exc()
            failed = (n, name)
            break
        if not ok:
            failed = (n, name)
            break
    print()
    if failed:
        n, name = failed
        print(f"=> {n}단계 '{name}' 에서 막혔다.")
        print({1: "   설정을 못 읽는다. transformers 가 이 model_type 을 모른다 --\n"
                  "   새 환경에 최신 transformers 를 깔거나 --trust 가 필요하다.",
               2: "   프로세서가 없거나 등급이 한 토큰이 아니다. 뒤엣것이면 분포를\n"
                  "   못 읽으므로 정수 등급만 쓰게 되고, 그건 신뢰도가 뭉개진다.",
               3: "   가중치가 안 올라간다. 메모리거나 구현이 체크포인트와 안 맞는다.\n"
                  "   GPU 를 늘려 보고, 그래도 안 되면 새 환경에 최신 transformers.",
               4: "   형식대로 못 답한다. 이 파이프라인에 못 쓴다 -- 프롬프트를\n"
                  "   바꾸는 것은 비교의 전제를 깨는 것이라 하지 않는다."}[n])
        return 1
    print("=> 네 단계 다 통과. 판정 서버로 띄워도 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
