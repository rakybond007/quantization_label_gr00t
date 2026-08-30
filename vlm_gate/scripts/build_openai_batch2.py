import json, base64, os, random
BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
G = open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
d = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
names = os.listdir(d)
random.seed(1); random.shuffle(names)
sample = names[:504]  # 84 requests x 6 frames
out = f"{BASE}/output/_gate_distill/openai_batch/batch_small2.jsonl"
tail = ("For EACH frame in order: p_yes in [0,1] = probability the next ~1 second of robot motion "
        "can be compressed (half control rate) without changing the outcome.\n"
        f"Guidance:\n{G}\n"
        "Output ONLY a JSON array of 6 numbers.")
with open(out, "w") as f:
    for i in range(0, len(sample), 6):
        chunk = sample[i:i+6]
        content = []
        for name in chunk:
            b = base64.b64encode(open(f"{d}/{name}", "rb").read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
        content.append({"type": "text", "text":
            "You are judging 6 frames (attached, in order) from RoboCasa simulated kitchen "
            "manipulation demos. Panels: left cam, right cam, wrist cam.\n" + tail})
        req = {"custom_id": "|".join(n[:-4] for n in chunk), "method": "POST",
               "url": "/v1/chat/completions",
               "body": {"model": "gpt-5.6-luna", "max_completion_tokens": 64,
                        "reasoning_effort": "none",
                        "messages": [{"role": "user", "content": content}]}}
        f.write(json.dumps(req) + "\n")
print("done", round(os.path.getsize(out)/1e6, 1), "MB")
