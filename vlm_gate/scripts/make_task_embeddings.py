"""robocasa 지시문 → MiniLM 임베딩 (transformers만 사용, 추가 설치 없음)"""
import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
out=sys.argv[1] if len(sys.argv)>1 else "output/_gate_distill/robocasa_task_embeddings.npz"
tasks=set()
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l)
    for t in d.get("tasks",[]):
        if isinstance(t,str) and len(t.split())>1 and t!="Valid": tasks.add(t)
tasks=sorted(tasks); print("지시문", len(tasks), "종")
mid="sentence-transformers/all-MiniLM-L6-v2"
tok=AutoTokenizer.from_pretrained(mid); mdl=AutoModel.from_pretrained(mid).eval()
embs=[]
with torch.no_grad():
    for i in range(0,len(tasks),64):
        b=tok(tasks[i:i+64], padding=True, truncation=True, return_tensors="pt")
        o=mdl(**b).last_hidden_state
        m=b["attention_mask"].unsqueeze(-1).float()
        e=(o*m).sum(1)/m.sum(1)                      # mean pooling
        e=torch.nn.functional.normalize(e, dim=1)
        embs.append(e.numpy().astype(np.float32))
emb=np.concatenate(embs)
np.savez(out, tasks=np.array(tasks, dtype=object), emb=emb)
print("저장:", out, emb.shape)
