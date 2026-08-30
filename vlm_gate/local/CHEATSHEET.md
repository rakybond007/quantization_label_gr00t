# 상태 확인 한 줄 모음

파일시스템을 훑을 수 없으므로 아래를 쓴다. 전부 `dev run '...'` 안에 넣어 실행한다.
`WS=~/quantization_agent_workspace` 로 가정.

## 잡

```bash
squeue -u $USER -o "%.8i %.9P %.28j %.2t %.10M %R"
sacct -X -u $USER -S $(date -d '1 day ago' +%F) --format=JobID,JobName%30,State,Elapsed,ExitCode
```

## 라벨링 진행률 (샤드 파일 줄 수 합계)

```bash
cd $WS/vlm_gate/output/_gate_distill && \
  for i in $(seq 0 15); do f=v6b_phase5_s16_$i.jsonl; \
  [ -f $f ] && printf "%s " $(wc -l < $f) || printf "MISSING(%d) " $i; done; echo
```

## 라벨 parquet 요약

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate quant_gate_eval && \
python -c "
import pandas as pd
d=pd.read_parquet('$WS/assets/labels/robocasa/v6b_phase5_1call_full.parquet')
print(d.shape)
print(d[['p_yes','p_raw']].describe().round(3))
print('계산플래그', {c:round(d[c].mean(),4) for c in d.columns if c.startswith('c_')})
print('VLM문항', {c:round(d[c].mean(),3) for c in d.columns if c.startswith('q_')})
"
```

## 평가 결과 (성공률·스텝)

`prediction.txt`의 `^episode` 줄만 세고 **에피소드 인덱스로 중복 제거**한다.
(끝에 붙은 요약 줄을 정규식으로 잘못 집는 사고가 있었다.)

```bash
python -c "
import sys,re,collections
p=sys.argv[1]; ep={}
for l in open(p):
    if not l.startswith('episode'): continue
    m=re.match(r'episode\s+(\d+).*?success[=: ]+(\d+).*?steps[=: ]+(\d+)',l)
    if m: ep[int(m.group(1))]=(int(m.group(2)),int(m.group(3)))
v=list(ep.values())
print(f'{len(v)}ep  성공률 {sum(a for a,_ in v)/max(len(v),1):.3f}  전체평균스텝 {sum(b for _,b in v)/max(len(v),1):.1f}')
s=[b for a,b in v if a]; print(f'  성공한 것만 평균스텝 {sum(s)/max(len(s),1):.1f} (n={len(s)})')
" <prediction.txt 경로>
```

성공만 뽑은 스텝과 전체 평균 스텝을 **섞어 비교하지 말 것.** 한 번 틀렸다.

## 최근 슬럼 로그 꼬리

```bash
ls -t $WS/vlm_gate/out/*.out | head -3 | while read f; do echo "[$f]"; tail -5 "$f"; done
```

## 타일 개수 (디렉터리를 훑지 않고)

```bash
wc -l < $WS/vlm_gate/output/_gate_distill/tiles_manifest.txt
```

## GPU 여유

```bash
sinfo -p sjw_alinlab,background -o "%.12P %.6a %.10l %.6D %.6t %N" | head
```
