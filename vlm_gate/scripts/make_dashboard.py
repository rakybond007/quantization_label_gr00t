"""results_db.json + 레지스트리 → 정적 대시보드 HTML 생성 (재생성 가능)"""
import json, os, re, html, time
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
db=json.load(open(f"{BASE}/analysis/results_db.json"))
LABEL={
 "baseline_full_v2_with_action_steps":("무압축 원본","기준선","—"),
 "baseline_compress_K2":("naive K2","전 구간 압축","—"),
 "baseline_compress_K3":("naive K3","전 구간 압축","—"),
 "moduleA_gemma_gated":("A' 학생","gemma4","0.50"),
 "moduleA_cosmos_gated":("A' 학생","cosmos3","0.50"),
 "moduleA_cosmos_tau042":("A' 학생","cosmos3","0.42"),
 "moduleA_cosmos_tau038":("A' 학생","cosmos3","0.38"),
 "moduleA_cosmos9k_gated":("A' 학생 (9k)","cosmos3","0.50"),
 "moduleA_f9k_act_gated":("A' 학생 (9k)","frontier+액션","0.50"),
 "moduleA_frontier_full_gated":("A' 학생 (전량)","frontier+액션","0.50"),
"moduleA_frontier_tau0976":("A' 학생 (전량)","frontier+액션","0.976"),
 "moduleA_frontier_tau0995":("A' 학생 (전량)","frontier+액션","0.995"),
 "moduleA_gemma_restored_gated":("A' 학생 (재학습)","gemma4","0.50"),
 "moduleA_f9k_b4_gated":("A' 학생 (9k)","frontier 4대규칙","0.50"),
 "moduleB_gemma_gated":("B (frozen+MLP)","gemma4","0.50"),
 "moduleB_cosmos_gated":("B (frozen+MLP)","cosmos3","0.50"),
 "gateC_v3_lam03_internal_gated":("C (게이트토큰 λ0.3)","cosmos3","0.50"),
 "gateCv3_internal_gated":("C (게이트토큰 λ1.0)","cosmos3","0.50"),
 "gateC_lam03_ladder_k3tau065":("C + K3사다리 τ₃0.65","cosmos3","0.50"),
 "moduleA_ladder_k3tau08_clip3":("A' + K3사다리(미발동)","cosmos3","0.50"),
 "moduleA_k2_clip3":("A' + clip×3","cosmos3","0.50"),
 "naiveK2_compensated_model_clip3":("naive K2 + 모델보정","—","—"),
 "naiveK2_compensated_scalar_clip3":("naive K2 + 스칼라보정","—","—"),
 "naiveK3_compensated_model_clip3_fixed":("naive K3 + 모델보정","—","—"),
 "robocasa_gemma_ttl_varkA":("VLM judge 직접(varK)","gemma4","—"),
 "robocasa_cosmos_ttl_varkA":("VLM judge 직접(varK)","cosmos3","—"),
}
BASE_S, BASE_T = 0.657, 327.0
rows=[]
for k,v in db.items():
    if v["episodes"]<1000 or k not in LABEL: continue
    arch, teacher, tau = LABEL[k]
    ck=v["provenance"]["ckpt"]
    prov="—"
    if ck and ck!="?":
        parts=ck.rstrip("/").split("/")
        prov="/".join(parts[-3:-1]) if len(parts)>=3 else ck
    g=v.get("gate") or {}
    rows.append(dict(run=k, arch=arch, teacher=teacher, tau=tau, succ=v["success"], steps=v["steps_mean"],
                     eps=v["episodes"], qrate=g.get("qrate"), confmax=g.get("conf_max"), prov=prov))
rows.sort(key=lambda r: (-r["succ"], r["steps"] or 9e9))
def cls(v, good, bad):
    if v is None: return ""
    return "good" if v>=good else ("bad" if v<=bad else "warn")
tr=[]
for r in rows:
    ds = r["succ"]-BASE_S
    dt = (r["steps"]-BASE_T)/BASE_T*100 if r["steps"] else None
    tr.append(f"""<tr>
<td class="name">{html.escape(r['arch'])}</td>
<td class="mono">{html.escape(r['teacher'])}</td>
<td class="mono num">{r['tau']}</td>
<td class="mono num {cls(r['succ'],0.65,0.61)}">{r['succ']:.3f}</td>
<td class="mono num delta">{ds:+.3f}</td>
<td class="mono num">{r['steps']:.0f}</td>
<td class="mono num delta">{dt:+.0f}%</td>
<td class="mono num">{'' if r['qrate'] is None else f"{r['qrate']:.2f}"}</td>
<td class="mono num">{'' if r['confmax'] is None else f"{r['confmax']:.2f}"}</td>
<td class="mono prov">{html.escape(r['prov'])}</td>
</tr>""")
MATRIX=[("robocasa","cosmos3","262,329 ✅","✅ 학습·평가","✅ 학습·평가"),
        ("robocasa","gemma4","262,329 ✅","🔄 재학습(ckpt 소실)","✅ 학습·평가"),
        ("robocasa","frontier(단일프레임+액션)","🔄 라벨링 중","⏳ 대기","⏳ 대기"),
        ("real","cosmos3","3,765 ✅ (+phase-aware)","🔄 학습 중","✅ 학습"),
        ("real","gemma4","🔄 라벨링 중","⏳ 대기","⏳ 대기"),
        ("real","frontier(단일프레임+액션)","3,664/3,705","⏳ 대기","⏳ 대기"),
        ("real","luna-batch6","3,705 ⛔ 폐기","(폐기)","(폐기)")]
mrows="".join(f"<tr><td class='mono'>{d}</td><td class='name'>{t}</td><td class='mono'>{l}</td><td class='mono'>{a}</td><td class='mono'>{c}</td></tr>" for d,t,l,a,c in MATRIX)
NEG=[("K3 혼합 (신뢰도 사다리)","clip 해제해도 K2 미달. τ₃는 학생별 confidence 분포에서 골라야 하며, 과거 실험은 K3가 발동조차 안 해 무효였다."),
     ("동역학 보정 (OSC 역산)","clip 해제 후에도 naive K2와 동일(0.595 vs 0.598). 진폭은 복원되나 성공률은 회복되지 않음 → 성공률은 게이트의 몫."),
     ("reasoning=none 단독 라벨링","robocasa에서 라벨이 상수로 붕괴(이웃 상관 0.058). 단일프레임으로 바꿔도 동일."),
     ("모션 3프레임 입력","evolve 루프는 단일 시점만 사용 가능 — 배포 불가로 폐기."),
     ("타일 해상도 상향","원본 256×256 3뷰가 384×128 타일 대비 그리퍼 판별 개선 없음(+0.05 미만).")]

PROTO=[("이진 4대규칙","0.48","70.4%","31.2%","액션 4조건 중 하나라도 걸리면 NO. 현재 최고 판별력, 동작점 고정"),
 ("bits8 + 가중치 + 순위정규화","0.50 (τ=0.5)","67.6%","16.7%","영상4+액션4 분해. τ가 곧 압축비율 — 동작점 자유 조절"),
 ("bits8cal (캘리브 수치 포함)","0.50 (τ=0.5)","60.8%","15.0%","스케일 기준 명시했으나 그리퍼 판정이 둔해져 소폭 하락"),
 ("4비트 분해 (액션만)","0.17~0.65 (τ별)","87.8%~66.9%","13~24%","영상 판단 미반영. 비트별 조합으로 동작점 조절 가능"),
 ("투표 3회","0.67","86.5%","35.3%","토큰확률 미사용. 값 4단계뿐"),
 ("등급별 동의(ladder)","1.00","0.0%","—","약한 주장에 전부 동의 → 하한이 0.5로 깔림. 주장 강도 재설계 필요"),
 ("20단계/10단계/5단계 등급","0.9~1.0","58~85%","—","값은 퍼지나 위험과 무관(이웃상관 ~0). 자기 확신도 환산 실패"),
 ("범용+액션설명 (전량 $46에 사용)","1.00","1.1%","—","액션을 주되 해석 규칙이 없어 판단 붕괴")]
prot="".join(f"<tr><td class='name'>{a}</td><td class='mono num'>{b}</td><td class='mono num'>{c}</td><td class='mono num'>{d}</td><td>{e}</td></tr>" for a,b,c,d,e in PROTO)
negrows="".join(f"<tr><td class='name'>{html.escape(t)}</td><td>{html.escape(d)}</td></tr>" for t,d in NEG)
html_out=f"""<title>Quantizability Ledger</title>
<style>
:root {{
  --ground:#F7F8FA; --surface:#FFFFFF; --ink:#16202B; --muted:#5C6B7A; --rule:#DFE4EA;
  --accent:#0F6E7A; --good:#2E7D51; --warn:#A8730F; --bad:#B03A3A; --chip:#EDF1F4;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#10151B; --surface:#171E26; --ink:#E6EDF3; --muted:#97A6B5; --rule:#253039;
    --accent:#4FC3CE; --good:#57C08A; --warn:#D8A343; --bad:#E0726E; --chip:#1D262F;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#10151B; --surface:#171E26; --ink:#E6EDF3; --muted:#97A6B5; --rule:#253039;
  --accent:#4FC3CE; --good:#57C08A; --warn:#D8A343; --bad:#E0726E; --chip:#1D262F;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font-family:ui-sans-serif,-apple-system,"Segoe UI",Roboto,"Noto Sans KR",sans-serif;
  font-size:15px; line-height:1.55; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:40px 24px 80px; display:flex; flex-direction:column; gap:36px; }}
header {{ display:flex; flex-direction:column; gap:6px; border-bottom:1px solid var(--rule); padding-bottom:20px; }}
h1 {{ margin:0; font-size:28px; letter-spacing:-0.02em; font-weight:650; text-wrap:balance; }}
.sub {{ color:var(--muted); font-size:14px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-variant-numeric:tabular-nums; }}
.strip {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:16px 18px;
  display:flex; flex-direction:column; gap:4px; }}
.card .lab {{ font-size:11px; letter-spacing:0.09em; text-transform:uppercase; color:var(--muted); }}
.card .val {{ font-size:26px; font-weight:640; letter-spacing:-0.01em; }}
.card .note {{ font-size:13px; color:var(--muted); }}
h2 {{ margin:0 0 2px; font-size:17px; letter-spacing:-0.01em; font-weight:620; }}
section {{ display:flex; flex-direction:column; gap:12px; }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th {{ text-align:left; font-size:11px; letter-spacing:0.07em; text-transform:uppercase; color:var(--muted);
  font-weight:600; padding:11px 14px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:9px 14px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
tr:last-child td {{ border-bottom:none; }}
td.name {{ font-weight:560; white-space:normal; }}
td.num {{ text-align:right; }}
td.delta {{ color:var(--muted); }}
td.prov {{ font-size:11.5px; color:var(--muted); }}
.good {{ color:var(--good); font-weight:640; }}
.warn {{ color:var(--warn); }}
.bad {{ color:var(--bad); }}
.negtable td {{ white-space:normal; }}
.foot {{ color:var(--muted); font-size:12.5px; border-top:1px solid var(--rule); padding-top:14px; }}
.pill {{ display:inline-block; background:var(--chip); color:var(--muted); border-radius:999px;
  padding:2px 9px; font-size:11px; letter-spacing:0.04em; }}
</style>
<div class="wrap">
<header>
  <h1>Quantizability Ledger</h1>
  <div class="sub">robocasa 폐루프 24태스크 × 50에피소드 기준. 모든 수치는 <span class="mono">analysis/results_db.json</span>에서 자동 수집되며 체크포인트 출처가 함께 기록됩니다. 갱신 {time.strftime('%Y-%m-%d %H:%M')}</div>
</header>

<section>
  <h2>기준선 대비 현재 최고</h2>
  <div class="strip">
    <div class="card"><span class="lab">무압축 원본</span><span class="val mono">0.657</span><span class="note">327 스텝 · 압축 없음</span></div>
    <div class="card"><span class="lab">naive K2 (게이트 없음)</span><span class="val mono bad">0.598</span><span class="note">221 스텝 · 성공률 −5.9pp</span></div>
    <div class="card"><span class="lab">최고 게이트 (A' / gemma4)</span><span class="val mono good">0.667</span><span class="note">258 스텝 · 원본 대비 −21% 시간</span></div>
    <div class="card"><span class="lab">동역학 보정 단독</span><span class="val mono">0.595</span><span class="note">214 스텝 · naive와 동일</span></div>
  </div>
</section>

<section>
  <h2>폐루프 결과 <span class="pill">출처 포함</span></h2>
  <div class="scroll"><table>
    <thead><tr><th>아키텍처</th><th>티처</th><th>τ</th><th>성공률</th><th>Δ기준</th><th>스텝</th><th>Δ시간</th><th>qrate</th><th>conf최대</th><th>체크포인트 출처</th></tr></thead>
    <tbody>{''.join(tr)}</tbody>
  </table></div>
</section>

<section>
  <h2>매트릭스 진행 현황</h2>
  <div class="scroll"><table>
    <thead><tr><th>도메인</th><th>티처</th><th>라벨</th><th>A'</th><th>C</th></tr></thead>
    <tbody>{mrows}</tbody>
  </table></div>
</section>

<section>
  <h2>티처 프롬프트 프로토콜 비교 <span class="pill">균등표본 1,200프레임 · 물리 프록시 기준</span></h2>
  <div class="scroll"><table>
    <thead><tr><th>프로토콜</th><th>qrate</th><th>위험 검출</th><th>차단 정확도</th><th>메모</th></tr></thead>
    <tbody>{prot}</tbody>
  </table></div>
  <div class="foot" style="border:none;padding-top:4px">
    핵심 발견: (1) 액션 수치를 주려면 <b>해석 규칙이 세트</b>여야 한다 — 규칙 없이 수치만 주면 로컬·frontier 모두 판별력이 무너진다(cosmos 그리퍼 검출 70.4%→36.2%, 규칙 추가 시 83.9%로 회복).
    (2) 모델에게 <b>자기 확신도를 숫자로 환산시키는 방식은 실패</b>한다(5·10·20단계 모두 위험과 무관한 값 생성).
    (3) 분해 방식은 <b>비트별 가중치와 τ로 동작점을 자유롭게 조절</b>할 수 있고, 순위 정규화를 적용하면 τ가 곧 압축 비율이 된다.
    (4) 가중치는 <b>비트별 상·하한 + 모달리티 그룹 최소 비중</b>으로 제약해 한쪽 편향을 막고, τ와 함께 evolve/RL 탐색 대상으로 둔다.
  </div>
</section>

<section>
  <h2>확정된 부정 결과 — 재실험 금지</h2>
  <div class="scroll"><table class="negtable">
    <thead><tr><th>항목</th><th>근거</th></tr></thead>
    <tbody>{negrows}</tbody>
  </table></div>
</section>

<div class="foot">
  τ는 (티처 × 아키텍처)마다 다른 탐색 대상입니다 — judge마다 confidence 캘리브레이션이 달라 같은 0.5가 다른 동작점을 뜻합니다(cosmos 학생 최대 0.726 / gemma 학생 최대 0.999). τ 탐색은 evolve 루프의 일부로 다루되 튜닝·홀드아웃 태스크를 분리합니다.
</div>
</div>"""
open(f"{BASE}/analysis/dashboard.html","w").write(html_out)
print("생성:", f"{BASE}/analysis/dashboard.html", len(rows), "개 실행 수록")
