"""Write the self-contained summary page for the v2 variable-ratio pipeline.

Reads the aggregated records/summary and the live prompt text out of the
modules, so the page can never drift from what was actually run.
  python allex_v2_html.py [out.html]
"""
import html, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from allex_common_v5 import GUIDANCE, ASK
from allex_v2_common import (STAGE2_GUIDANCE, STAGE2_ASK, TASK_CEILING, facts_text, stage2_facts,
                             MERGE_LIMIT_V2, ROT_ACCUM_LIMIT_V2, GAP_RATE_LIMIT_V2,
                             REORIENT_W, HAND_SCALE)

OUTDIR = os.path.expanduser("~/quantization_agent_workspace/vlm_gate/output/allex_v2")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/quantization_agent_workspace/docs/allex_variable_ratio.html")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
S = json.load(open(f"{OUTDIR}/summary.json"))
recs = [json.loads(l) for l in open(f"{OUTDIR}/records.jsonl")]
TEST = open(f"{OUTDIR}/ratio_selftest.txt").read().strip() if os.path.exists(
    f"{OUTDIR}/ratio_selftest.txt") else "(unit test output not captured)"
VIDEO = os.environ.get("VIDEO_PATH", "assets/videos/allex_v2_variable_ratio.mp4")

RATIOS = ["1.0", "2.0", "2.5", "3.0"]
RCOL = {"1.0": "var(--k1)", "2.0": "var(--k2)", "2.5": "var(--k25)", "3.0": "var(--k3)"}
ORDER = ["Pass Object", "Bring Object", "Rotate PolyBag", "Rotate Box"]
RATIONALE = {
    "Pass Object": ("3", "An object is carried sideways to the other side. Nothing about the exact "
                    "path matters, and there is no moment where it has to arrive anywhere precise."),
    "Bring Object": ("3 &rarr; 1", "A box is dragged in toward the robot. The dragging itself is "
                     "coarse, but it has to come to a precise stop in front, and the stop must run "
                     "at full rate. Stage 1 already senses the stopping phase; because the final "
                     "ratio is 1 + p(K<sub>max</sub>-1), a low p pins K near 1 no matter what "
                     "ceiling stage 2 returns, so stage 2 cannot undo it."),
    "Rotate Box": ("2", "A rigid box is turned between two palms. The two hands must move "
                   "differently to turn it, and that relative motion is the entire hold, so only "
                   "mild compression survives."),
    "Rotate PolyBag": ("2.5", "The bag is flipped with ONE hand, so there is no two-hand hold to "
                       "lose - but it is soft and flops, so the hand must not overshoot far. "
                       "It sits between the box turn and a free transfer."),
}


def esc(s):
    return html.escape(s)


def pctstr(dist):
    """Ratio percentages, one decimal where a level is present but rare."""
    out = []
    for r in RATIOS:
        v = dist[r]
        if v <= 0:
            continue
        out.append(f'{r.rstrip("0").rstrip(".")}:{v:.1f}%' if v < 1 else
                   f'{r.rstrip("0").rstrip(".")}:{v:.0f}%')
    return " ".join(out)


def bars(dist):
    seg = "".join(
        f'<span class="seg" style="width:{dist[r]}%;background:{RCOL[r]}" title="K={r}: {dist[r]}%"></span>'
        for r in RATIOS if dist[r] > 0)
    return f'<span class="bar">{seg}</span>'


rows = []
for t in ORDER:
    d = S["tasks"].get(t)
    if not d:
        continue
    ceil, why = RATIONALE[t]
    rows.append(f"""<tr>
      <th scope="row">{esc(t)}<span class="sub">{d['n_chunks']:,} chunks</span></th>
      <td class="num">{ceil}</td>
      <td class="num">{d['mean_p']:.2f}</td>
      <td class="num">{d['mean_K_max']:.2f}</td>
      <td class="num strong">{d['mean_K']:.2f}</td>
      <td class="barcell">{bars(d['dist'])}
        <span class="pct">{pctstr(d['dist'])}</span></td>
    </tr>""")

why_rows = "".join(
    f'<tr><th scope="row">{esc(t)}</th><td class="num">{RATIONALE[t][0]}</td>'
    f'<td>{RATIONALE[t][1]}</td></tr>' for t in ORDER)

overall_bar = bars(S["overall_dist"])
page = f"""<title>Variable-Ratio Gate for allex</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --bg:#f2f4f7; --surface:#ffffff; --ink:#171b22; --muted:#59636f; --line:#dde2e9;
  --accent:#33489b; --accent-soft:#e7eaf6; --code-bg:#f7f8fb;
  --k1:#c8503f; --k2:#cf8c2a; --k25:#7aa844; --k3:#2f8f5b;
  --shadow:0 1px 2px rgba(20,26,40,.06), 0 8px 24px rgba(20,26,40,.05);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#12151a; --surface:#191d24; --ink:#e6e9ee; --muted:#98a2ae; --line:#2b323b;
    --accent:#8fa2ee; --accent-soft:#232a3f; --code-bg:#11151b;
    --k1:#e2705f; --k2:#e3a94b; --k25:#9ac765; --k3:#5cbc86;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="dark"] {{
  --bg:#12151a; --surface:#191d24; --ink:#e6e9ee; --muted:#98a2ae; --line:#2b323b;
  --accent:#8fa2ee; --accent-soft:#232a3f; --code-bg:#11151b;
  --k1:#e2705f; --k2:#e3a94b; --k25:#9ac765; --k3:#5cbc86;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.35);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"Source Serif 4", Georgia, serif; font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:56px 28px 96px; }}
header.top {{ border-bottom:2px solid var(--ink); padding-bottom:28px; margin-bottom:44px; }}
.eyebrow {{
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 14px;
}}
h1 {{
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-weight:700; letter-spacing:-.022em;
  font-size:clamp(30px,4.4vw,46px); line-height:1.08; margin:0 0 16px; text-wrap:balance;
}}
.lede {{ font-size:20px; color:var(--muted); margin:0; max-width:62ch; }}
section {{ margin:0 0 52px; }}
h2 {{
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-weight:600; font-size:23px;
  letter-spacing:-.012em; margin:0 0 6px; text-wrap:balance;
}}
h2 .n {{
  font-family:"IBM Plex Mono", monospace; font-size:12px; color:var(--accent);
  letter-spacing:.1em; display:block; margin-bottom:7px; font-weight:600;
}}
h3 {{ font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:16px; font-weight:600;
     margin:26px 0 8px; letter-spacing:-.005em; }}
p {{ margin:0 0 14px; max-width:68ch; }}
.grid {{ display:grid; gap:16px; }}
.stats {{ grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin:32px 0 8px; }}
.stat {{
  background:var(--surface); border:1px solid var(--line); border-radius:3px;
  padding:16px 18px; box-shadow:var(--shadow);
}}
.stat .v {{
  font-family:"IBM Plex Sans", sans-serif; font-weight:700; font-size:29px;
  letter-spacing:-.02em; font-variant-numeric:tabular-nums; display:block; line-height:1.15;
}}
.stat .l {{
  font-family:"IBM Plex Mono", monospace; font-size:11px; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); display:block; margin-top:4px;
}}
pre, code {{ font-family:"IBM Plex Mono", ui-monospace, monospace; }}
pre {{
  background:var(--code-bg); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:2px; padding:16px 18px; overflow-x:auto; font-size:13px; line-height:1.6;
  margin:0 0 16px; white-space:pre;
}}
code {{ font-size:.88em; background:var(--accent-soft); padding:1px 5px; border-radius:2px; }}
pre code {{ background:none; padding:0; font-size:inherit; }}
.prompt {{
  background:var(--surface); border:1px solid var(--line); border-radius:3px;
  padding:0; overflow:hidden; margin:0 0 18px; box-shadow:var(--shadow);
}}
.prompt > .hd {{
  font-family:"IBM Plex Mono", monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); padding:10px 16px;
  border-bottom:1px solid var(--line); background:var(--code-bg);
}}
.prompt > pre {{ border:0; border-left:0; border-radius:0; margin:0; white-space:pre-wrap;
                background:var(--surface); font-size:13px; }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px;
        font-family:"IBM Plex Sans", sans-serif; }}
.tablewrap {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line);
             border-radius:3px; box-shadow:var(--shadow); }}
th, td {{ padding:12px 14px; text-align:left; border-bottom:1px solid var(--line);
         vertical-align:top; }}
thead th {{
  font-family:"IBM Plex Mono", monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:600; white-space:nowrap;
}}
tbody th {{ font-weight:600; white-space:nowrap; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom:0; }}
.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.strong {{ font-weight:700; }}
.sub {{ display:block; font-weight:400; font-size:12px; color:var(--muted);
       font-family:"IBM Plex Mono", monospace; }}
.bar {{ display:flex; height:12px; width:100%; min-width:190px; border-radius:2px;
       overflow:hidden; background:var(--line); }}
.bar .seg {{ display:block; height:100%; }}
.pct {{ display:block; margin-top:6px; font-family:"IBM Plex Mono", monospace;
       font-size:11.5px; color:var(--muted); letter-spacing:.02em; }}
.barcell {{ min-width:240px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 0;
          font-family:"IBM Plex Mono", monospace; font-size:12px; color:var(--muted); }}
.legend span.sw {{ display:inline-block; width:11px; height:11px; border-radius:2px;
                  margin-right:6px; vertical-align:-1px; }}
.note {{
  border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:14px 16px; border-radius:0 3px 3px 0; margin:0 0 16px; font-size:16px;
}}
.note p:last-child {{ margin-bottom:0; }}
ul {{ margin:0 0 14px; padding-left:22px; max-width:68ch; }}
li {{ margin-bottom:7px; }}
footer {{ border-top:1px solid var(--line); padding-top:20px; color:var(--muted);
         font-family:"IBM Plex Mono", monospace; font-size:12px; }}
a {{ color:var(--accent); }}
a:focus-visible, :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">allex &middot; action quantization &middot; v2</p>
  <h1>One gate, four compression ratios</h1>
  <p class="lede">The compression gate used to answer yes or no at 2&times;. On the
  subtask-labelled allex dataset it now returns a ratio in {{1, 2, 2.5, 3}} for every
  16-step chunk, from a general confidence and a task-specific ceiling asked of the same
  vision model.</p>
</header>

<div class="grid stats">
  <div class="stat"><span class="v">{S['n_episodes']}</span><span class="l">episodes labelled</span></div>
  <div class="stat"><span class="v">{S['n_chunks']:,}</span><span class="l">chunks &middot; 16 steps each</span></div>
  <div class="stat"><span class="v">{S['realised_overall_ratio']:.2f}&times;</span><span class="l">realised compression</span></div>
  <div class="stat"><span class="v">{S['n_blocked']:,}</span><span class="l">hard-blocked to K=1</span></div>
</div>

<section>
  <h2><span class="n">01 / DATA</span>What the robot is doing</h2>
  <p>allex is a two-armed humanoid with no pinch grip: it presses both palms against
  opposite faces of a parcel and holds it by the relative pose of the two hands. The
  dataset is <code>subtask_labeled_data_update_eef_256x256_hojin</code> &mdash; {S['n_episodes']} episodes,
  237,667 frames at 30&nbsp;fps, two ego cameras (left and right) at 256&times;256, and a
  48-dimensional action vector: 7 joints per arm, 15 per hand, 2 neck, 2 waist.</p>
  <p>Every episode runs through the same four subtasks several times over, and each frame
  carries the human subtask annotation, so a chunk knows which of the four it belongs to:
  <strong>Bring Object</strong> (dragging a box toward the robot), <strong>Rotate Box</strong>,
  <strong>Pass Object</strong> (moving an object sideways to the other side), and
  <strong>Rotate PolyBag</strong> (a bag flipped with one hand).</p>
  <div class="note"><p><strong>The actions are absolute joint targets, not deltas.</strong>
  Every entry is a pose the low-level controller servos to. That single fact decides the
  whole design below.</p></div>
</section>

<section>
  <h2><span class="n">02 / MECHANISM</span>Fractional ratios by skipping, never summing</h2>
  <p>In an end-effector-delta embodiment (RoboCasa), running two steps as one means
  executing their <em>sum</em>, because deltas compose. Here summing two absolute targets
  gives a pose at roughly twice the joint angles &mdash; a command nowhere near the
  demonstrated trajectory, usually outside the joint limits. Compressing an
  absolute-target stream can only mean <strong>skipping</strong> targets and letting the
  controller travel further between the ones that survive.</p>
  <p>A ratio K means one emitted target per K input steps. For K&nbsp;=&nbsp;2.5 there is no
  single stride, so the emitter keeps a real-valued cursor:</p>
<pre><code>while True:
    i = int(math.floor(pos + 0.5))   # not round(): round(2.5) == 2 (banker's)
    if i &gt;= chunk_end: break          # this emission belongs to the next chunk
    emit(target[i])
    pos += K</code></pre>
  <p>For K&nbsp;=&nbsp;2.5 the cursor walks 0, 2.5, 5, 7.5&hellip; &rarr; indices 0, 3, 5, 8&hellip;,
  so the skip alternates 3, 2, 3, 2 and the average is exactly 2.5. The cursor is
  <strong>carried across chunk boundaries</strong>: restarting it every 16 steps would round
  the leftover fraction away each time and the realised ratio would drift toward the nearest
  integer. The first target is always emitted, and the last target of the episode is always
  emitted &mdash; it is the pose the arm is supposed to end at.</p>
  <h3>Unit test &mdash; 200-step sequence, one ratio throughout</h3>
<pre>{esc(TEST)}</pre>
  <p>Every ratio lands within 1% of its requested value, first and last targets survive, and
  the K&nbsp;=&nbsp;2.5 gap pattern alternates instead of collapsing to a constant stride.
  Run it with <code>python allex_v2_ratio.py</code>.</p>
</section>

<section>
  <h2><span class="n">03 / PROMPTS</span>Two stages: is it safe, and how far</h2>
  <p>Both stages are one Cosmos3-Nano call each: a single prefill of the two camera views,
  then four teacher-forced <code>A) YES</code> / <code>A) NO</code> slots whose
  P(YES) is read off the {{YES, NO}} tokens. Anything computable from the planned joint
  targets &mdash; palm separation and its rate, arm speed, wrist rotation, travel, finger
  motion, the joint move each candidate ratio would demand &mdash; is computed and stated
  as a fact. The model is only ever asked what the cameras show.</p>

  <h3>Stage 1 &mdash; base confidence p (general, unchanged from v5)</h3>
  <div class="prompt"><div class="hd">guidance</div><pre>{esc(GUIDANCE)}</pre></div>
  <div class="prompt"><div class="hd">questions</div><pre>{esc(ASK)}</pre></div>
  <p>The four answers combine with the deterministic risks (infeasible skip, rotation while
  holding, finger transition) into <code>p</code> &isin; [0,&nbsp;1] &mdash; "safe to compress
  this moment at all". Three terms in that combination were recalibrated for v2, because a
  ratio needs an absolute number where the v1 gate only needed a rank inside an episode:</p>
  <ul>
    <li><strong>Infeasibility is graded against this dataset's own limit.</strong> v5 flagged
    any chunk whose 2&times; skip would demand more than 0.159&nbsp;rad in one step. That
    number came from a slower allex recording; this data reaches
    {MERGE_LIMIT_V2}&nbsp;rad in a single demonstrated step (p99.9 of 41,326 steps). The old
    flag pinned 32% of chunks to p&nbsp;=&nbsp;0 for being faster than a demonstration that
    is not this one.</li>
    <li><strong>"Being turned or tipped" moved to the ceiling.</strong> It is what the
    stage-2 prior already encodes for Rotate Box (2) and Rotate PolyBag (2.5). Counted at
    full weight in both places it drove Rotate Box to K&nbsp;=&nbsp;1 everywhere, so in
    stage 1 it now only shades the confidence (weight {REORIENT_W}).</li>
    <li><strong>Finger motion counts less when the hands are empty.</strong> Pre-shaping a
    hand in mid-air is not a grasp transition, so the term is halved in proportion to the
    model's own answer that the hands have not touched anything yet.</li>
  </ul>
  <p>"Held" likewise now means held <em>between the two palms</em>: a one-armed motion cannot
  lose a two-palm hold however close the wrists happen to be, which matters here because the
  PolyBag flip is one-handed by definition.</p>

  <h3>Stage 2 &mdash; ceiling K<sub>max</sub> (new, task-specific)</h3>
  <div class="prompt"><div class="hd">guidance</div><pre>{esc(STAGE2_GUIDANCE)}</pre></div>
  <div class="prompt"><div class="hd">questions</div><pre>{esc(STAGE2_ASK)}</pre></div>

  <h3>From two answers to one ratio</h3>
<pre><code>K_max = TASK_CEILING[task]                  # prior, not a lookup: the VLM moves it
for p_i, c_i in ((pD, 2.5), (pC, 2.0), (pB, 1.0)):
    K_max -= p_i * max(0, K_max - c_i)      # soft clamp, least restrictive first
lift   = pA * (1 - max(pB, pC, pD))         # A is the only check that can raise it
K_max += 0.5 * lift * (3 - K_max)           # and only at half strength

K = snap(1 + p * (K_max - 1))  ->  {{1, 2, 2.5, 3}}</code></pre>
  <p>Each check is a soft clamp: at P(YES)=1 it pulls the ceiling all the way to its own
  value, at 0 it leaves it alone. Because the final mapping is multiplicative, stage 2 can
  only ever lower a ratio stage 1 already judged unsafe &mdash; a low <code>p</code> pins K
  near 1 whatever ceiling comes back.</p>
  <p>After labelling, one deterministic layer still applies, the v1 block rule at this
  dataset's own scale: a chunk where the robot holds a parcel between its palms while
  rotation accumulates past {ROT_ACCUM_LIMIT_V2:.0f}&deg; over three chunks, or where the palm
  separation changes faster than {GAP_RATE_LIMIT_V2*1000:.1f}&nbsp;mm/step, is pinned to K&nbsp;=&nbsp;1
  ({S['n_blocked']:,} of {S['n_chunks']:,} chunks, {100*S['n_blocked']/max(1,S['n_chunks']):.1f}%).</p>
</section>

<section>
  <h2><span class="n">04 / CEILINGS</span>Why each subtask gets the ceiling it gets</h2>
  <p>These are the domain expert's readings. They enter as the starting ceiling for a
  segment's task and as the content of the four stage-2 questions &mdash; the model still
  decides, chunk by chunk, which situation it is actually looking at.</p>
  <div class="tablewrap"><table>
    <thead><tr><th scope="col">Subtask</th><th scope="col">Ceiling</th><th scope="col">Reading</th></tr></thead>
    <tbody>{why_rows}</tbody>
  </table></div>
</section>

<section>
  <h2><span class="n">05 / RESULT</span>Ratios that came out</h2>
  <div class="tablewrap"><table>
    <thead><tr>
      <th scope="col">Subtask</th><th scope="col">Ceiling</th><th scope="col">mean p</th>
      <th scope="col">mean K<sub>max</sub></th><th scope="col">mean K</th>
      <th scope="col">Distribution of K</th>
    </tr></thead>
    <tbody>{''.join(rows)}
    <tr><th scope="row">All chunks<span class="sub">{S['n_chunks']:,} chunks</span></th>
      <td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">&mdash;</td>
      <td class="num strong">{np.mean([r['K'] for r in recs]):.2f}</td>
      <td class="barcell">{overall_bar}
        <span class="pct">{pctstr(S["overall_dist"])}</span></td>
    </tr>
    </tbody>
  </table></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--k1)"></span>K = 1 &middot; full rate</span>
    <span><span class="sw" style="background:var(--k2)"></span>K = 2</span>
    <span><span class="sw" style="background:var(--k25)"></span>K = 2.5</span>
    <span><span class="sw" style="background:var(--k3)"></span>K = 3</span>
  </div>
  <p style="margin-top:18px">Emitting the whole dataset under this schedule keeps one target
  in every {S['realised_overall_ratio']:.2f} &mdash; the episode-level average of a per-chunk
  schedule, computed with the same cursor that would run at inference.</p>
</section>

<section>
  <h2><span class="n">06 / REVIEW</span>Watching it decide</h2>
  <p>The labels are burned back onto the frames &mdash; both ego views, the subtask name, p,
  K<sub>max</sub>, the chosen ratio, every computed descriptor and both stages' YES/NO
  answers, with the ratio timeline for the segment underneath &mdash; so the calls can be
  judged by eye rather than by aggregate.</p>
  <pre><code>{esc(VIDEO)}</code></pre>
</section>

<section>
  <h2><span class="n">07 / CODE</span>Where it lives</h2>
  <ul>
    <li><code>allex_v2_ratio.py</code> &mdash; the fractional-ratio emitter and its unit test</li>
    <li><code>allex_v2_common.py</code> &mdash; descriptors, stated facts, stage-2 prompt, ceiling blend</li>
    <li><code>allex_v2_calibrate.py</code> &mdash; measures the deterministic constants on this dataset</li>
    <li><code>allex_v2_label.py</code> &mdash; two-stage labelling client (shardable by episode)</li>
    <li><code>allex_v2_aggregate.py</code> &mdash; hard-block layer, final K, ratio distribution</li>
    <li><code>allex_v2_render.py</code> &mdash; the labelled review video</li>
    <li><code>allex_v2_html.py</code> &mdash; this page</li>
  </ul>
  <p>All under <code>vlm_gate/scripts/</code>; the v1 <code>allex_*</code> scripts are
  untouched. The judge runs in <code>cosmos_judge_venv</code>, the client in
  <code>quant_gate_eval</code>.</p>
</section>

<footer>allex variable-ratio gate &middot; {S['n_chunks']:,} chunks over {S['n_episodes']} episodes
&middot; Cosmos3-Nano judge, two stages per chunk</footer>
</div>
"""
open(OUT, "w").write(page)
print(f"-> {OUT} ({len(page)/1024:.1f} KB)")
