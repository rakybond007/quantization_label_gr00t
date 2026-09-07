# 논문 Figure 생성 프롬프트 — 문항 도출·라벨링 파이프라인

아래 `---` 사이를 **통째로 복사**해서 웹 Claude 나 ChatGPT 에 붙여 넣는다.
우리 방법을 처음 듣는 모델도 그대로 그릴 수 있게 내용이 전부 들어 있다.

**이미지 생성 기능을 쓰지 말 것.** 그림 생성 모델은 도식의 글자를 뭉갠다.
아래 프롬프트는 **SVG** 를 받도록 되어 있다 — 글자가 정확하고, 벡터라
Illustrator·Inkscape 에서 열리며, PDF 로 뽑아 LaTeX 에 바로 넣을 수 있다.
아티팩트에서 복사할 때 상자·화살표가 사라지는 문제도 없다(파일로 저장하면 된다).

---

You are making a **method figure for a robotics/ML paper**. Output a single
self-contained **SVG** file. No raster images, no external fonts, no scripts.
Use only `<rect>`, `<path>`, `<line>`, `<polygon>`, `<text>`, `<g>`, `<defs>`,
`<marker>`. Every label must be real `<text>` so it stays crisp and editable.

## What the figure explains

A robot policy predicts a chunk of future actions. Executing every step is slow.
We can **compress** the chunk — merge k consecutive steps into one command — but
compression damages some tasks and not others. Our method decides, **for each
moment**, how much compression is safe. The figure shows how we get there.

The pipeline has five stages, left to right. Do not add stages.

### Stage 1 — Measure damage

Run the policy on every task at several **fixed compression ratios** and record
success rate. Nothing is learned here; it is measurement.

Show a small table with these real numbers (success rate; 50 episodes per task):

| ratio | spatial | object | goal | long-horizon |
|---|---|---|---|---|
| 1.0x | 0.98 | 0.99 | 0.96 | 0.89 |
| 1.7x | 0.96 | 0.99 | 0.95 | 0.86 |
| 2.0x | 0.82 | 0.97 | 0.94 | 0.82 |
| 2.5x | 0.41 | 0.82 | 0.62 | 0.54 |

Highlight the 2.5x row: that is where tasks separate most.

### Stage 2 — Split tasks into two pools

Sort tasks by measured damage at the ratio with the widest spread. Tasks that
lose success go in the **RISK pool**; tasks that keep it go in the **STABLE pool**.
Colour RISK warm, STABLE cool, and keep those two colours consistent everywhere
else in the figure.

Show 3 example instructions per pool, verbatim, in a monospace face:

RISK pool
- `put both the cream cheese box and the butter in the basket` — 0.02
- `pick up the black bowl on the cookie box and place it on the plate` — 0.10
- `put the bowl on the stove` — 0.18

STABLE pool
- `pick up the alphabet soup and place it in the basket` — 1.00
- `put the bowl on top of the cabinet` — 0.92
- `push the plate to the front of the stove` — 1.00

### Stage 3 — An LLM reads both pools and proposes questions

Both pools of **task instructions only** (no images, no action values) go to an
LLM. It looks for what the RISK instructions share that the STABLE ones do not,
and vice versa, and writes each shared thing as **one yes/no question about the
unit action being performed** — not about which task this is.

Draw this as instructions flowing into an LLM block, and candidate questions
flowing out. Label the block plainly (e.g. "LLM"); do not draw a brain, a robot
face, or a chat bubble.

Show two surviving questions, one per pool, with their sign:

- from RISK, scored as a **penalty**:
  *"Is the object being set down on a small fixed spot it must sit squarely on —
  a plate, a burner, a rack — rather than into something that would catch it?"*
- from STABLE, scored as a **bonus**:
  *"Is this a job that needs no hold kept on the object — one push and it carries
  on where it was sent?"*

Also show, smaller and struck through or greyed, one **rejected** candidate, to
make clear that candidates are filtered:

- rejected: *"Does the object go into an enclosed space — a drawer, a cabinet?"*
  — covers both pools, so it separates nothing.

### Stage 4 — Rank, sign, weight

Three short rules, shown compactly (a small table or three stacked rules):

- **rank** — a candidate scores `(tasks it covers in its own pool) − (tasks it
  wrongly covers in the other pool)`.
- **sign** — set by which pool it came from. RISK → penalty. STABLE → bonus.
  It is never guessed.
- **weight** — the number of tasks the question covers, normalised so the
  penalties sum to 1 and the bonuses sum to 1.

### Stage 5 — Label every moment, then place it in the task's band

A VLM sees the camera frames at one moment and answers **every** question on a
**1–5 scale** (each question carries its own 5-level rubric). Grades are rescaled
to 0–1 as `(g − 1) / 4`, then combined:

```
confidence = ( 1 + Σ wᵢ·bonusᵢ − Σ wⱼ·penaltyⱼ ) / 2        ∈ [0, 1]
```

Confidence is an **ordering**, not a ratio. So it is mapped into the band the
task's own measurement allows, by quantile within that task:

```
speed-up ratio at this moment  =  band_place( confidence, lo, hi )
```

where `[lo, hi]` comes from Stage 1 for that task — the ceiling is given by
measurement, and the questions only decide **how much of that ceiling to use**.

End the figure with the output: a short time-line of one episode where each
chunk is tinted by its assigned ratio, so slow moments and fast moments are
visible at a glance. Label a couple of chunks (e.g. `1.0x` near a careful
placement, `2.5x` during free transit).

## Design requirements

- Landscape, roughly 1600 × 620, sized for a two-column paper's full width.
  Nothing may overflow the canvas; give the outermost group ~24 px padding.
- One accent colour for RISK, one for STABLE, and a neutral grey for structure.
  Muted, print-safe, readable in greyscale. No gradients, no drop shadows,
  no 3-D, no clip art, no emoji.
- Stage titles ~15 px semi-bold, body ~12 px, table and instruction text ~11 px
  monospace. Use a generic stack such as
  `font-family="Helvetica Neue, Arial, sans-serif"` and `"SFMono-Regular, Menlo,
  monospace"` — never a font the reader may not have.
- Number the stages 1–5; the pipeline really is a sequence, so numbering carries
  information.
- Arrows between stages: thin, one consistent arrowhead defined once in `<defs>`.
- Keep every text string exactly as written above. Do not paraphrase the
  questions — their wording is the contribution.
- Do not invent extra numbers, stages, datasets, or model names.

Return only the SVG.

---

## 프롬프트를 고칠 때

- **숫자를 바꾸려면** Stage 1 표와 Stage 2 의 여섯 줄만 고친다.
  정본은 `vlm_gate/analysis/eval_results/libero_ladder_noclip.md` 와
  `libero_taskwise_ladder.md` 다.
- **문항을 바꾸려면** Stage 3 의 두 문항과 기각된 하나를 고친다. 확정 문항 전문은
  `vlm_gate/prompts/` 에 있다.
- **더 넓게/좁게** 하려면 Design requirements 의 크기 한 줄만 고친다.

## 안 되면

SVG 가 어긋나게 나오면 "the boxes overlap in stage 3, redo the layout with a
fixed 5-column grid" 처럼 **무엇이 어디서 겹쳤는지** 를 말해 주면 고쳐 준다.
처음부터 다시 시키지 말 것.
