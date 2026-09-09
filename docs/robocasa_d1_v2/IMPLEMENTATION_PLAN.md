# RoboCasa D1 v2: implementation handoff

2026-09-10. Status: design and CPU arithmetic checks complete; production code,
GPU smoke, semantic pilot, full labeling and HF upload NOT executed for this version.
Supersedes the earlier recommendation to use only fine/task-cap. That binary rule
is an optional ablation, not the proposed main method. Existing releases and running
DexJoCo jobs must remain unchanged. User will request implementation after switching
reasoning effort; the agent must not claim it changed the model/effort itself.

## 1. Decision and explicit assumptions

Implement a D1-based, four-question, five-grade + U teacher. Preserve
`R=max(A,B)`, `S=max(C,D)`, `score=S*(1-R)`. Use expected numeric grades, a common
absolute score-to-speed map, and task cap only as a final limit. No sample ranks,
histogram equalization, per-task normalization, quotas, or temporal smoothing.

This is an operational weak-supervision policy, not measured local failure risk.
It is preferable to the binary proposal as a next candidate because it retains the
requested multi-speed decision, has explicit monotonic behavior, and avoids the
known floor/endpoint and task-cap rescaling defects. This is not evidence of better
robot success. Normal task-level eval will test that later; local counterfactual
rollouts and new simulator branches are not prerequisites for annotation.

Retained design choices, not discoveries:

- The numeric spacing of grades, max aggregation and product are policy choices.
  Max prevents one risk being averaged away; either support source may suffice.
  It is not an independent-probability calculation. Freeze the question set.
- Positive guidance and negative manipulation burden can coexist. Ask them
  independently; never tell the VLM to pre-subtract risk from support.
- A temporal window alternating two support types can score lower than either
  pure phase. The coverage rubric makes that conservative bias explicit.
- Softmax over grade tokens is conditional model belief over these answer tokens,
  NOT calibrated success probability. No entropy penalty is multiplied into score.
- Taking grade expectations discards some uncertainty distinctions. Preserve all
  probabilities and use them for auxiliary training; do not claim preservation
  magically corrects the decision made from the expectation. Do not manufacture a
  joint grade distribution by assuming independent questions.
- Same score means same preference inside this RoboCasa/operator configuration.
  A shared numeric scale across different robots/controllers is NOT asserted.

## 2. Evidence and reuse boundary

Sources: `vlm_gate/analysis/eval_results/LADDERS.{md,json}`,
`vlm_gate/scripts/phase9_checks.py`, `vlm_gate/analysis/robocasa_task_ceilings.json`.
OpenDrawer .72 -> .36 at 2.5 vs CloseDrawer 1.00 -> 1.00 motivates distinguishing
maintained manipulation from passive pushing. CoffeePressButton .82 -> .92
contradicts assigning negative weight solely to small contact size. These are
50-trial task-level associations, not localized causal effects. The RoboCasa table
treats some different clipping runs as comparable; preserve that qualification.

Reuse: episode identities, raw data, existing cap table, existing contact-fixed
decisions with provenance, Cosmos environment/server protocol, resumable worker
pattern, fractional block implementation. DO NOT reuse old A-E/conf as new A-D
or invent probabilities from old hard labels. New semantics require new VLM calls.
Original release remains the comparison arm.

The original cap table contains [lower,upper]. Use ONLY upper. Fine remains allowed
in every task. Preserve the table for the first comparison, including cap=1 tasks.
Its loss tolerance (including up to 10 percentage points absolute drop in ceiling
derivation) is inherited policy, not a certified safety threshold. Store its hash
and provenance. Unknown task/no ceiling is a preflight error, not a silent default.
Changing cap later only reruns CPU policy conversion; grade inference is reusable.

## 3. Frozen proposed questions and rubric

`prompt_v1.txt` provides the full English rubric and answer syntax. Four axes:

| slot | meaning | sign | strong vs weak anchor |
|---|---|---|---|
| A | alignment / maintained manipulation relation | risk | open or already aligned supported push -> tight mating/unassisted endpoint |
| B | passage clearance | risk | unobstructed -> near-fitting passage |
| C | passive support/guidance during interaction | support | absent -> visibly maintained throughout |
| D | open movement / broad landing allowance | support | absent -> entire window including arrival is open/broad |

Each axis has its own 1..5 anchors, plus U for genuinely unobservable necessary
relations. These are not absent/approach/engaged recoded into severity. Do not add
task names, success percentages or desired speed into the VLM prompt.
Prompt bytes, category order, facts and image layout are versioned together.

Why five numeric grades: under this proposed 2.5-reference policy, deterministic
three-grade answers produce only C in {0,.25,.5,1}, so 2x is unreachable after
nearest rounding. It would appear only through probability mixing. Five grades
allow ALL four speeds from confidently distinguished rubric states. This is an
arithmetic reason to require useful extra anchors, not proof the VLM distinguishes
five levels. The semantic pilot must verify that; do not label everything merely
because a parser returns five digits.

## 4. Exact score and ratio policy

For a parsed numeric answer q, let p be the six-category probabilities in order
`1,2,3,4,5,U`. Preserve p without four-decimal rounding. With numeric mass m:

```
e_q = sum(g * p_q[g], g=1..5) / m
z_q = (e_q - 1) / 4
R = max(z_A, z_B)
S = max(z_C, z_D)
C = S * (1-R)
k_target = 1 + 1.5*C
k = argmin_{g in [1,1.5,2,2.5], g<=task_cap} (abs(g-k_target), g)
```

The second sort key selects the lower speed on a tie. Equivalent score cutoffs
are 1/6, 1/2, 5/6 (ties lower). `reference_ratio=2.5` is frozen, NOT dynamically
`max(available_grid)` or task cap. Adding a future decoder must not silently alter
all previous thresholds. No empirical distribution enters this function.

| confidently graded A,B,C,D | C | unconstrained selection |
|---|---:|---:|
| 1,1,1,5 | 1 | 2.5 |
| 2,1,1,5 | .75 | 2 |
| 3,1,1,4 | .375 | 1.5 |
| 5,1,5,1 | 0 | 1 |

These examples test the policy, not prescribed task-level outputs. A 2x-decoder
label does not prove 2x is optimal. Nearest rounding minimizes distance to the
chosen nominal preference, not measured robot loss or control distortion.
Do not silently switch to inverse-speed/time-savings coordinates; that is another
policy with different thresholds. Log nominal ratio and actual block count separately.

Unknown handling:

1. Malformed output, missing probability, nonfinite/negative entries, wrong token
   positions are technical failures. Retry at most twice (batch then individual),
   retain error record, and block a complete release if any interior rows remain
   technically unprocessed. They are not U and not fine labels.
2. Parsed U OR category argmax U makes that axis unknown (ties involving U -> U).
   Numerical mass <=1e-12 also makes it unknown. Otherwise use conditional numeric
   expectation above. Store U mass even when its category is not selected.
3. For unknown axes use z in [0,1]. With per-axis intervals [l,h]:
   `Clo=max(lC,lD)*(1-max(hA,hB))`,
   `Chi=max(hC,hD)*(1-max(lA,lB))`.
4. Select a ratio from both bounds after cap. If identical, this ratio is valid;
   otherwise write ratio=1, ratio_valid=0, reason=semantic_unknown. A known strong
   alternative support can make an unknown support irrelevant; no blanket any-U veto.
5. Valid legacy contact-fixed rows override to fine/valid=1 even if semantic
   judgment is U. Question U remains its own learnable category. Missing full
   input window at the episode tail takes precedence: supervision there is invalid.

No high-probability risk hard veto, extra entropy multiplier, percentile threshold
or smoothing is added to the initial policy. Save numeric entropy and distance to
nearest score boundary as diagnostics. CPU conversion may report choices under
boundary offsets +/-0.05 as a sensitivity diagnostic; do not tune offsets to fill
classes. This number is a diagnostic perturbation, not a safety margin.

## 5. Actual input, execution horizon, contact and indexing

Read-only inspected dataset:
`/mnt/lustre/slurm/users/prehj/quant_agent_workspace/assets/datasets/robocasa_mg_gr00t_300`
7,200 episodes, 2,073,457 frames, 24 tasks, 20 Hz; images 256x256 from left/right/wrist.
The local RoboCasa service executes the WHOLE compressed chunk, unlike LIBERO's
replan-5 client (`vlm_gate/scripts/robocasa_service_moe.py`). Use 16 fine intervals,
continuous-delta block sums, discrete channels block-last, raw singleton tail,
no carry. Block counts for 1/1.5/2/2.5 are 16/11/8/7. At 2.5 nominal the full-chunk
effective count ratio is 16/7, not literally 2.5. Freeze this execution contract.

Teacher input: a labeled 3x3 sheet, rows t/t+8/t+16, columns left/right/wrist,
all cells initially 256x256. Natural canvas 768x768 plus small readable headers.
Supply instruction and exact command facts for actions [t,t+16). Record processor
resizing and visual-token count; do not silently downsample unreadable contact
regions to fit batches. This is privileged OFFLINE demonstration context, already
available data. Student sees current observation/instruction/proprio and predicted
future latent, not these ground-truth future views during deployment.

All starts satisfying t+16<L are eligible. Explicitly mask last 16 rows of each
episode, giving 1,958,257 expected full-context labels if all episodes are longer
than 16 (assert against metadata). This differs from the old release's last FOUR
masked rows; never keep EXPECTED_TAIL=4 in a new converter. Do not silently copy
future frames at tails or fill interior missing rows by interpolation. Full stride1
is planned; stride4 output is not silently promoted to full supervision.

Raw action layout, verified from `meta/modality.json`:
`base 0:4, mode 4, EE delta xyz 5:8, rotation 8:11, gripper 11`.
Model concat layout: `xyz 0:3, rot 3:6, gripper 6, base 7:11, mode 11`.
Thus the legacy raw descriptor's xyz slice itself is CORRECT. They are two different
layouts, not evidence of an existing indexing bug. Build a metadata-driven adapter
and test raw->model->raw equality. Do not use model hold_dims=(6,11) on raw arrays.

New facts describe numeric command displacement, direction and gripper/mode values,
without invented meters, forces or realized speed. In particular remove the legacy
text inference 'holding something while creeping' from closed-gripper/low-command
proxy. Rotation command vector statistics are not asserted to be exact realized
SO(3) motion. All such facts belong to the recorded fine trajectory, not compressed
rollout. Do not add heuristic clip/jump guards with unverified controller limits.

Main contact handling preserves the existing release's fixed=1 decisions exactly
on eligible rows. Record `legacy_fixed`, source revision and fixed-label provenance.
The release includes proxy fallback, so these are NOT uniformly sensor truth.
Before full labeling inspect original sidecar scope/validity; legacy scalar fixed
alone does not establish full-window contact coverage. Store raw contact fields if
available; lack of provenance must be explicit, never relabeled 'measured'. No new
contact detector or changed contact window is part of this first comparison.
Separately save full-window commanded gripper/mode transition facts for auditing;
adding them as hard guards is a separately versioned policy, not an unreported change.

## 6. Portable output contract

Use a NEW versioned staging directory; final requested naming ends in `-Astra`.
Do not overwrite either previous public labels or the stopped LIBERO Astra draft.
Separate expensive raw annotation from cheap derived ratio decisions:

```
raw/shard_0.jsonl, raw/shard_1.jsonl   # append/resume, raw answer + probabilities
labels/robocasa_training.parquet     # canonical sorted training/audit table
labels/robocasa.parquet              # optional legacy scalar view, clearly labeled
policy/policy.json, ceilings/robocasa.json, prompts/robocasa.txt
manifest.json, README.md, TRAINING_HANDOFF.md, apply_to_dataset.py
```

Identity: dataset fingerprint + episode_index + episode-local zero-based frame_index.
Also preserve raw global `index`, task id/name and instruction. Never join on the
name alone, global index as local frame, or merely assume episode numbering agrees.

Per-row fields include raw A-D category strings; gp_A..D fixed arrays of 6 floats;
numeric expectation eg_A..D (null for U); grade_valid_A..D; U masses; risk/support/
score or their bounds; score-boundary distance when numeric; task_cap; target;
ratio_before_contact; final ratio; 0-based ratio class; ratio_valid; reason;
legacy_fixed/contact provenance; observation indices; block_sizes and action_count.
Generated category and numeric expectation are distinct. U is not grade3 or grade0.

Manifest freezes dataset info/episodes/modality hashes, relevant parquet identities,
code SHA plus dirty diff hash, model/tokenizer/processor revision, exact prompt/facts/
layout hashes, policy and cap hashes, category order, execution contract and stride.
Pin the actual available Cosmos3-Nano snapshot (previous working snapshot
`7a312c868bcce8e40b3eb40861300a9d0ba3fde1`) after local availability check; no auto-update.
Previous contact release pin: c9d9fbd76a863cdac76afe8f09269c19e48bc1bf;
canonical label SHA: ebaaaddb2055bfbaa56997c906468b12ca1f2b7918ae0d5e7b3dc2f602094534.

apply_to_dataset: portable numpy/pandas/pyarrow CLI, dry-run first; compare episode
task+length and all frame keys; no missing interiors/duplicates; explicit 16-tail mask;
preserve original action/state/video columns and metadata, add feature statistics
for carriers, support safe sidecar mode and copied-parquet mode with manifest/hash.
Neither output overwrites the source. Per-task cap changes can be supplied as a
versioned mapping and reapply CPU policy; prompt-incompatible old conf cannot.

## 7. F-level training integration (must actually implement)

Current code inspection: `flare/dataset_ratio.py` and `transforms_ratio.py` fix
GRADE_TARGET_DIM=10 and five slots; preparation pins one old source SHA and tail4.
Head question/class counts are configurable, but `_grade_loss` currently accepts
only hard CE classes. Merely storing probabilities in parquet WILL NOT train them.

Add a separate schema/config `robocasa_ratio_grade_d1v2`, preserving old configs:

- ratio_label = [class0..3, valid], same 2 floats as existing interface.
- grade_prob_label = flattened [4 questions x 6 category probabilities] +
  [4 question valid flags], 28 floats, order fixed in manifest.
- Optional audit grade_label = [4 hard classes0..5, 4 masks], 8 floats; not confused
  with the old five-question 10-float carrier. Default auxiliary training uses soft
  probabilities including U, not rounded expectations fed into CE.
- Strip all virtual carriers before action padding, normalization-dependent action
  losses or decoder targets. Raw action remains 12D and model action padding 32D;
  concat width42 (12+2+28) must be stripped BEFORE max_action_dim=32 handling.
- Add model grade_num_questions=4, grade_num_classes=6, grade_target_type=soft.
  Loss = masked sum(-p * log_softmax(student_logits)), normalized by global valid
  question count in DDP. An all-invalid global batch must return connected zero.
  Never cast expectation2.4 to a categorical index. Keep hard/soft backwards paths.
- Retain existing main ratio hard CE on valid labels and all multilevel action
  losses. Initial grade coefficient remains 0.1, to isolate label-method changes.
- Use the requested FLARE arm (`ratio_future`): predicted future-token hidden
  state + instruction representation. Auxiliary four question heads share the
  same readout features. No new recurrent memory, no future-demo input at inference.
- Default action selection remains learned ratio classifier (cap-limited). Auxiliary
  grades are outputs and diagnostics. For predicted U, apply the same grade-bound
  unknown check and fallback fine if bounds disagree. Masking teacher ratio loss
  alone does NOT implement fallback. Invalid/malformed model outputs fail closed.
  Unknown fallback quality is learned and must be evaluated, not guaranteed.
- Existing contact-fixed labels supervise the classifier's fine choice; there is
  no oracle contact guard at inference. Do not claim the teacher's hard override
  automatically becomes a perfect student detector. Add grade-rule-only routing
  as an optional eval comparison, not silently replace the learned ratio head.
- Save category order, grade schema, ratio policy, cap policy and fallback mode in
  checkpoint; serving must verify compatibility and return grade probabilities.

RoboCasa full training remains global batch64. Two GPUs: per-GPU32 x accum1 if it
fits; otherwise16 x accum2 gives64. A6000 label throughput tests are NOT H100/H200
batch-capacity evidence. Full training is outside this labeling-plan turn; do not
submit it or alter the existing training environment yet.

## 8. File-by-file implementation sequence

In quantization_label_gr00t use an isolated worktree and prepare delivery on the
user-requested `hj` branch after checking its existing history (no force push).
Do not reset dirty files; existing `hj-libero-labeling` has unrelated live
modifications. Carry over only the required, reviewed Cosmos protocol changes:

1. Add `vlm_gate/scripts/robocasa_d1_checks.py` from frozen prompt and category order.
2. Add `robocasa_d1_policy.py`: pure validated expectation/bounds/map/contact policy.
   Keep `experiments/dexjoco_d1/ratio_d1.py` and its running jobs unchanged.
3. Add `robocasa_d1_reader.py` + `prepare_robocasa_d1_manifest.py`: metadata-driven
   action facts, exact key/frame validity, deterministic pilot and full manifests.
   Decode per episode, bounded cache, render requested windows on demand rather
   than materialize two million 9-cell PNGs in advance.
4. Add `label_robocasa_d1.py`: adapt resumable DexJoCo client, four slots/six categories,
   exact output-to-token matching. Reuse existing Cosmos text-batch probability
   extraction. Changes to server must pass old LIBERO/DexJoCo protocol regression.
5. Add `build_robocasa_d1_labels.py`, `validate_robocasa_d1.py` and portable application
   CLI. Expensive raw grades are immutable; ratio conversion takes policy as input.
6. Add experiment scripts for two independent workers, one per allocated GPU,
   stable episode-based sharding. Each worker uses its own port and logs; preserve
   Slurm CUDA_VISIBLE_DEVICES remapping. No blind global kill or cancellation.
7. In F_level_hj branch hj, add new dataset/transform/config and soft-grade loss path,
   checkpoint verifier and serving fallback. Unit tests precede 2GPU smoke.
8. Package README/handoff with exact local-tested commands, dependencies, download
   revision, dataset overlay and training config. Before publish, review manifest
   and hashes; no HF upload is authorized by this plan-only turn.

Resume validates semantic fingerprint before accepting old records. Runtime batch
size and worker count are scheduling metadata, not changes to label semantics;
allow resharding by exact completed keys. Prompt/model/input mismatch rejects resume.
Use transactional shard progress; detect/truncate only an incomplete final JSONL
record with a recorded recovery event. Never discard valid prior answers silently.

## 9. Gates before full labeling

Gate A — CPU correctness:

- `check_policy_spec.py` already PASS: 625 numeric combinations, 1296 including U,
  all four ratios reachable, risk/support monotonicity, cap-only clipping, endpoint
  reachability, exact lower-tie behavior and redundant-U support handling.
- During implementation additionally test probability conditioning and validation,
  deterministic contact override, tail precedence, missing/duplicate/shifted keys,
  raw/model layout roundtrip, and categorical-logit token-position regression.
- Same row alone or in a differently ordered dataset must get the same ratio.
  Compare standalone vs batch arithmetic in float64; implement tie tolerance
  <=1e-12 in ratio distance, document it and test just outside tolerance.

Gate B — allocated GPU pilot, no full label job until it passes:

- 24 tasks x 2 deterministically selected episodes x 4 starts at 5/35/65/90%
  of eligible index range =192 windows (deduplicate; short episodes handled explicitly).
  Add up to32 unique event/occlusion cases selected by deterministic command/legacy
  contact facts. Log sampling rule; no sampling by desired ratio.
- Start Cosmos on one A6000 within a normal allocation, batch1 then4/8 as memory
  permits. Second GPU uses an independent worker, not DDP for inference. Compare
  at least16 identical windows at batch1 and chosen batch size: category agreement,
  probability drift and ratio flips, not only successful process exit.
- Exact parser/probability completion required after bounded retries. Inspect at
  least96 windows (4/task) visually, comparing numeric answers with the explicit
  rubric and U observability; record per-axis >=2-grade disagreements and U mistakes.
  More than10% such disagreements on any axis fails this rubric pilot. This is a
  single-reviewer semantic quality screen, not independently measured accuracy.
- Do not fail merely because a speed class is rare. Diagnose constant answers,
  task-name-only answers, positive/negative semantic duplication and unreadable
  images. If an axis fails, revise the prompt once with version/hash change and
  rerun a disjoint pilot before considering full labeling; do not loop to fit ratios.
- Benchmark sustained windows/sec, memory and error rate over a longer warm segment,
  excluding model load. Compute full ETA from actual eligible count and measured
  two-worker rate (report range); no guessed wall-time commitment before measurement.

Gate C — training/application integration:

- Overlay two full episodes with probabilities and U; exact original-column equality,
  new feature/stats widths, boundary/tail masks and base metadata fingerprint.
- Include valid-U questions, semantic-invalid ratio, contact-fixed fine and an
  all-invalid batch. Check four heads x six classes and finite soft CE + backward.
- 2GPU short FLARE smoke with nonzero ratio/grade/action gradients, save/reload and
  inference grade outputs/fallback. Small batch validates code, not batch32 fit.
  Verify labels never enter action loss or future input as features.
- Full labeling can proceed after Gates A/B while C completes independently, but
  do not call the release training-ready until C passes. No new label meanings are
  changed to accommodate a stale loader; update the loader.

Full run: stride1, two GPUs maximum, reuse any suitable existing allocation without
disturbing other work. If none, request normal srun allocation with CPU/memory/time
defaults as user requested; accept normal queueing. Keep an interactive allocation
alive for reuse until scheduler expiry; don't proactively release it. Restart only
owned worker processes. Never cancel existing DexJoCo labeling without user consent.
Pending jobs do not justify policy-bypassing scheduler experiments.

Gate D — release audit:

Exact eligible coverage and tail masks; zero duplicate/interior missing/technical
failure rows; all known caps respected; all eligible legacy_fixed rows fine;
valid probability vectors; schema/range/hash verification; task-wise class counts
before cap, after cap and after contact, plus U/technical failure counts separately.
Report nominal mean, selected block counts, switching frequency and sensitivity;
none is itself a robot-success estimate. Include model-comparison-ready same-key
old/new tables without using old conf as new semantic data. Final dataset name ends
`-Astra`, immutable raw grades preserved for CPU remapping and per-task policy changes.

## 10. Cost discipline and next-turn scope

No more open-ended method redesign is required before implementation. This proposal
is ready for medium-effort implementation and the bounded pilot; it is NOT already
validated for full labeling. User explicitly intends to switch effort and then give
the execution instruction. Do not start a GPU job in this plan turn.
Use existing working Cosmos environment/editable F-level environment; no unnecessary
reinstallation. Keep large raw logs out of conversational context. CPU policy
variants reuse one VLM annotation, so later speed threshold/cap ablations cost no
new Cosmos calls. User may use mostly 5.6 for routine implementation when delegated;
do not spawn agents for this plan unless separately asked.

Recommended later scientific comparisons (same model/controller, ordinary task eval):
original teacher; same new questions with old task-scaled floor; new common-nearest
policy. Store these as CPU policy views of one annotation where semantics permit.
Do not equate old-question/new-question comparisons with a pure mapping ablation.
Report success and step counts together, including baseline and sample counts.
