# dexjoco K2 temporal-quantization dynamics probe

Date: 2026-08-13. Env: `PandaHammerNailGymEnv` (external_dependencies/dexjoco), control_dt=0.02s, physics_dt=0.002s (10 substeps/tick), conda env `dexjoco`, CPU-only run (rendering bypassed).

## Action semantics (verified in code)
- **Arm**: absolute EE pose target (mocap body) tracked by OSC `opspace` torques — no re-anchoring; target is absolute world pose.
- **Hand (Allegro, 16 joints)**: absolute joint position targets to position actuators.

So *both* halves of the action are absolute-target controllers, unlike robocasa's delta-EE + achieved-mode re-anchoring.

## Protocol
Scripted linear ramp of targets over 2N=50 ticks (1.0 s), N=25 K2 commands.
- Fine: 2N absolute targets, one per tick.
- K2: every other target, each held 2 ticks; **identical final target**.
- 25 extra settle ticks holding the final target for settled reference.
- Arm: amplitudes 0.02/0.05/0.10 m on x/y/z axes. Hand: +0.1/0.3/0.6 rad ramps on all 16 joints (clipped to ctrl range).
- Fresh env reset (seed 0) per run; identical initial state for fine vs K2.

## Results

| probe | ratio K2/fine at trajectory end | ratio after settle | traj RMS dev | traj max dev |
|---|---|---|---|---|
| Arm (all 9 axis×amp cases) | 1.0142–1.0143 | 1.0029 | 0.14–0.69 mm (scales with amp) | 0.19–0.93 mm |
| Hand (3 amplitudes) | 1.0081–1.0085 | 1.0000 | 0.001–0.0057 rad | 0.0012–0.0072 rad |

(K2 slightly *leads* at ramp end because the held target is the odd-index, half-tick-ahead one — a transient phase artifact, not gain error.)

## Conclusion
- **No systematic endpoint compression under K2.** Ratios are ~1.00–1.01 vs robocasa's 0.89–0.90. The robocasa compression came from delta-action re-anchoring on achieved pose; absolute targets eliminate it, as hypothesized.
- Residual K2 effect is only a sub-mm / <0.01 rad transient tracking deviation that vanishes on settle (hand exactly 1.0000; arm 1.0029, within OSC steady-state tolerance shared by both modes).
- **No model-side inverse-dynamics correction is needed for dexjoco K2.** Amplitude- and axis-independent for the arm; hand joints likewise unbiased.

Data: `dexjoco_k2_dynamics_probe.json` (same dir). Probe script archived in session scratchpad (`dexjoco_k2_probe.py`).
