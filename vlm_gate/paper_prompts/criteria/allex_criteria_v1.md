# Quantizability criteria — allex bimanual package handling
Authored by the frontier model (Claude) from direct inspection of 720p demonstration
footage (10 episodes, 96 frames viewed), with the operator's framing of the task.
This document is the *specification*; the teacher prompt is generated from it, and
teacher quality is measured as agreement with labels produced under it.

## The question being answered
The policy emits 16 absolute joint targets (0.53 s at 30 Hz). Halving the control
rate means executing every second target and skipping the one between. Ask only:
**would that skipping change the outcome of this half-second?**

Not "is the robot touching something". Contact is common and mostly harmless — this
robot spends most of every episode in contact while transporting a package.

## What this robot does
Two 7-DoF arms with multi-finger hands. It does *not* pinch with fingers: it presses
both palms against opposite faces of a package and carries it squeezed between them.
Fingers stay near-static; the grip is maintained by arm posture. Consequently
"gripper open/close" has no meaning here, and finger-joint motion is not a grasp signal.

## COMPRESSIBLE (YES)
- Reaching toward a package, hands still clear of it.
- Retracting, returning to rest, repositioning the torso between items.
- **Transporting a package already squeezed between both palms**, when the grip
  geometry is steady — the wrists hold a roughly constant separation and the package
  is not being reoriented. Skipping intermediate targets shifts the path by
  millimetres, and a package held by opposing palms tolerates that.
- Broad sweeps over empty conveyor.

## NOT COMPRESSIBLE (NO)
- **Establishing the grip**: palms converging onto a package, up to the moment the
  load is taken. Arriving late or overshooting by millimetres is the difference
  between a secure hold and a knocked-over package.
- **Releasing**: palms separating from a package that is being set down, until the
  package is free-standing.
- **Placing where pose matters**: lowering onto the conveyor, straightening, or
  turning so the barcode faces up. The task specification makes final orientation
  part of success, so the last centimetres are not interchangeable.
- **Handling deformable or unstable items** (plastic mailer bags, loose piles) at any
  moment of contact — these shift under abrupt motion in a way rigid boxes do not.
- Any window whose merged step demands joint motion beyond what the robot produced in
  the demonstrations (computed, not judged: > 0.159 rad in one step).

## Deciding rule when torn
Ask what a millimetre-scale path error would cost *in this specific window*. If the
answer is "nothing" → YES. If it is "the package is dropped, mis-seated, or
mis-oriented" → NO. Default to YES; compression is the point.

## What is computed rather than judged
Wrist separation and its trend, arm speed, finger-pose change, and merge feasibility
are measured exactly from the planned chunk and stated to the judge as facts. The
judge is never asked to estimate them. It is asked only what the measurements cannot
say: what kind of object this is, and which phase of the manipulation this is.
