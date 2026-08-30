---
name: deliverable-format-match
description: "Meeting materials must MATCH the user's uploaded deck format (pptx, 0707 style) — not web pages"
metadata:
  type: feedback
---

When the user asks to summarize progress "슬라이드로", the deliverable must follow their
EXISTING deck format (see `Action_quantization_0707.pdf` in the workspace): 16:9 pptx,
white background, bold blue left-aligned titles ("Recap : …", "Method", "Experimental
Results (N) - <Bench> w <Judge>", "TODO"), black ●/○ bullets, booktabs-style tables with
the best row highlighted green, cover slide = centered blue title + name + date.

**Why:** they present by appending to this deck; a styled web page (artifact) was rejected
("당연히 그 포맷을 유지해야") — artifacts are fine as an EXTRA, never the primary deliverable
for meeting slides. Outputs belong in the workspace (`~/quantization_agent_workspace/docs/`),
not only in session scratchpads.

**How to apply:** build with python-pptx (installed, user site); generator script kept at
`docs/make_update_deck.py` — clone its helpers (title/bullets/table, BLUE=1F78C8,
GREEN=E2EFDA highlight) for future updates.
