"""논문 overview figure 를 PPTX 한 장으로 만든다 (AutoML-Agent 스타일).

뼈대는 **네이티브 도형**이다 -- 둥근 사각형, 화살표, 반투명 묶음 패널, 세로로
돌린 레인 이름, 점선 구분선. PowerPoint 에서 열면 낱개로 잡히고 Google Slides 에
올려도 그대로 편집된다. 거기에 **납작한 색 일러스트**를 얹는다(AutoML-Agent 그림의
라마·로봇 자리). 그림 생성 모델은 안 쓴다 -- 도식의 글자를 뭉갠다.

구조는 가로 레인 셋이고, 왼쪽에 세로 글씨로 이름이 붙는다.

    (1) MEASURE   배속별로 돌려 태스크별 손상을 잰다
    (2) DERIVE    두 풀로 가르고, LLM 이 문항을 쓰고, 순위·부호·가중치를 준다
    (3) LABEL     VLM 이 매 순간 등급을 매기고 신뢰도를 띠에 앉힌다

검증 루프는 주황색 `fail` 화살표로 (2) 로 돌아간다 -- 형식 준수·부호 일치·
이전 라벨과의 일치. 통과해야 (3) 으로 내려간다.

일러스트는 Microsoft Fluent Emoji(MIT) 의 256px 투명 PNG 다. `vlm_gate/assets/icons/`
에 받아 두고 그림 파일로 넣으므로 어느 기계에서 열어도 같게 보인다(이모지 **글자**로
넣으면 기계마다 다른 꼴이 나온다). 없으면 `tools/fetch_icons.sh` 로 다시 받는다.

무리를 물건 그림으로 보여 준다 -- 버터를 바구니에, 그릇을 접시에. 문항이 무엇을
근거로 나왔는지가 글보다 그림에서 먼저 읽힌다.

    python vlm_gate/scripts/make_overview_pptx.py [out.pptx]

맥에서 돌린다(슬라이드 작업이 로컬이므로). python-pptx 가 필요하다.
"""
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

OUT = sys.argv[1] if len(sys.argv) > 1 else "method_overview.pptx"

# ---- 색 -------------------------------------------------------------------
# 참조 그림과 같은 역할 분담: 묶음 패널은 연보라, 되먹임은 주황, 안쪽 목록은 초록.
INK = RGBColor(0x14, 0x16, 0x1A)
MUTED = RGBColor(0x55, 0x59, 0x61)
RULE = RGBColor(0x9A, 0x9E, 0xA6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

PANEL = RGBColor(0xDE, 0xDF, 0xF2)      # 묶음 패널 (연보라)
PANEL_L = RGBColor(0xC2, 0xC4, 0xE4)
NAVY = RGBColor(0x1E, 0x24, 0x5E)       # 패널 제목

WARM = RGBColor(0xFB, 0xDF, 0xC7)       # 위험 / 되먹임 상자
WARM_L = RGBColor(0xE8, 0x8B, 0x3C)
WARM_T = RGBColor(0x8A, 0x43, 0x0C)

COOL = RGBColor(0xD6, 0xEA, 0xF6)       # 안정
COOL_L = RGBColor(0x4A, 0x8F, 0xC0)
COOL_T = RGBColor(0x14, 0x4E, 0x75)

MINT = RGBColor(0xE7, 0xF4, 0xEC)       # 안쪽 목록 묶음
MINT_L = RGBColor(0x5E, 0xA8, 0x7B)
MINT_T = RGBColor(0x1E, 0x5E, 0x3A)

PASS_C = RGBColor(0x1E, 0x7A, 0x3C)
FAIL_C = RGBColor(0xC4, 0x3B, 0x14)

FONT = "Helvetica Neue"
FONT_EA = "Apple SD Gothic Neo"
MONO = "Menlo"

IN = 914400
W, H = int(16.0 * IN), int(8.70 * IN)    # 두 단 폭에 맞는 가로 그림


def _ea(run, latin=FONT, ea=FONT_EA):
    """라틴과 동아시아 글꼴을 같이 박는다. 한쪽만 두면 한글이 다른 꼴로 나온다."""
    run.font.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:cs"):
        e = rPr.find(qn(tag))
        if e is None:
            e = rPr.makeelement(qn(tag), {})
            rPr.append(e)
        e.set("typeface", latin)
    e = rPr.find(qn("a:ea"))
    if e is None:
        e = rPr.makeelement(qn("a:ea"), {})
        rPr.append(e)
    e.set("typeface", ea)


def text(sl, x, y, w, h, runs, size=13, color=INK, bold=False, mono=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False, space=1.15):
    """runs 는 문자열 하나이거나 (문자열, dict) 목록."""
    tb = sl.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tb.text_frame.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [(runs, {})]
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = space
    for s, o in runs:
        if s == "\n":
            p = tf.add_paragraph()
            p.alignment = align
            p.line_spacing = space
            continue
        r = p.add_run()
        r.text = s
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", italic)
        r.font.color.rgb = o.get("color", color)
        _ea(r, MONO if o.get("mono", mono) else FONT, FONT_EA)
    return tb


def box(sl, x, y, w, h, fill=WHITE, line=RULE, lw=1.25, radius=None,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = sl.shapes.add_shape(shape, Emu(x), Emu(y), Emu(w), Emu(h))
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sh.adjustments[0] = radius
        except Exception:
            pass
    return sh


def label(sl, sh, s, size=13, color=INK, bold=True, mono=False):
    """도형 한가운데에 글씨. 도형의 text_frame 을 쓰면 세로 가운데가 맞는다."""
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(int(0.06 * IN))
    tf.margin_top = tf.margin_bottom = Emu(int(0.03 * IN))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    lines = s.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 1.05
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        _ea(r, MONO if mono else FONT, FONT_EA)
    return sh


def conn(sl, pts, color=INK, lw=2.5, head=True, dashed=False):
    """꺾인 화살표. pts 는 (x, y) 목록이고 마지막 마디에 화살촉이 붙는다."""
    last = None
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        ln = sl.shapes.add_connector(1, Emu(int(x1)), Emu(int(y1)),
                                     Emu(int(x2)), Emu(int(y2)))
        ln.line.color.rgb = color
        ln.line.width = Pt(lw)
        if dashed:
            ln.line.dash_style = 4
        last = ln
    if head and last is not None:
        lnPr = last.line._get_or_add_ln()
        e = lnPr.makeelement(qn("a:tailEnd"),
                             {"type": "triangle", "w": "med", "len": "med"})
        lnPr.append(e)
    return last


def vline(sl, x, y1, y2, color=RULE, lw=1.5, dashed=True):
    ln = sl.shapes.add_connector(1, Emu(int(x)), Emu(int(y1)), Emu(int(x)), Emu(int(y2)))
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    if dashed:
        ln.line.dash_style = 4
    return ln


def hline(sl, x1, x2, y, color=RULE, lw=1.5, dashed=True):
    ln = sl.shapes.add_connector(1, Emu(int(x1)), Emu(int(y)), Emu(int(x2)), Emu(int(y)))
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    if dashed:
        ln.line.dash_style = 4
    return ln


def lane_name(sl, x, y, h, s):
    """왼쪽 세로 레인 이름. 텍스트 상자를 270도 돌린다."""
    tb = sl.shapes.add_textbox(Emu(int(x)), Emu(int(y)), Emu(int(h)), Emu(int(0.42 * IN)))
    tb.rotation = 270
    tf = tb.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = s
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = INK
    _ea(r)
    return tb


# ---- 아이콘 ---------------------------------------------------------------
# Microsoft Fluent Emoji (MIT). 투명 배경 256px PNG 라 논문 그림에 그대로 쓴다.
# 이모지 **글자**가 아니라 그림 파일이므로 PowerPoint·Slides 어디서나 같게 보인다.
# 없으면 `tools/fetch_icons.sh` 로 다시 받는다.
ICONS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "assets", "icons")


def icon(sl, cx, cy, name, size=0.62):
    """(cx, cy) 를 한가운데로 아이콘 PNG 를 놓는다. size 는 인치."""
    path = os.path.join(ICONS, name + ".png")
    if not os.path.exists(path):
        print(f"[!] 아이콘 없음: {path}")
        return None
    w = int(size * IN)
    return sl.shapes.add_picture(path, Emu(int(cx - w / 2)), Emu(int(cy - w / 2)),
                                 Emu(w), Emu(w))


def chip(sl, x, y, items, gap=0.40, size=0.34):
    """물건 아이콘을 가로로 늘어놓는다. 무리가 무엇으로 이루어졌는지 눈으로 보인다."""
    cx = x
    for nm in items:
        icon(sl, cx, y, nm, size)
        cx += int(gap * IN)
    return cx


# ===========================================================================
prs = Presentation()
prs.slide_width, prs.slide_height = Emu(W), Emu(H)
s = prs.slides.add_slide(prs.slide_layouts[6])


def I(v):
    return int(v * IN)


L = I(0.60)                 # 레인 이름이 앉는 자리
X0 = I(1.16)                # 본문 왼쪽
XE = W - I(0.34)

LANE = [(I(0.26), I(2.62)),      # (1) MEASURE
        (I(2.76), I(6.16)),      # (2) DERIVE
        (I(6.30), I(8.50))]      # (3) LABEL

for (y1, y2), nm in zip(LANE, ["(1) MEASURE", "(2) DERIVE", "(3) LABEL"]):
    lane_name(s, L - (y2 - y1) // 2 + I(0.21), y1 + (y2 - y1) // 2 - I(0.21),
              y2 - y1, nm)
hline(s, I(0.22), XE, LANE[0][1] + I(0.07))
hline(s, I(0.22), XE, LANE[1][1] + I(0.07))

# ---------------------------------------------------------------- 레인 1
y = LANE[0][0]
MID = y + I(0.86)
icon(s, X0 + I(0.34), MID - I(0.10), "robot", 0.72)
text(s, X0 - I(0.06), MID + I(0.32), I(0.80), I(0.28), "policy",
     size=11, color=MUTED, align=PP_ALIGN.CENTER)

b = box(s, X0 + I(0.96), MID - I(0.42), I(2.26), I(0.84))
label(s, b, "Roll out at fixed\ncompression ratios", size=12)
text(s, X0 + I(0.96), MID + I(0.48), I(2.26), I(0.28),
     "1.0x · 1.7x · 2.0x · 2.5x", size=10.5, color=MUTED, mono=True,
     align=PP_ALIGN.CENTER)

icon(s, X0 + I(3.50), MID, "videogame", 0.46)
conn(s, [(X0 + I(0.76), MID), (X0 + I(0.96), MID)])
conn(s, [(X0 + I(3.22), MID), (X0 + I(3.26), MID)], head=False)
conn(s, [(X0 + I(3.74), MID), (X0 + I(4.00), MID)])

# --- 손상표
PX, PW = X0 + I(4.00), I(5.20)
box(s, PX, y + I(0.06), PW, I(2.30), fill=PANEL, line=PANEL_L)
icon(s, PX + I(0.34), y + I(0.30), "chart", 0.34)
text(s, PX + I(0.56), y + I(0.18), PW, I(0.30),
     "(a) Measured damage — success rate per task family",
     size=12.5, color=NAVY, bold=True)

HD = ["ratio", "spatial", "object", "goal", "long-h."]
ROWS = [("1.0x", "0.98", "0.99", "0.96", "0.89", False),
        ("1.7x", "0.96", "0.99", "0.95", "0.86", False),
        ("2.0x", "0.82", "0.97", "0.94", "0.82", False),
        ("2.5x", "0.41", "0.82", "0.62", "0.54", True)]
cw = I(0.94)
tx, ty = PX + I(0.26), y + I(0.62)
for j, hd in enumerate(HD):
    text(s, tx + j * cw, ty, cw, I(0.26), hd, size=11, color=MUTED, bold=True,
         align=PP_ALIGN.CENTER)
for i, row in enumerate(ROWS):
    ry = ty + I(0.32) + i * I(0.30)
    if row[5]:
        box(s, tx - I(0.08), ry - I(0.05), cw * 5 + I(0.16), I(0.30),
            fill=WARM, line=WARM_L, lw=1.0)
    for j, v in enumerate(row[:5]):
        text(s, tx + j * cw, ry, cw, I(0.26), v, size=11.5,
             color=INK if row[5] else MUTED, bold=row[5], mono=True,
             align=PP_ALIGN.CENTER)
text(s, PX + I(0.26), y + I(2.02), PW - I(0.5), I(0.26),
     "widest spread — split the tasks here", size=10.5, color=WARM_T,
     italic=True)

# --- 상한 띠
CBX = PX + PW + I(0.30)
cb = box(s, CBX, y + I(0.66), XE - CBX, I(1.10), fill=COOL, line=COOL_L)
label(s, cb, "per-task ceiling band\n[ lo , hi ]", size=13, color=COOL_T)
text(s, CBX, y + I(1.82), int((XE - CBX) * 0.66), I(0.28),
     "measurement gives the ceiling", size=10.5, color=COOL_T, italic=True,
     align=PP_ALIGN.CENTER)
conn(s, [(PX + PW, y + I(1.21)), (CBX, y + I(1.21))])

# ---------------------------------------------------------------- 레인 2
y = LANE[1][0]
BW = I(3.10)
PH = I(1.24)


def pool(py, title, fill, line, tcol, items, note):
    box(s, X0, py, BW, PH, fill=fill, line=line)
    text(s, X0 + I(0.16), py + I(0.10), BW - I(0.32), I(0.28), title,
         size=13, color=tcol, bold=True)
    iy = py + I(0.66)
    for k, (a, arrow_txt, bnm, val) in enumerate(items):
        ox = X0 + I(0.30) + k * I(1.52)
        icon(s, ox, iy, a, 0.36)
        if bnm:
            conn(s, [(ox + I(0.22), iy), (ox + I(0.40), iy)], color=line, lw=1.6)
            icon(s, ox + I(0.62), iy, bnm, 0.36)
            vx = ox + I(0.84)
        else:
            text(s, ox + I(0.24), iy - I(0.11), I(0.5), I(0.24), arrow_txt,
                 size=10, color=tcol, bold=True)
            vx = ox + I(0.62)
        text(s, vx, iy - I(0.10), I(0.56), I(0.22), val, size=10,
             color=tcol, bold=True, mono=True)
    text(s, X0 + I(0.16), py + I(0.98), BW - I(0.32), I(0.24), note,
         size=10, color=tcol, italic=True, align=PP_ALIGN.CENTER)


pool(y + I(0.10), "RISK pool", WARM, WARM_L, WARM_T,
     [("butter", None, "basket", "0.02"), ("bowl", None, "plate", "0.10")],
     "a small fixed spot it must sit squarely on")
pool(y + I(1.52), "STABLE pool", COOL, COOL_L, COOL_T,
     [("canned", None, "basket", "1.00"), ("plate", "push", None, "1.00")],
     "caught by a container, or never held")

# --- LLM
LX = X0 + BW + I(0.74)
icon(s, LX, y + I(1.02), "memo", 0.76)
text(s, LX - I(0.60), y + I(1.46), I(1.20), I(0.28), "LLM",
     size=12.5, color=INK, bold=True, align=PP_ALIGN.CENTER)
text(s, LX - I(0.68), y + I(2.34), I(1.36), I(0.46),
     "task instructions\nonly", size=10, color=MUTED, align=PP_ALIGN.CENTER)
conn(s, [(X0 + BW, y + I(0.72)), (LX - I(0.32), y + I(0.72)),
         (LX - I(0.32), y + I(0.86))])
conn(s, [(X0 + BW, y + I(2.14)), (LX - I(0.32), y + I(2.14)),
         (LX - I(0.32), y + I(1.20))])

# --- 후보 문항
QX = LX + I(0.62)
QW = I(6.30)
box(s, QX, y + I(0.06), QW, I(2.62), fill=PANEL, line=PANEL_L)
text(s, QX + I(0.20), y + I(0.18), QW, I(0.30),
     "(b) Candidate questions — each names one unit action",
     size=12.5, color=NAVY, bold=True)
conn(s, [(LX + I(0.40), y + I(1.02)), (QX, y + I(1.02))])

qy = y + I(0.58)
box(s, QX + I(0.20), qy, QW - I(0.40), I(0.72), fill=WHITE, line=WARM_L)
text(s, QX + I(0.36), qy + I(0.09), QW - I(0.72), I(0.58),
     [("PENALTY   ", {"bold": True, "color": WARM_T, "size": 10.5}),
      ("Is the object being set down on a small fixed spot it must sit "
       "squarely on — a plate, a burner, a rack — rather than into something "
       "that would catch it?", {"size": 11})], size=11)

qy = y + I(1.40)
box(s, QX + I(0.20), qy, QW - I(0.40), I(0.60), fill=WHITE, line=COOL_L)
text(s, QX + I(0.36), qy + I(0.08), QW - I(0.72), I(0.48),
     [("BONUS   ", {"bold": True, "color": COOL_T, "size": 10.5}),
      ("Is this a job that needs no hold kept on the object — one push and "
       "it carries on where it was sent?", {"size": 11})], size=11)

qy = y + I(2.10)
box(s, QX + I(0.20), qy, QW - I(0.40), I(0.48), fill=WHITE, line=RULE, lw=1.0)
text(s, QX + I(0.36), qy + I(0.08), QW - I(0.72), I(0.36),
     [("REJECTED   ", {"bold": True, "color": RULE, "size": 10.5}),
      ("…goes into an enclosed space — a drawer, a cabinet?    "
       "covers both pools, so it separates nothing",
       {"size": 10.5, "color": MUTED})], size=10.5)

# --- 점수
SX = QX + QW + I(0.32)
box(s, SX, y + I(0.06), XE - SX, I(2.62), fill=MINT, line=MINT_L)
text(s, SX + I(0.16), y + I(0.18), XE - SX, I(0.30), "(c) Score",
     size=12.5, color=MINT_T, bold=True)
for i, (k, v) in enumerate([("rank", "own pool − other pool"),
                            ("sign", "set by pool of origin"),
                            ("weight", "tasks covered, normalised")]):
    bb = box(s, SX + I(0.16), y + I(0.60) + i * I(0.68), XE - SX - I(0.32),
             I(0.58), fill=WHITE, line=MINT_L, lw=1.0)
    label(s, bb, k + "\n" + v, size=10, color=INK)

# --- 검증 루프 (레인 2 안의 제 줄)
VY = y + I(2.84)
VX, VW = QX + I(1.40), I(3.60)
vb = box(s, VX, VY, VW, I(0.46), fill=WARM, line=WARM_L)
label(s, vb, "verify    format ≥99% · sign · agreement", size=11.5, color=WARM_T)
icon(s, VX - I(0.28), VY + I(0.23), "mag", 0.38)
conn(s, [(VX + VW // 2, y + I(2.68)), (VX + VW // 2, VY)])
# fail -> 두 풀로
conn(s, [(VX, VY + I(0.23)), (X0 - I(0.34), VY + I(0.23)),
         (X0 - I(0.34), y + I(1.42)), (X0, y + I(1.42))], color=WARM_L, lw=2.5)
text(s, X0 + I(3.30), VY - I(0.04), I(1.4), I(0.26), "fail — rewrite",
     size=11, color=FAIL_C, bold=True, italic=True)
# pass -> 레인 3
conn(s, [(VX + VW, VY + I(0.23)), (VX + VW + I(0.60), VY + I(0.23)),
         (VX + VW + I(0.60), LANE[2][0] + I(0.22))], color=INK, lw=2.5)
text(s, VX + VW + I(0.14), VY - I(0.24), I(1.0), I(0.26), "pass",
     size=11, color=PASS_C, bold=True, italic=True)

# ---------------------------------------------------------------- 레인 3
y = LANE[2][0] + I(0.30)
MID = y + I(0.62)
icon(s, X0 + I(0.34), MID - I(0.08), "eyes", 0.68)
text(s, X0 - I(0.06), MID + I(0.32), I(0.80), I(0.28), "VLM",
     size=12, color=INK, bold=True, align=PP_ALIGN.CENTER)

b = box(s, X0 + I(0.96), MID - I(0.46), I(2.50), I(0.92))
label(s, b, "grade every question\nat every moment", size=12)
text(s, X0 + I(0.96), MID + I(0.52), I(2.50), I(0.28),
     "1–5, each question its own rubric", size=10.5, color=MUTED,
     align=PP_ALIGN.CENTER)
conn(s, [(X0 + I(0.76), MID), (X0 + I(0.96), MID)])

CX, CW = X0 + I(3.80), I(5.60)
box(s, CX, y - I(0.14), CW, I(1.66), fill=PANEL, line=PANEL_L)
text(s, CX + I(0.20), y - I(0.02), CW, I(0.30),
     "(d) Combine, then place inside that task's band",
     size=12.5, color=NAVY, bold=True)
text(s, CX + I(0.20), y + I(0.36), CW - I(0.40), I(0.32),
     "confidence = ( 1 + Σ w·bonus − Σ w·penalty ) / 2", size=12.5,
     color=INK, mono=True, align=PP_ALIGN.CENTER)
text(s, CX + I(0.20), y + I(0.74), CW - I(0.40), I(0.30),
     "an ordering, not a ratio — placed by quantile within the task",
     size=10.5, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
text(s, CX + I(0.20), y + I(1.08), CW - I(0.40), I(0.32),
     "ratio here = band_place( confidence, lo, hi )", size=12.5,
     color=INK, mono=True, align=PP_ALIGN.CENTER)
conn(s, [(X0 + I(3.46), MID), (CX, MID)])

# 상한 띠가 레인 1 에서 오른쪽 가장자리를 타고 내려온다
RX = XE - I(0.18)
DX = CBX + int((XE - CBX) * 0.84)
conn(s, [(DX, LANE[0][0] + I(1.76)),
         (DX, LANE[0][1] - I(0.08)),
         (RX, LANE[0][1] - I(0.08)), (RX, y + I(1.58)),
         (CX + CW - I(1.40), y + I(1.58)), (CX + CW - I(1.40), y + I(1.52))],
     color=COOL_L, lw=2.2, dashed=True)

# --- 출력
OX = CX + CW + I(0.36)
OW = XE - OX - I(0.30)
icon(s, OX + I(0.14), y + I(0.02), "stopwatch", 0.32)
text(s, OX + I(0.36), y - I(0.10), OW, I(0.30), "(e) One episode",
     size=12.5, color=NAVY, bold=True)
SEG = [(0.20, COOL), (0.13, PANEL), (0.10, WARM), (0.21, COOL),
       (0.14, PANEL), (0.10, WARM), (0.12, COOL)]
sx = OX
for frac, col in SEG:
    bw = int(OW * frac)
    box(s, sx, y + I(0.40), bw - I(0.025), I(0.48), fill=col, line=None,
        shape=MSO_SHAPE.RECTANGLE)
    sx += bw
text(s, OX, y + I(0.96), OW, I(0.30),
     [("2.5x", {"color": COOL_T, "bold": True, "size": 10.5, "mono": True}),
      (" free transit        ", {"color": MUTED, "size": 10}),
      ("1.0x", {"color": WARM_T, "bold": True, "size": 10.5, "mono": True}),
      (" careful placement", {"color": MUTED, "size": 10})], size=10)
conn(s, [(CX + CW, MID), (OX, MID)])

prs.save(OUT)
print(f"-> {OUT}")
