"""method 슬라이드를 진짜 pptx 로 만든다. 그림은 이미지가 아니라 네이티브 도형이다.

아티팩트(HTML)를 슬라이드 앱에 복사하면 텍스트만 넘어간다. 여기서는 박스·화살표·
점을 PowerPoint 도형으로 만들므로, 열어서 집어 옮기고 색을 바꿀 수 있다.
구글 슬라이드에 업로드해도 도형으로 남는다.

    python3 make_method_pptx.py [출력.pptx]

맥에서 돌린다. 서버에 올릴 일이 아니다 -- 슬라이드 작업이 로컬에서 일어난다.
"""
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

OUT = sys.argv[1] if len(sys.argv) > 1 else "method_deck.pptx"

INK = RGBColor(0x1A, 0x1C, 0x21)
MUTED = RGBColor(0x58, 0x5C, 0x64)
FAINT = RGBColor(0x8A, 0x8E, 0x96)
RULE = RGBColor(0xC4, 0xBF, 0xB2)
PAPER = RGBColor(0xF7, 0xF5, 0xF0)
CARD = RGBColor(0xFF, 0xFD, 0xF9)
ACC = RGBColor(0x1F, 0x4E, 0x5F)
ACC_S = RGBColor(0xE2, 0xEC, 0xEE)
RISK = RGBColor(0xA3, 0x3A, 0x22)
RISK_S = RGBColor(0xF6, 0xE8, 0xE3)
STABLE = RGBColor(0x3C, 0x6B, 0x4A)
STABLE_S = RGBColor(0xE6, 0xEF, 0xE8)

# 한글이 들어가므로 라틴과 동아시아 서체를 같이 지정한다. 하나만 지정하면
# PowerPoint 가 한글에 기본 서체를 끼워 넣어 줄마다 굵기가 달라진다.
FONT = "Helvetica Neue"
FONT_EA = "Apple SD Gothic Neo"
MONO = "Menlo"

IN = 914400  # EMU per inch
W, H = int(13.333 * IN), int(7.5 * IN)


def _ea(run, latin, ea):
    """동아시아 서체를 rPr 에 직접 넣는다. python-pptx 가 안 열어 준다."""
    run.font.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.makeelement(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}" + tag.split(":")[1],
            {"typeface": ea})
        rPr.append(el)


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, mono=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.25):
    """runs 는 문자열이거나 (문자열, 옵션dict) 목록이다."""
    tb = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    lines = runs if isinstance(runs, list) else [runs]
    for k, line in enumerate(lines):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        # 한 줄은 문자열이거나, ("글", {옵션}) 한 조각이거나, 그 조각들의 목록이다.
        # 가운데 것을 목록으로 오해하면 옵션 dict 가 글로 들어간다.
        if isinstance(line, list) and len(line) == 2 and isinstance(line[1], dict):
            parts = [line]
        elif isinstance(line, list):
            parts = line
        else:
            parts = [line]
        for part in parts:
            s, o = (part, {}) if isinstance(part, str) else part
            r = p.add_run()
            r.text = s
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
            _ea(r, MONO if o.get("mono", mono) else FONT,
                MONO if o.get("mono", mono) else FONT_EA)
    return tb


def box(slide, x, y, w, h, fill=None, line=RULE, lw=1.0, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = slide.shapes.add_shape(shape, Emu(x), Emu(y), Emu(w), Emu(h))
    sh.shadow.inherit = False
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
    try:
        sh.adjustments[0] = 0.06
    except Exception:
        pass
    sh.text_frame.text = ""
    return sh


def arrow(slide, x1, y1, x2, y2, color=RULE, lw=1.0, dashed=False):
    cn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(x1), Emu(y1),
                                    Emu(x2), Emu(y2))
    cn.line.color.rgb = color
    cn.line.width = Pt(lw)
    ln = cn.line._get_or_add_ln()
    tail = ln.makeelement(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
        {"type": "triangle", "w": "sm", "len": "sm"})
    ln.append(tail)
    if dashed:
        d = ln.makeelement(
            "{http://schemas.openxmlformats.org/drawingml/2006/main}prstDash",
            {"val": "dash"})
        ln.insert(0, d)
    return cn


def dot(slide, cx, cy, r, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, Emu(int(cx - r)), Emu(int(cy - r)),
                                Emu(int(2 * r)), Emu(int(2 * r)))
    sh.shadow.inherit = False
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


prs = Presentation()
prs.slide_width, prs.slide_height = Emu(W), Emu(H)
BLANK = prs.slide_layouts[6]
M = int(0.72 * IN)          # 좌우 여백
TOPY = int(0.62 * IN)


def new(section, num):
    s = prs.slides.add_slide(BLANK)
    bg = box(s, 0, 0, W, H, fill=CARD, line=None, shape=MSO_SHAPE.RECTANGLE)
    bg.shadow.inherit = False
    text(s, M, TOPY, int(6 * IN), int(0.3 * IN), section.upper(), size=10, color=ACC,
         mono=True)
    text(s, W - M - int(2 * IN), TOPY, int(2 * IN), int(0.3 * IN), num, size=10,
         color=FAINT, mono=True, align=PP_ALIGN.RIGHT)
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(M), Emu(TOPY + int(0.3 * IN)),
                                Emu(W - M), Emu(TOPY + int(0.3 * IN)))
    ln.line.color.rgb = RULE
    ln.line.width = Pt(0.75)
    return s


def head(s, t, y=None):
    y = y if y is not None else TOPY + int(0.52 * IN)
    text(s, M, y, W - 2 * M, int(0.8 * IN), t, size=27, bold=True, spacing=1.15)
    return y + int(0.85 * IN)


# ============================================================ 1 표지
s = prs.slides.add_slide(BLANK)
bg = box(s, 0, 0, W, H, fill=CARD, line=None, shape=MSO_SHAPE.RECTANGLE)
text(s, M, int(2.1 * IN), int(9 * IN), int(0.3 * IN),
     "ACTION QUANTIZATION · METHOD", size=11, color=ACC, mono=True)
text(s, M, int(2.55 * IN), int(10.5 * IN), int(1.8 * IN),
     ["압축이 망가뜨린 곳에서", "문항을 뽑는다"], size=40, bold=True, spacing=1.1)
text(s, M, int(4.55 * IN), int(9.6 * IN), int(1.2 * IN),
     "로봇 시연을 빨리 감아 학습에 쓸 때, 어느 순간을 얼마나 감아도 되는지를 VLM 이 "
     "판정한다. 그 판정에 쓸 문항을 장면 인상으로 짓지 않고 측정된 손상에서 끌어내는 "
     "절차와, 그것이 지켜야 할 규칙.", size=15, color=MUTED)
text(s, M, int(5.75 * IN), int(9 * IN), int(0.4 * IN),
     "robocasa 24 태스크 · allex 실로봇 6 구간에 적용", size=12, color=FAINT)

# ============================================================ 2 발단
s = new("발단", "01")
y = head(s, "성공률은 끝에 한 번 찍힌다")
text(s, M, y, int(11 * IN), int(0.8 * IN),
     "태스크 성공률은 처음부터 끝까지 압축한 결과다. 그래서 성공률이 떨어졌다는 사실은 "
     "그 태스크가 압축에 약하다는 것만 말하고, 그 안의 어느 단위 행동이 무너뜨렸는지는 "
     "말하지 않는다.", size=15)
yy = y + int(0.95 * IN)
seq = ["다가가고", "쥐고", "옮기고", "놓고"]
bw, gap = int(1.7 * IN), int(0.32 * IN)
for i, t in enumerate(seq):
    x = M + i * (bw + gap)
    box(s, x, yy, bw, int(0.55 * IN), fill=PAPER)
    text(s, x, yy, bw, int(0.55 * IN), t, size=13, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE)
    if i < 3:
        arrow(s, x + bw, yy + int(0.27 * IN), x + bw + gap, yy + int(0.27 * IN))
x_end = M + 4 * (bw + gap)
box(s, x_end, yy, int(1.9 * IN), int(0.55 * IN), fill=RISK_S, line=RISK)
text(s, x_end, yy, int(1.9 * IN), int(0.55 * IN), "성공 / 실패", size=13, color=RISK,
     align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, x_end, yy + int(0.62 * IN), int(2.4 * IN), int(0.3 * IN), "여기 한 번만 찍힌다",
     size=11, color=RISK)

yy2 = yy + int(1.35 * IN)
for i, (t, d) in enumerate([
        ("문항은 태스크를 가리지 않는다",
         "모든 태스크에 같은 문항을 묻는다. 태스크마다 다른 문항을 물으면 그건 문항이 "
         "아니라 그 태스크의 주석이고, 새 태스크가 오면 못 쓴다."),
        ("문항은 단위 행동을 이름 붙인다",
         "그 국면을 약하게 만드는 바로 그것을 부른다. 곁들여 보이는 자세를 부르면 "
         "문항이 아니라 그 장면의 묘사가 된다.")]):
    x = M + i * int(5.9 * IN)
    box(s, x, yy2, int(5.4 * IN), int(1.35 * IN), fill=PAPER)
    text(s, x + int(0.22 * IN), yy2 + int(0.2 * IN), int(5 * IN), int(0.3 * IN), t,
         size=13.5, bold=True)
    text(s, x + int(0.22 * IN), yy2 + int(0.55 * IN), int(5 * IN), int(0.7 * IN), d,
         size=12, color=MUTED)

# ============================================================ 3 파이프라인 (그림 1)
s = new("전체 얼개 · 그림 1", "02")
y = head(s, "둘로 나뉘는 판단")
BW, BH = int(2.45 * IN), int(0.95 * IN)
yc = y + int(1.25 * IN)          # 가운데 줄
yu, yl = yc - int(1.15 * IN), yc + int(1.15 * IN)

box(s, M, yc, BW, BH, fill=PAPER)
text(s, M + int(0.18 * IN), yc + int(0.16 * IN), int(2.2 * IN), int(0.6 * IN),
     [["압축 손상 측정", {"size": 12.5, "bold": True}],
      ["eval 표 / 운용자 실측", {"size": 10.5, "color": MUTED}]], spacing=1.3)

x2 = M + BW + int(0.55 * IN)
box(s, x2, yu, int(2.9 * IN), int(0.72 * IN), fill=ACC_S, line=ACC)
text(s, x2 + int(0.18 * IN), yu, int(2.6 * IN), int(0.72 * IN), "태스크별 상한·하한",
     size=12.5, anchor=MSO_ANCHOR.MIDDLE)
box(s, x2, yl, int(2.9 * IN), int(0.72 * IN), fill=PAPER)
text(s, x2 + int(0.18 * IN), yl, int(2.6 * IN), int(0.72 * IN), "문항 5개 · 부호 · 가중치",
     size=12.5, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, M + BW, yc + int(0.3 * IN), x2, yu + int(0.36 * IN))
arrow(s, M + BW, yc + int(0.65 * IN), x2, yl + int(0.36 * IN))

x3 = x2 + int(3.15 * IN)
box(s, x3, yl, int(2.3 * IN), int(0.72 * IN), fill=PAPER)
text(s, x3 + int(0.16 * IN), yl, int(2.1 * IN), int(0.72 * IN), "VLM 이 등급을 쓴다",
     size=12.5, anchor=MSO_ANCHOR.MIDDLE)
arrow(s, x2 + int(2.9 * IN), yl + int(0.36 * IN), x3, yl + int(0.36 * IN))

x4 = x3 + int(2.55 * IN)
box(s, x4, yl, int(1.9 * IN), int(0.72 * IN), fill=PAPER)
text(s, x4 + int(0.14 * IN), yl, int(1.7 * IN), int(0.72 * IN), "확신 0~1", size=12.5,
     anchor=MSO_ANCHOR.MIDDLE)
arrow(s, x3 + int(2.3 * IN), yl + int(0.36 * IN), x4, yl + int(0.36 * IN))

x5 = x4 + int(2.15 * IN)
box(s, x5, yc, int(2.3 * IN), BH, fill=ACC_S, line=ACC)
text(s, x5 + int(0.16 * IN), yc + int(0.16 * IN), int(2 * IN), int(0.6 * IN),
     [["배속 / p_yes", {"size": 12.5, "bold": True}],
      ["띠 안의 자리", {"size": 10.5, "color": MUTED}]], spacing=1.3)
arrow(s, x2 + int(2.9 * IN), yu + int(0.36 * IN), x5, yc + int(0.3 * IN),
      color=ACC, dashed=True)
arrow(s, x4 + int(1.9 * IN), yl + int(0.36 * IN), x5, yc + int(0.65 * IN))

box(s, x5, yl, int(2.3 * IN), int(0.72 * IN), fill=PAPER)
text(s, x5 + int(0.16 * IN), yl, int(2 * IN), int(0.72 * IN), "학생 게이트", size=12.5,
     anchor=MSO_ANCHOR.MIDDLE)
arrow(s, x5 + int(1.15 * IN), yc + BH, x5 + int(1.15 * IN), yl)

text(s, x2, yu - int(0.4 * IN), int(5 * IN), int(0.3 * IN),
     "상한은 태스크가 준다", size=11.5, color=ACC)
text(s, x2, yl + int(0.85 * IN), int(5 * IN), int(0.3 * IN),
     "문항은 그 안에서 자리를 정한다", size=11.5, color=MUTED)

# ============================================================ 4 두 풀 (그림 2)
s = new("절차 ① · 그림 2", "03")
y = head(s, "측정된 손상으로 두 풀을 가른다")
DMG = [-.28, -.22, -.18, -.18, -.16, -.16, -.14, -.12, -.10, -.10, -.08, -.08,
       -.06, -.02, -.02, 0, 0, 0, .02, .04, .04, .08, .14, .18]
ax0, ax1 = M + int(0.9 * IN), W - M - int(1.2 * IN)
axy = y + int(1.5 * IN)


def dx(d):
    return ax0 + (d + 0.30) / 0.50 * (ax1 - ax0)


box(s, dx(-0.30), axy - int(0.95 * IN), dx(-0.155) - dx(-0.30), int(0.95 * IN),
    fill=RISK_S, line=None, shape=MSO_SHAPE.RECTANGLE)
box(s, dx(-0.025), axy - int(0.95 * IN), dx(0.20) - dx(-0.025), int(0.95 * IN),
    fill=STABLE_S, line=None, shape=MSO_SHAPE.RECTANGLE)
ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(ax0), Emu(axy), Emu(ax1), Emu(axy))
ln.line.color.rgb = RULE
ln.line.width = Pt(0.75)
seen = {}
for d in DMG:
    seen[d] = seen.get(d, 0) + 1
    c = RISK if d <= -.16 else (STABLE if d >= -.02 else RULE)
    dot(s, dx(d), axy - int(0.14 * IN) - (seen[d] - 1) * int(0.17 * IN),
        int(0.055 * IN), c)
for v, lab in ((-.30, "−0.30"), (-.15, "−0.15"), (0, "0"), (.18, "+0.18")):
    text(s, dx(v) - int(0.35 * IN), axy + int(0.08 * IN), int(0.7 * IN), int(0.25 * IN),
         lab, size=10, color=FAINT, mono=True, align=PP_ALIGN.CENTER)
text(s, ax0, axy + int(0.4 * IN), int(9 * IN), int(0.3 * IN),
     "K2 압축이 성공률에 낸 차이 · 태스크당 50 에피소드", size=11.5, color=MUTED)
text(s, dx(-0.30), axy - int(1.3 * IN), int(2 * IN), int(0.3 * IN), "위험 풀 6",
     size=12.5, color=RISK, bold=True)
text(s, dx(-0.14), axy - int(1.3 * IN), int(1.5 * IN), int(0.3 * IN), "중립 3",
     size=12.5, color=FAINT)
text(s, dx(-0.015), axy - int(1.3 * IN), int(2.4 * IN), int(0.3 * IN), "안정 풀 15",
     size=12.5, color=STABLE, bold=True)

py = axy + int(1.0 * IN)
box(s, M, py, int(5.4 * IN), int(1.1 * IN), fill=RISK_S, line=RISK)
text(s, M + int(0.24 * IN), py + int(0.18 * IN), int(5 * IN), int(0.8 * IN),
     [["감점 문항", {"size": 13.5, "bold": True}],
      ["A 0.667   B 0.333", {"size": 11.5, "color": MUTED, "mono": True}]], spacing=1.35)
box(s, M + int(5.9 * IN), py, int(5.4 * IN), int(1.1 * IN), fill=STABLE_S, line=STABLE)
text(s, M + int(6.14 * IN), py + int(0.18 * IN), int(5 * IN), int(0.8 * IN),
     [["가점 문항", {"size": 13.5, "bold": True}],
      ["C 0.267   D 0.333   E 0.400", {"size": 11.5, "color": MUTED, "mono": True}]],
     spacing=1.35)
arrow(s, dx(-0.23), axy + int(0.72 * IN), dx(-0.23), py)
arrow(s, dx(0.06), axy + int(0.72 * IN), dx(0.06), py)
text(s, M, py + int(1.25 * IN), int(11 * IN), int(0.35 * IN),
     "부호는 어느 풀에서 뽑혔는지가, 가중치는 덮는 태스크 수가 정한다 — 둘 다 도출의 부산물이다.",
     size=12, color=FAINT)

# ============================================================ 5 문항 두 부분
s = new("문항의 조건", "04")
y = head(s, "앞 줄은 일반적으로, 근거는 뒤에")
text(s, M, y, int(11.2 * IN), int(0.8 * IN),
     "문항은 두 부분이다. 앞 줄이 어떤 물건을 어떻게 하고 있는가를 일반적인 말로 쓰고, "
     "뒤에 붙는 절이 그것이 왜 압축에 약한가·강한가를 쓴다. 앞 줄에 과정의 세부를 넣으면 "
     "그 작업장의 주석이 되어 다른 데서 못 쓴다.", size=14.5)
by = y + int(1.0 * IN)
box(s, M, by, int(5.4 * IN), int(2.1 * IN), fill=RISK_S, line=RISK)
text(s, M + int(0.24 * IN), by + int(0.2 * IN), int(5 * IN), int(1.7 * IN),
     [["나쁨", {"size": 12.5, "bold": True, "color": RISK}],
      "",
      ["봉투 살을 한 줌 쥐어 눌러 넘긴다", {"size": 12, "mono": True}],
      "",
      ["각진 상자의 옆면을 잡아 몸쪽으로 끌어온다", {"size": 12, "mono": True}]],
     spacing=1.3)
box(s, M + int(5.9 * IN), by, int(5.4 * IN), int(2.1 * IN), fill=STABLE_S, line=STABLE)
text(s, M + int(6.14 * IN), by + int(0.2 * IN), int(5 * IN), int(1.7 * IN),
     [["좋음", {"size": 12.5, "bold": True, "color": STABLE}],
      "",
      ["쉽게 잡을 수 있는 물건을 뒤집는다", {"size": 12, "mono": True}],
      ["  — 어디를 잡아도 결과가 같다", {"size": 12, "mono": True, "color": MUTED}],
      ["모양이 잡힌 물건을 옮긴다", {"size": 12, "mono": True}],
      ["  — 흔들려도 손에서 안 빠진다", {"size": 12, "mono": True, "color": MUTED}]],
     spacing=1.3)
text(s, M, by + int(2.35 * IN), int(11.2 * IN), int(0.4 * IN),
     "뒷받침 절은 세부를 써도 된다. 거기가 근거를 적는 자리다. "
     "정지 화면이 담는 배치는 등급표가 지는 짐이지 문항이 지는 짐이 아니다.",
     size=12, color=MUTED)

# ============================================================ 6 등급표
s = new("등급표", "05")
y = head(s, "등급은 장면에 붙인다, 확신에 붙이지 않는다")
dead = ["5  확실히 그렇다", "4  대체로 그렇다", "3  반쯤 그렇다",
        "2  대체로 아니다", "1  아니다"]
live = ["5  지금 일어나고 있다", "4  아직인데 한 동작 남았다", "3  그쪽으로 가는 중",
        "2  대상은 있는데 팔은 딴 일", "1  그럴 만한 것이 화면에 없다"]
for i, (title, rows, note, col) in enumerate([
        ("죽는 등급표", dead, "2등급이 9,490칸 중 43개 (0.45%)  ·  눈으로 1과 구별이 안 된다", RISK),
        ("사는 등급표", live, "같은 자리에 볼 수 있는 상태를 넣으면 8.5% 로 산다", STABLE)]):
    x = M + i * int(5.9 * IN)
    box(s, x, y, int(5.4 * IN), int(2.5 * IN), fill=PAPER, line=col)
    text(s, x + int(0.24 * IN), y + int(0.18 * IN), int(5 * IN), int(0.3 * IN), title,
         size=13.5, bold=True, color=col)
    text(s, x + int(0.24 * IN), y + int(0.58 * IN), int(5 * IN), int(1.5 * IN), rows,
         size=12, mono=True, spacing=1.5)
    text(s, x + int(0.24 * IN), y + int(2.05 * IN), int(5 * IN), int(0.35 * IN), note,
         size=10.5, color=MUTED)
ry = y + int(2.75 * IN)
for t in ["중간 등급이 도피처가 되면 안 된다 — “하러 가는 중”은 늘 움직이는 로봇에게 언제나 참이다.",
          "1등급은 “그런 것이 없다”가 아니라 경쟁 문항의 장면이어야 한다.",
          "상위 등급이 그 데이터셋에 실제로 있는 장면이어야 한다. 쓰기 전에 프레임을 뜯어본다."]:
    text(s, M, ry, int(11.2 * IN), int(0.3 * IN), "·  " + t, size=12.5)
    ry += int(0.34 * IN)

# ============================================================ 7 후처리 (그림 3)
s = new("후처리 · 그림 3", "06")
y = head(s, "등급에서 값으로")
text(s, M, y, int(11.2 * IN), int(0.55 * IN),
     [["등급 → 값      (g − 1) / 4", {"size": 12.5, "mono": True}],
      ["값 → 확신      ( 1 + Σw·가점 − Σw·감점 ) / 2", {"size": 12.5, "mono": True}]],
     spacing=1.45)
CELLS = [("Rotate Box", 1.5, 2.0, 1), ("Bring PolyBag", 1.5, 2.5, 1),
         ("Rotate PolyBag", 2.0, 2.5, 1), ("Bring Box", 2.0, 3.0, 0),
         ("Pass Box", 2.0, 3.0, 0), ("Pass PolyBag", 2.0, 3.0, 0)]
bx0, bx1 = M + int(2.3 * IN), W - M - int(1.5 * IN)


def bxp(v):
    return bx0 + (v - 1.0) / 2.0 * (bx1 - bx0)


ty = y + int(0.95 * IN)
for k, (nm, lo, hi, risky) in enumerate(CELLS):
    yy = ty + k * int(0.46 * IN)
    c = RISK if risky else STABLE
    text(s, M, yy - int(0.1 * IN), int(2.1 * IN), int(0.3 * IN), nm, size=12)
    g = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(bxp(1.0)), Emu(yy),
                               Emu(bxp(3.0)), Emu(yy))
    g.line.color.rgb = RULE
    g.line.width = Pt(0.5)
    band = box(s, bxp(lo), yy - int(0.075 * IN), bxp(hi) - bxp(lo), int(0.15 * IN),
               fill=c, line=None)
    band.fill.fore_color.rgb = c
    for q in (.08, .3, .52, .74, .94):
        dot(s, bxp(lo + q * (hi - lo)), yy, int(0.045 * IN), c)
    text(s, bxp(hi) + int(0.12 * IN), yy - int(0.1 * IN), int(1.2 * IN), int(0.3 * IN),
         f"{lo:g}–{hi:g}x", size=10.5, color=MUTED, mono=True)
axl = ty + 6 * int(0.46 * IN) + int(0.12 * IN)
g = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(bxp(1.0)), Emu(axl),
                           Emu(bxp(3.0)), Emu(axl))
g.line.color.rgb = RULE
for v in (1.0, 1.5, 2.0, 2.5, 3.0):
    text(s, bxp(v) - int(0.3 * IN), axl + int(0.06 * IN), int(0.6 * IN), int(0.25 * IN),
         f"{v:g}x", size=10, color=FAINT, mono=True, align=PP_ALIGN.CENTER)
text(s, M, axl + int(0.45 * IN), int(11.2 * IN), int(0.35 * IN),
     "띠는 측정이 준다. 점은 그 칸 안에서 확신 순위대로 놓인 청크다 — 같은 답을 낸 청크는 같은 자리에 묶인다.",
     size=12, color=FAINT)

# ============================================================ 8 검증
s = new("검증", "07")
y = head(s, "합격선은 근거가 있어야 판정이다")
text(s, M, y, int(11.2 * IN), int(0.55 * IN),
     "근거가 “그럴듯한 문장”뿐이면 그건 판정 기준이 아니라 참고 지표다. 그런 선 하나 때문에 "
     "결과물을 계속 고치면, 걸린 쪽 대신 검사를 통과시키려고 문항을 흔들게 된다.", size=14.5)
rows = [("형식 준수", "요청한 형태로 답하는 비율", "판정", STABLE),
        ("부호가 맞는가", "감점은 위험 풀에서, 가점은 안정 풀에서 높은가", "판정", STABLE),
        ("기존 라벨과 수렴", "같은 청크에 몇 %나 같은 값을 주는가", "판정", STABLE),
        ("답이 갈리는가", "어떤 동작은 처음부터 끝까지 정말 비슷할 수 있다", "참고", FAINT),
        ("문항끼리 안 겹치는가", "겹치는 문항을 뺄 때마다 결과가 나빠졌다", "참고", FAINT)]
ry = y + int(0.9 * IN)
for nm, d, tag, col in rows:
    box(s, M, ry, int(11.2 * IN), int(0.62 * IN), fill=PAPER, line=RULE)
    text(s, M + int(0.24 * IN), ry, int(3 * IN), int(0.62 * IN), nm, size=13,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + int(3.3 * IN), ry, int(6.4 * IN), int(0.62 * IN), d, size=12,
         color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + int(10.1 * IN), ry, int(1 * IN), int(0.62 * IN), tag, size=11.5,
         color=col, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    ry += int(0.72 * IN)

# ============================================================ 9 인과성 (그림 4)
s = new("증류 · 그림 4", "08")
y = head(s, "학생이 보는 것")
cy = y + int(1.55 * IN)
lx, mx, rx = M, M + int(6.6 * IN), W - M
box(s, lx, cy - int(0.5 * IN), mx - lx - int(0.12 * IN), int(1.0 * IN),
    fill=ACC_S, line=ACC)
box(s, mx + int(0.12 * IN), cy - int(0.5 * IN), rx - mx - int(0.12 * IN), int(1.0 * IN),
    fill=RISK_S, line=RISK)
text(s, lx + int(0.3 * IN), cy - int(0.36 * IN), int(6 * IN), int(0.7 * IN),
     [["입력에 들어간다", {"size": 13.5, "bold": True}],
      ["state[f]     action[f−16 : f]", {"size": 12, "mono": True, "color": MUTED}]],
     spacing=1.4)
text(s, mx + int(0.42 * IN), cy - int(0.36 * IN), int(4.5 * IN), int(0.7 * IN),
     [["절대 안 들어간다", {"size": 13.5, "bold": True, "color": RISK}],
      ["action[f] 이후", {"size": 12, "mono": True, "color": MUTED}]], spacing=1.4)
v = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(mx), Emu(cy - int(0.95 * IN)),
                           Emu(mx), Emu(cy + int(0.75 * IN)))
v.line.color.rgb = INK
v.line.width = Pt(1.75)
text(s, mx - int(0.3 * IN), cy - int(1.3 * IN), int(0.6 * IN), int(0.3 * IN), "f",
     size=13, mono=True, align=PP_ALIGN.CENTER)
text(s, mx + int(0.35 * IN), cy - int(1.3 * IN), int(4 * IN), int(0.3 * IN),
     "지금 판단하는 관측", size=11.5, color=MUTED)
text(s, M, cy + int(1.0 * IN), int(11.2 * IN), int(0.7 * IN),
     "추론에서는 env.step 이 실행된 뒤에 실제로 보낸 명령만 기록한다 — 방금 생성한 미래 "
     "청크를 미리 넣으면 미래 정보가 새어 든다. 학습(오프라인)과 추론(온라인)이 같은 "
     "경계를 지켜야 한다.", size=13)
py = cy + int(2.0 * IN)
for i, (t, rows_) in enumerate([
        ("기준 학생  317,249", ["현재 이미지 3뷰", "지시문 임베딩"]),
        ("제안 학생  346,369  (+9.2%)",
         ["현재 이미지 3뷰", "지시문 임베딩", "현재 proprio 53", "직전 실행된 명령 16×12"])]):
    x = M + i * int(5.9 * IN)
    box(s, x, py, int(5.4 * IN), int(1.5 * IN), fill=PAPER)
    text(s, x + int(0.24 * IN), py + int(0.16 * IN), int(5 * IN), int(0.3 * IN), t,
         size=13, bold=True)
    text(s, x + int(0.24 * IN), py + int(0.52 * IN), int(5 * IN), int(0.9 * IN), rows_,
         size=11.5, color=MUTED, mono=True, spacing=1.3)

# ============================================================ 10 정리
s = new("정리", "09")
y = head(s, "이 방법이 주장하는 것")
claims = [
    "문항은 발명하지 않는다. 측정된 압축 손상이 태스크를 두 풀로 가르고, 문항은 그 풀들의 공통 국면에서 나온다.",
    "모델에게는 한 가지만 맡긴다. 어느 태스크인지, 상한이 얼마인지는 주석과 측정이 답한다.",
    "합격선은 근거에서 나온다. 지어낸 선은 참고로 내린다.",
    "고치는 절차 자체가 명세다. 무엇을 얼리고 무엇을 지목하는지가 미리 적혀 있어야 재현된다.",
    "같은 절차가 시뮬과 실로봇에 그대로 간다. 바뀌는 것은 상한이 어디서 오는가뿐이다.",
]
ry = y + int(0.2 * IN)
for i, c in enumerate(claims, 1):
    n = box(s, M, ry, int(0.36 * IN), int(0.36 * IN), fill=None, line=ACC,
            shape=MSO_SHAPE.OVAL)
    text(s, M, ry, int(0.36 * IN), int(0.36 * IN), str(i), size=11, color=ACC,
         mono=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, M + int(0.58 * IN), ry, int(10.6 * IN), int(0.7 * IN), c, size=14.5)
    ry += int(0.86 * IN)

prs.save(OUT)
print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} 장 -> {OUT}")
