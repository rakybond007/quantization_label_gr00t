"""장면 문항 -- 에피소드 하나의 시작 장면이 얼마나 압축을 견딜지 묻는다.

앞선 배속 문항 아홉 판(v1a~sa2)은 전부 **태스크당 순위상관**으로 채점했다. 하네스
R1 이 금지한 방식이고, R1 이 예고한 대로 라벨이 태스크 상수로 수렴했다 -- sa1 은
2.2~2.5 범위, sa2 는 69% 가 한 답으로 붕괴.

그리고 태스크 축은 애초에 작다. pin4 폐루프 75 에피의 성패 분산을 나누면 태스크
간이 6.9%, 태스크 안이 93.1% 다. 같은 태스크의 20 에피소드에 같은 배속 스케줄을
먹였는데 절반만 성공한다 -- 나머지를 가르는 것은 장면이다.

장면은 **에피소드마다 변하는 유일한 입력**이다. 지시문은 태스크당 하나로 고정이고,
액션 계산은 그리퍼 상태를 줄 뿐 물체가 어디에 어떻게 놓였는지는 모른다. 여기서는
VLM 말고 볼 수 있는 것이 없다.

문항은 R6 대로 그림을 먼저 보고 골랐다. PnPCounterToSink 에서 실패한 ep0 은 키 큰
캔이 싱크 안쪽 벽에 붙어 서 있었고, 성공한 ep1 은 납작한 물체가 트인 조리대에
누워 있었다. OpenDoubleDoor 는 성공 장면에서 문이 화면에 잡히고 실패 장면에서는
화면 위로 벗어나 있었다. 거리 · 좁음 · 물체 크기 세 가지가 눈에 띄었고 그 셋만
묻는다.

    A REACH     닿아야 할 것이 멀리 있는가
    B CONFINED  그것이 좁은 자리에 있는가
    C SMALL     그것이 작고 잘 흐트러지는가

셋 다 위험쪽이다 -- 등급이 높을수록 압축에 약하다.
"""
NGRADE = 5
SIGN = {"A": -1, "B": -1, "C": -1}
WEIGHT = {"A": 0.30, "B": 0.40, "C": 0.30}
NAME = {"A": "REACH", "B": "CONFINED", "C": "SMALL"}

GUIDANCE = (
    "You are looking at the opening frame of one attempt at a kitchen task. The same "
    "robot will run the same task many times; what differs between attempts is only how "
    "this particular scene is laid out -- where the object happens to sit, how much room "
    "is around it, how far the arm has to go.\n\n"
    "The motion will be played back faster than it was planned, with steps merged "
    "together. Merging steps costs accuracy: the arm cuts corners, overshoots a little, "
    "and cannot correct as often. Some layouts absorb that and some do not.\n\n"
    "Judge this layout only. Do not judge the task in general, and do not judge how hard "
    "the task would be at normal speed -- judge how much this particular arrangement "
    "would suffer from a coarser, faster motion."
)

ASK = (
    "Answer each check on its own line as \"A) 3\", in order, nothing else -- one digit "
    "from 1 to 5 per check:\n"
    "  5 = clearly true of this scene   4 = mostly   3 = partly   2 = barely\n"
    "  1 = not true at all\n"
    "A) Is the thing the hand must reach FAR from where the hand starts -- across the\n"
    "   counter, high up, at the edge of view, or out of view -- rather than already\n"
    "   close in front of the hand?\n"
    "B) Is that thing in a CONFINED spot -- inside a basin or cabinet, pressed against a\n"
    "   wall or a raised edge, wedged between other things -- rather than standing clear\n"
    "   on an open surface?\n"
    "C) Is that thing SMALL OR EASILY UPSET -- narrow, tall and tippy, thin-walled, or\n"
    "   light enough to slide away -- rather than large, low and steady?"
)
