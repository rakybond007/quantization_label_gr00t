"""배속 문항 sa2 -- 지시문이 후보를 좁히고 장면이 그중 하나를 고른다.

**sa1 이 왜 부족했나.** 문항 B("좁고 정확한 자리에 놓으려는가")가 목적지가 화면에
들어왔을 때만 켜졌다. 같은 태스크·같은 물체인데 손목 카메라에 접시가 잡히면
eB 2.8, 안 잡히면 1.3 이었다. 목적지는 조작 후반에만 보이므로 한 장면만으로는
"이 에피소드가 어디에 놓을 것인가" 를 알 수 없다.

그 정보는 **지시문에 있다.** 그래서 지시문에서 이 태스크가 지나갈 서브액션 목록을
뽑아 **후보로 제시하고**, VLM 은 그중 지금이 어느 단계인지만 고른다. 지시문을
통째로 주면 모델이 그것만 읽고 그림을 무시하지만(태스크 안 답 변동이 0.08 배로
죽었다), 후보를 좁히는 용도로만 쓰면 그 문제가 없다 -- 고르는 일은 화면을 봐야
한다.

    지시문  이 태스크가 지나갈 서브액션 목록
    장면    지금이 그중 어느 것인가          <- VLM 이 하는 일
    액션    쥠/놓음·속도·변위                <- 계산 사실
"""
NGRADE = 5          # 후보는 최대 5개. 등급 i = i번째 후보.
NAME = {"A": "STAGE"}
SIGN = {"A": +1}
WEIGHT = {"A": 1.0}

# 지시문 분해에서 쓰는 어휘와 같다. 한계는 사다리에서 역산한 값이다.
LIMITS = {
    "place_precise": 1.25, "pull_swing": 1.67, "carry": 1.83,
    "place_catcher": 2.37, "grasp_confined": 2.61, "place_open": 2.67,
    "contact_knob": 3.17, "contact_coarse": 3.26,
    "grasp_open": 4.00,
}
# 사람이 읽을 이름. 후보 목록을 프롬프트에 이 문구로 적는다.
PHRASE = {
    "grasp_open":     "reaching for something that stands clear, and closing on it",
    "grasp_confined": "reaching into a narrow place -- a microwave, a sink, a cabinet, "
                      "a pan, under a machine -- for what is inside",
    "carry":          "carrying something already held, with room around it",
    "place_catcher":  "putting what is held into something that catches it -- a basin, "
                      "a bin, an open cabinet",
    "place_precise":  "setting what is held down on a small exact spot -- a plate, a "
                      "rack, a burner, under a machine",
    "pull_swing":     "holding a handle and swinging a door or drawer open",
    "place_open":     "setting what is held down on a wide open surface -- a "
                      "countertop -- where it only has to land flat",
    "contact_knob":   "taking hold of a knob or tap and turning it in place",
    "contact_coarse": "pushing or pressing a fixture that moves as one piece -- a "
                      "drawer face, a door, a button, a knob",
}

GUIDANCE = (
    "You are watching one moment of a robot arm in a kitchen. Below are the stages this "
    "particular job passes through, in order, and measurements computed from the planned "
    "motion. The measurements tell you whether the gripper closes or opens in this "
    "stretch, how fast the arm moves, and how far a merged step would jump -- do not "
    "re-estimate them.\n\n"
    "Your one job is to say **which of the listed stages this moment belongs to.** "
    "Read the picture: what is in the hand, what the hand is near, whether the fingers "
    "are on something or empty. The list is what this job does; the picture is where in "
    "that list the arm is right now."
)


def ask(stages):
    """후보 목록을 받아 문항을 만든다. 등급 i = i번째 후보."""
    lines = "\n".join(f"  {i+1} = {PHRASE[s]}" for i, s in enumerate(stages))
    return ("Answer with one line, \"A) 2\", nothing else -- one digit choosing the "
            "stage this moment belongs to:\n" + lines +
            "\nA) Which stage is this moment?\nAnswer:")
