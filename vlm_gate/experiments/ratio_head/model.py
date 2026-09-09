"""배속 예측 머리. 신뢰도(이진) 대신 **배속 눈금 네 칸**을 고른다.

`proprio_history/model.py` 에서 갈라져 나왔고 세 군데가 다르다.

  1  출력이 1칸 로짓이 아니라 **4칸 로짓**이다 (1.0 · 1.5 · 2.0 · 2.5).
     회귀가 아니라 분류로 간다 -- 눈금이 이산이고 디코더가 그 네 칸만 낼 수
     있으므로, 1.7 같은 중간값을 내면 결국 반올림해서 쓰게 된다.

  2  액션이 **과거에 실행한 것**이 아니라 **이제 실행할 청크**다. 라벨이 그
     청크에서 나왔으므로 같은 정보를 준다. 배치 시점에 정책이 그 청크를 이미
     갖고 있으니 추론에서도 쓸 수 있다.

  3  접촉 가드를 **따로 낸다**. 라벨의 16.6% 가 가드로 1.0 에 박힌 것이고, 그
     자리는 신뢰도가 아니라 접촉이 정한다. 한 머리로 뭉뚱그리면 "왜 1.0 인가" 가
     안 갈린다 -- 가드라서인지 신뢰도가 낮아서인지.

`gate` 로짓은 그 시점이 가드인지, `ratio` 로짓은 가드가 아닐 때 어느 칸인지다.
추론은 `gate` 가 서면 1.0, 아니면 `ratio` 의 argmax 다.
"""
import numpy as np
import torch
import torch.nn as nn

GRID = (1.0, 1.5, 2.0, 2.5)


def pack_planned(actions, frame, horizon=16):
    """시점 f 에서 **앞으로 실행할** 청크 a[f : f+horizon].

    `proprio_history.pack_history` 의 거울상이다. 그쪽은 a[:f] 만 합법이었는데,
    여기서는 앞을 본다 -- **라벨이 바로 이 청크에서 나왔기 때문**이다.
    `descriptors(a, f)` 가 이 창을 읽어 사실 문장을 만들었고 VLM 이 그걸 보고
    등급을 매겼다. 같은 창을 주지 않으면 모델이 라벨의 근거를 못 본다.

    추론에서도 합법이다. 정책이 청크를 낸 직후에 이 머리를 부르므로 그 청크가
    이미 손에 있다. 다만 **학습은 시연 액션(GT)으로 하고 추론은 정책이 낸
    청크를 보게 되므로** 그 사이에 분포 차이가 있다 -- 첫 학습에서는 감수하고,
    나중에 정책 청크로 한 번 재는 것이 맞다.

    에피소드 끝에서 짧아지는 꼬리는 오른쪽을 0 으로 채우고 마스크로 가린다.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or not 0 <= frame <= len(actions) or horizon < 1:
        raise ValueError("액션 배열·프레임·horizon 이 올바르지 않다")
    out = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.float32)
    count = max(0, min(len(actions) - frame, horizon))
    if count:
        out[:count] = actions[frame:frame + count]
        mask[:count] = 1
    return out, mask


class RatioHead(nn.Module):
    def __init__(self, state_dim=53, action_dim=12, horizon=16, text_dim=384,
                 width=32, n_ratio=len(GRID)):
        super().__init__()
        self.config = dict(state_dim=state_dim, action_dim=action_dim,
                           horizon=horizon, text_dim=text_dim, width=width,
                           n_ratio=n_ratio, grid=list(GRID))

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        # 뷰 셋을 채널로 쌓아 넣는다(3뷰 x RGB = 9).
        self.net = nn.Sequential(blk(9, width), blk(width, width * 2),
                                 blk(width * 2, width * 4), blk(width * 4, width * 4),
                                 nn.AdaptiveAvgPool2d(1))
        # 상태 + 앞으로 실행할 청크. 마스크는 짧은 꼬리를 가린다.
        self.motion = nn.Sequential(
            nn.Linear(state_dim + horizon * (action_dim + 1), 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        trunk = width * 4 + text_dim + 64
        self.shared = nn.Sequential(nn.Linear(trunk, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU())
        self.ratio_head = nn.Linear(64, n_ratio)
        self.gate_head = nn.Linear(64, 1)

        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))

    def set_normalization(self, states, actions, mask):
        """학습 행에서만 맞춘다. 패딩은 통계에서 뺀다."""
        for prefix, values in (("state", states), ("action", actions[mask.astype(bool)])):
            if not len(values) or not np.isfinite(values).all():
                raise ValueError("정규화에는 유한한 학습 데이터가 필요하다")
            m = values.mean(axis=0, dtype=np.float64).astype(np.float32)
            s = values.std(axis=0, dtype=np.float64).astype(np.float32)
            s = np.where(s < 1e-5, 1.0, s)
            getattr(self, prefix + "_mean").copy_(torch.from_numpy(m))
            getattr(self, prefix + "_std").copy_(torch.from_numpy(s))

    def forward(self, images, text, state, planned_actions, action_mask):
        c = self.config
        b = images.shape[0]
        if state.shape != (b, c["state_dim"]) or text.shape != (b, c["text_dim"]):
            raise ValueError("state/text 모양이 안 맞는다")
        if planned_actions.shape != (b, c["horizon"], c["action_dim"]):
            raise ValueError("planned_actions 모양이 안 맞는다")
        if action_mask.shape != (b, c["horizon"]):
            raise ValueError("action_mask 모양이 안 맞는다")
        s = (state - self.state_mean) / self.state_std
        a = ((planned_actions - self.action_mean) / self.action_std) \
            * action_mask.unsqueeze(-1)
        motion = self.motion(torch.cat((s, a.flatten(1), action_mask), dim=1))
        visual = self.net(images).flatten(1)
        h = self.shared(torch.cat((visual, text, motion), dim=1))
        return self.ratio_head(h), self.gate_head(h).squeeze(-1)

    @torch.no_grad()
    def predict_ratio(self, *args, gate_thresh=0.5):
        """가드가 서면 1.0, 아니면 눈금 argmax."""
        rl, gl = self.forward(*args)
        g = torch.sigmoid(gl) > gate_thresh
        idx = rl.argmax(dim=1)
        grid = torch.tensor(self.config["grid"], device=rl.device, dtype=torch.float32)
        return torch.where(g, grid[0], grid[idx])
