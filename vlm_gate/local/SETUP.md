# 맥북 최초 세팅 (1회)

## 사용자가 직접 해야 하는 것

### 1. ssh 설정 — 연결 재사용이 핵심

에이전트는 명령마다 ssh를 부른다. 매번 새로 붙으면 느리고 서버에도 부담이다.
`~/.ssh/config`에 아래를 넣으면 연결 하나를 10분간 재사용한다.

세 대의 로그인 노드가 같은 `/sjw_alinlab` 를 공유한다. 어느 쪽으로 붙어도
파일과 slurm 큐는 같다 — 고르는 기준은 그 노드의 부하뿐이다.

```
Host rlwrld2_0 rlwrld2_1 rlwrld2_2
    User            hojin2
    IdentityFile    ~/.ssh/id_rlwrld2
    ControlMaster   auto
    ControlPath     ~/.ssh/cm-%r@%h:%p
    ControlPersist  10m
    ServerAliveInterval 30

Host rlwrld2_0
    HostName 210.109.80.159
Host rlwrld2_1
    HostName 210.109.80.118
Host rlwrld2_2
    HostName 210.109.80.237
```

| 별칭 | 노드 | 2026-08-31 관측 |
|---|---|---|
| `rlwrld2_2` | login-node2 | load 3, 11명, 여유 113G — **기본값** |
| `rlwrld2_1` | login-node3 | load 14, 56명, 여유 38G |
| `rlwrld2_0` | login-node1 | 동시 로그인 한도 초과로 거부 중 |

`rlwrld2_0` 은 `There were too many logins for 'hojin2'` 로 막힌다. 세션이 남아
있어서지 서버가 죽은 게 아니다. 붙는 노드를 바꿀 일이 생기면 `DEV_HOST` 만
바꾸면 된다 — tmux 세션은 노드마다 따로 살아있으므로 `dev up` 을 다시 한다.

키 로그인 확인: `ssh rlwrld2_2 hostname` → `login-node2` 가 나와야 한다.

### 2. 저장소 클론

```bash
git clone git@github.com:rakybond007/quantization_label_gr00t.git
cd quantization_label_gr00t
export DEV_HOST=rlwrld2_2        # ~/.zshrc 에도 넣어두기
```

이 저장소가 지금의 원본이다. `GR00T-action-quantization` 의
`action-quantization-gate-v2` 브랜치가 예전 자리였는데, 8/30 에 작업 트리
전체를 옮겨 담으면서 대체됐다 — 두 이력은 이어져 있지 않으니 옛 브랜치를
참고하지 말 것. (`GR00T-action-quantization` 의 `action-quantization-impl`
브랜치는 아직 살아있는 별개의 줄기다: `~/multigpu_workspace` 의 ATQ
MoE·라우터 작업.)

이 디렉터리에서 Claude Code를 연다. 루트의 `CLAUDE.md`가 자동으로 읽힌다.

### 3. API 키를 맥북으로 옮긴다

인프라팀 정책상 **클러스터에 비밀정보를 두지 않는다.** 현재 서버 홈에
`gemini_key`, `openai_key`가 평문으로 있다. 맥북으로 내리고 서버에서 지운다.

```bash
scp rlwrld2_2:~/quantization_agent_workspace/gemini_key ~/.config/quant/gemini_key
scp rlwrld2_2:~/quantization_agent_workspace/openai_key ~/.config/quant/openai_key
chmod 600 ~/.config/quant/*
ssh rlwrld2_2 'shred -u ~/quantization_agent_workspace/gemini_key ~/quantization_agent_workspace/openai_key'
```

이후 API를 쓰는 잡은 키를 그때만 주입한다:
`dev run "GEMINI_API_KEY=$(cat ~/.config/quant/gemini_key) python ..."`

### 4. 첫 동작 확인

```bash
tools/dev up
tools/dev run 'hostname; ls ~/quantization_agent_workspace | head'
```

서버 이름과 디렉터리 목록이 나오면 끝이다.

## 인프라팀에 물어봐야 할 것 하나

정책의 "AI 툴 트래픽 차단"이 **Gemini/Claude API 호출까지 막는지** 확인이 필요하다.
막힌다면 서버에서 API 라벨링이 불가능해진다.
다만 주력 라벨러는 가중치가 서버 GPU에 로컬로 있는 **cosmos**라 파이프라인 본체는
영향이 없다. API는 검증용 보조 경로였다.

## 맥북에 설치할 것

없다. `git`, `ssh`, `rsync`는 macOS에 기본으로 있다. 파이썬은 쓰지 않는다.
