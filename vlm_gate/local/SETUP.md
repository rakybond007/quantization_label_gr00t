# 맥북 최초 세팅 (1회)

## 사용자가 직접 해야 하는 것

### 1. ssh 설정 — 연결 재사용이 핵심

에이전트는 명령마다 ssh를 부른다. 매번 새로 붙으면 느리고 서버에도 부담이다.
`~/.ssh/config`에 아래를 넣으면 연결 하나를 10분간 재사용한다.

```
Host login-node1
    HostName        <서버 주소>
    User            hojin2
    ControlMaster   auto
    ControlPath     ~/.ssh/cm-%r@%h:%p
    ControlPersist  10m
    ServerAliveInterval 30
```

키 로그인이 되는지 먼저 확인: `ssh login-node1 hostname` → `login-node1` 이 나와야 한다.
비밀번호를 묻는다면 `ssh-copy-id login-node1` 로 키를 올린다.

### 2. 저장소 클론

```bash
git clone -b action-quantization-gate-v2 \
  git@github.com:rakybond007/GR00T-action-quantization.git ~/quant
cd ~/quant
export DEV_HOST=login-node1      # ~/.zshrc 에도 넣어두기
```

이 디렉터리에서 Claude Code를 연다. 루트의 `CLAUDE.md`가 자동으로 읽힌다.

### 3. API 키를 맥북으로 옮긴다

인프라팀 정책상 **클러스터에 비밀정보를 두지 않는다.** 현재 서버 홈에
`gemini_key`, `openai_key`가 평문으로 있다. 맥북으로 내리고 서버에서 지운다.

```bash
scp login-node1:~/quantization_agent_workspace/gemini_key ~/.config/quant/gemini_key
scp login-node1:~/quantization_agent_workspace/openai_key ~/.config/quant/openai_key
chmod 600 ~/.config/quant/*
ssh login-node1 'shred -u ~/quantization_agent_workspace/gemini_key ~/quantization_agent_workspace/openai_key'
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
