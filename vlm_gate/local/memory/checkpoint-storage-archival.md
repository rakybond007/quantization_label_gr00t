---
name: checkpoint-storage-archival
description: /rlwrld-unified-checkpoints는 7일 후 오브젝트 스토리지 이관 + 90일 후 삭제 — 쓰는 자산은 홈의 assets/로 옮길 것
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-20T00:37:47.938Z
---

`/rlwrld-unified-checkpoints`는 일정 기간 미접근 데이터를 각 클러스터 오브젝트 스토리지로 자동 이관한다(기준 7일, 폴더 단위 atime 우선·없으면 mtime). 이관본은 옮긴 시각 기준 날짜 폴더로 생성되며, **오브젝트 스토리지에서도 90일 후 삭제**된다.

**Why:** robocasa 전량 프레임 캐시(262k 프레임 uint8 memmap, 39GB)가 이렇게 사라졌다. 8/17 04:51 학습이 정상적으로 읽었는데(로그 `vlm_gate/out/102915-*.out`, 175,782프레임 매칭) 8/18 13:19에 `checkpoints/` 에서 제거됐다. atime 우선 규칙대로면 3일 경과라 대상이 아니어야 하는데 사라졌다 — 마운트가 `relatime`이라 NFS 읽기가 atime에 반영되지 않고 mtime(7/30 생성 이후 쓰기 없음) 폴백이 적용된 것으로 추정하나 확증은 없다. 2026-08-20 기준 `/rlwrld-unified-checkpoints` 어디에도 날짜 폴더가 없어 이관본 경로를 못 찾았고, 관리자 확인이 필요하다.

**How to apply:** ① 계속 쓰는 자산은 `~/quantization_agent_workspace/assets/` (홈, `/sjw_alinlab`)에 둔다 — 프레임 캐시, A' 모듈, 라벨 parquet, task embedding, gateC 가중치를 2026-08-20에 이관했다. ② 새 산출물의 출력 경로를 unified-checkpoints로 잡을 때는 그게 재생성 가능한지 먼저 따진다. ③ **읽기만으로는 보호되지 않는다**고 가정할 것. ④ 사라진 게 있으면 원인을 추측하기 전에 상위 디렉터리 mtime으로 제거 시각부터 특정한다. 관련: [[libero-install-repoint]], [[home-migration]], [[ops-download-and-bg-task-lessons]]
