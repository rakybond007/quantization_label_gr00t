---
name: nfs-load-discipline
description: "공유 NFS에 재귀 find/du/grep -r 금지, 26만 파일 타일 디렉터리는 매니페스트로 접근 — 서버 접속 장애 유발 가능"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-20T07:49:38.499Z
---

공유 마운트(`/rlwrld-unified-checkpoints`, `/rlwrld1~4`, 홈 NFS)에 **깊이 제한 없는 `find`, `du -sh`, `grep -r`를 걸지 않는다.** 대상 경로를 알고 있으면 `ls -d`/`stat`으로 직접 찍고, 탐색이 꼭 필요하면 `-maxdepth`를 좁히고 내 디렉터리 안으로 한정한다.

**Why:** 2026-08-20 서버 접속 장애 후 사용자가 "특정 디렉터리 파일 과다 또는 툴의 과도한 파일시스템 순회"를 원인 후보로 지목했다. 점검해보니 내가 해당하는 조작을 27건 했다 — 사용자 64명 전체를 훑는 `find /rlwrld-unified-checkpoints -maxdepth 2/3` 2회, 깊이 무제한 `find`로 `gate_module*.pt` 탐색(타임아웃 후 백그라운드 지속), `.claude/projects` 전체 `grep -r`, 그리고 26만 파일 디렉터리에 `du -sh`(5분 타임아웃). 장애 원인이라 단정할 근거는 없으나 정확히 그 유형이다.

**How to apply:** ① `vlm_gate/output/_gate_distill/luna_robocasa_full/tiles`는 **평평한 구조에 260,031개 PNG** — 여기에 `ls`/`du`/글롭 금지. `output/_gate_distill/tiles_manifest.txt`(파일명 목록)를 읽을 것. `cosmos_2call_fast.py`는 이 매니페스트를 쓰도록 고쳐 놨다(샤드마다 READDIR 26만이 터지던 것을 제거). ② 새로 파생 데이터를 만들 때 한 디렉터리에 수만 개를 평평하게 쌓지 말고 서브디렉터리로 쪼갠다. ③ 파일 개수를 셀 때도 `ls -U | wc -l`(정렬·stat 없음)을 쓴다. 관련: [[checkpoint-storage-archival]], [[ops-download-and-bg-task-lessons]]
