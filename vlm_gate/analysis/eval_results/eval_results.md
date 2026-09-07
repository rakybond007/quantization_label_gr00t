# 압축 배속별 평가 결과

`export_eval_results.py` 가 서버의 eval 로그에서 뽑았다. 다른 서버·세션에서 이 파일만 보고 손상표를 만들 수 있게 하려는 것이다.

손상은 압축 없는 기준선 대비 성공률 차이다. 음수가 압축이 깎은 만큼이다.


## robocasa

기준선 `baseline_full_v2_with_action_steps` — 성공률 **0.657**, 태스크 24개, 에피소드 1200개


### 균일 고정 배속

| 실행 | 성공률 | 에피소드 | 태스크 |
|---|---:|---:|---:|
| `_INVALID_offbyone_interp_cap2p0` | 0.595 | 1200 | 24 |
| `_INVALID_offbyone_interp_cap3p0` | 0.504 | 1200 | 24 |
| `baseline_compress_K2` | 0.598 | 1200 | 24 |
| `baseline_compress_K3` | 0.497 | 1200 | 24 |
| `baseline_compress_K4` | 0.404 | 1200 | 24 |
| `baseline_full_v2_with_action_steps` | 0.657 | 1200 | 24 |


### 태스크별 손상 — `baseline_full_v2_with_action_steps` → `baseline_compress_K2`

| 태스크 | 기준선 | 압축 | 차이 |
|---|---:|---:|---:|
| OpenDrawer | 0.72 | 0.44 | -0.280 |
| PnPCabToCounter | 0.62 | 0.40 | -0.220 |
| CoffeeServeMug | 0.78 | 0.60 | -0.180 |
| PnPCounterToSink | 0.70 | 0.52 | -0.180 |
| PnPStoveToCounter | 0.74 | 0.58 | -0.160 |
| CloseDoubleDoor | 0.86 | 0.70 | -0.160 |
| OpenSingleDoor | 0.80 | 0.66 | -0.140 |
| OpenDoubleDoor | 0.92 | 0.80 | -0.120 |
| TurnOnSinkFaucet | 0.68 | 0.58 | -0.100 |
| CoffeeSetupMug | 0.26 | 0.16 | -0.100 |
| PnPMicrowaveToCounter | 0.30 | 0.22 | -0.080 |
| TurnOffSinkFaucet | 0.90 | 0.82 | -0.080 |
| PnPSinkToCounter | 0.60 | 0.54 | -0.060 |
| TurnSinkSpout | 0.82 | 0.80 | -0.020 |
| PnPCounterToStove | 0.52 | 0.50 | -0.020 |
| CloseSingleDoor | 0.96 | 0.96 | +0.000 |
| TurnOffStove | 0.20 | 0.20 | +0.000 |
| CloseDrawer | 1.00 | 1.00 | +0.000 |
| PnPCounterToMicrowave | 0.24 | 0.26 | +0.020 |
| TurnOffMicrowave | 0.94 | 0.98 | +0.040 |
| PnPCounterToCab | 0.46 | 0.50 | +0.040 |
| TurnOnMicrowave | 0.56 | 0.64 | +0.080 |
| TurnOnStove | 0.36 | 0.50 | +0.140 |
| CoffeePressButton | 0.82 | 1.00 | +0.180 |


## libero

기준선 `baseline_raw` — 성공률 **0.932**, 태스크 40개, 에피소드 2000개


### 균일 고정 배속

| 실행 | 성공률 | 에피소드 | 태스크 |
|---|---:|---:|---:|
| `_INVALID_replanbug_baseline_K2` | 0.800 | 2000 | 40 |
| `baseline_K3` | 0.438 | 2000 | 40 |
| `baseline_K4` | 0.261 | 2000 | 40 |
| `baseline_K4_fixed` | 0.393 | 1400 | 28 |
| `baseline_raw` | 0.932 | 2000 | 40 |


### 태스크별 손상 — `baseline_raw` → `baseline_K3`

| 태스크 | 기준선 | 압축 | 차이 |
|---|---:|---:|---:|
| libero_spatial_9 | 0.98 | 0.00 | -0.980 |
| libero_goal_1 | 1.00 | 0.04 | -0.960 |
| libero_spatial_3 | 0.98 | 0.04 | -0.940 |
| libero_spatial_6 | 0.98 | 0.06 | -0.920 |
| libero_goal_4 | 0.94 | 0.02 | -0.920 |
| libero_spatial_4 | 0.92 | 0.04 | -0.880 |
| libero_goal_9 | 0.86 | 0.00 | -0.860 |
| libero_spatial_0 | 0.96 | 0.14 | -0.820 |
| libero_goal_2 | 1.00 | 0.18 | -0.820 |
| libero_10_1 | 0.98 | 0.18 | -0.800 |
| libero_spatial_1 | 0.98 | 0.18 | -0.800 |
| libero_goal_8 | 1.00 | 0.26 | -0.740 |
| libero_10_4 | 0.88 | 0.16 | -0.720 |
| libero_object_6 | 0.96 | 0.24 | -0.720 |
| libero_goal_3 | 0.78 | 0.08 | -0.700 |
| libero_10_6 | 0.90 | 0.26 | -0.640 |
| libero_10_0 | 0.82 | 0.20 | -0.620 |
| libero_goal_5 | 0.94 | 0.36 | -0.580 |
| libero_spatial_8 | 0.98 | 0.42 | -0.560 |
| libero_goal_6 | 0.92 | 0.42 | -0.500 |
| libero_spatial_7 | 0.94 | 0.48 | -0.460 |
| libero_10_2 | 0.84 | 0.48 | -0.360 |
| libero_object_3 | 1.00 | 0.64 | -0.360 |
| libero_object_5 | 1.00 | 0.66 | -0.340 |
| libero_10_7 | 0.94 | 0.60 | -0.340 |
| libero_object_2 | 1.00 | 0.68 | -0.320 |
| libero_object_4 | 0.98 | 0.68 | -0.300 |
| libero_object_9 | 0.98 | 0.70 | -0.280 |
| libero_spatial_5 | 0.96 | 0.72 | -0.240 |
| libero_spatial_2 | 1.00 | 0.78 | -0.220 |
| libero_object_1 | 0.98 | 0.76 | -0.220 |
| libero_object_8 | 1.00 | 0.80 | -0.200 |
| libero_goal_7 | 1.00 | 0.82 | -0.180 |
| libero_goal_0 | 0.94 | 0.78 | -0.160 |
| libero_10_5 | 0.90 | 0.78 | -0.120 |
| libero_object_7 | 0.98 | 0.88 | -0.100 |
| libero_10_9 | 0.80 | 0.70 | -0.100 |
| libero_10_3 | 0.94 | 0.88 | -0.060 |
| libero_object_0 | 0.96 | 0.90 | -0.060 |
| libero_10_8 | 0.40 | 0.52 | +0.120 |


## dexjoco

기준선 `baseline_60k_K1` — 성공률 **0.760**, 태스크 6개, 에피소드 300개


### 균일 고정 배속

| 실행 | 성공률 | 에피소드 | 태스크 |
|---|---:|---:|---:|
| `baseline_60k_K1` | 0.760 | 300 | 6 |
| `baseline_60k_K2` | 0.593 | 300 | 6 |
| `baseline_60k_K3` | 0.450 | 300 | 6 |


### 태스크별 손상 — `baseline_60k_K1` → `baseline_60k_K2`

| 태스크 | 기준선 | 압축 | 차이 |
|---|---:|---:|---:|
| fold_glasses | 0.78 | 0.48 | -0.300 |
| click_mouse | 0.88 | 0.62 | -0.260 |
| hammer_nail | 0.86 | 0.72 | -0.140 |
| pinch_tongs | 0.44 | 0.32 | -0.120 |
| pick_bucket | 0.84 | 0.74 | -0.100 |
| water_plant | 0.76 | 0.68 | -0.080 |
