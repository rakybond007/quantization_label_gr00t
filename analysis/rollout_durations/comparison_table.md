# 성공 에피소드 평균 동영상 길이 (frames @ 20fps → seconds)

환경 fps=20, 동영상 길이 = 성공한 에피소드의 ffprobe nb_read_packets 평균.

| Task | baseline (16-step main) | baseline + client merge 16→8 | mh_m8 + ens→8 | mh_m8+econsist + ens→8 |
|------|------|------|------|------|
| CloseDoubleDoor | 21.5s (429f, n=42) | 17.5s (350f, n=39) | 15.3s (306f, n=10) | 14.6s (292f, n=36) |
| CloseDrawer | 10.2s (204f, n=42) | 6.0s (121f, n=50) | 6.5s (129f, n=50) | 5.1s (103f, n=16) |
| CloseSingleDoor | 13.0s (260f, n=47) | 8.8s (175f, n=46) | 8.9s (178f, n=49) | 7.6s (153f, n=47) |
| CoffeePressButton | 6.0s (121f, n=31) | 4.4s (89f, n=45) | 3.4s (68f, n=43) | 3.1s (61f, n=11) |
| CoffeeServeMug | 15.2s (304f, n=38) | 9.4s (188f, n=39) | 9.7s (194f, n=38) | 9.2s (185f, n=14) |
| CoffeeSetupMug | 13.2s (264f, n=7) | 8.0s (160f, n=12) | 7.7s (155f, n=10) | 5.9s (117f, n=4) |
| OpenDoubleDoor | 36.8s (736f, n=39) | 25.1s (502f, n=34) | 22.7s (455f, n=35) | 22.8s (456f, n=12) |
| OpenDrawer | 11.3s (226f, n=37) | 7.9s (157f, n=23) | 6.8s (137f, n=29) | 7.1s (143f, n=7) |
| OpenSingleDoor | 13.8s (277f, n=39) | 13.8s (277f, n=35) | 17.5s (350f, n=14) | 12.6s (252f, n=37) |
| PnPCabToCounter | 22.1s (442f, n=7) | 12.0s (240f, n=26) | 14.1s (282f, n=9) | 11.3s (225f, n=6) |
| PnPCounterToCab | 14.9s (299f, n=25) | 10.2s (204f, n=25) | 9.2s (184f, n=15) | 11.1s (223f, n=23) |
| PnPCounterToMicrowave | 19.3s (386f, n=4) | 14.3s (287f, n=11) | 13.6s (272f, n=6) | 25.4s (507f, n=3) |
| PnPCounterToSink | 25.0s (500f, n=37) | 15.0s (300f, n=31) | 18.9s (378f, n=1) | 14.2s (285f, n=9) |
| PnPCounterToStove | 18.5s (370f, n=19) | 12.1s (242f, n=26) | 11.0s (220f, n=22) | 20.4s (407f, n=5) |
| PnPMicrowaveToCounter | 15.3s (306f, n=12) | 9.3s (186f, n=7) | 10.3s (206f, n=11) | 8.0s (159f, n=4) |
| PnPSinkToCounter | 16.2s (324f, n=35) | 11.3s (226f, n=19) | 10.2s (203f, n=20) | 11.8s (236f, n=8) |
| PnPStoveToCounter | 15.8s (317f, n=39) | 9.3s (185f, n=35) | 8.4s (169f, n=8) | 10.0s (200f, n=36) |
| TurnOffMicrowave | 12.5s (250f, n=30) | 8.0s (161f, n=48) | 8.6s (171f, n=47) | 7.1s (141f, n=14) |
| TurnOffSinkFaucet | 10.9s (217f, n=42) | 11.6s (232f, n=38) | 8.6s (173f, n=37) | 8.2s (163f, n=43) |
| TurnOffStove | 13.4s (268f, n=5) | 14.5s (289f, n=11) | 35.4s (707f, n=3) | 5.2s (104f, n=4) |
| TurnOnMicrowave | 12.4s (248f, n=25) | 7.2s (144f, n=28) | 8.4s (168f, n=27) | 4.8s (96f, n=6) |
| TurnOnSinkFaucet | 12.4s (249f, n=13) | 8.7s (174f, n=29) | 8.0s (160f, n=35) | 7.6s (152f, n=6) |
| TurnOnStove | 16.5s (331f, n=19) | 11.8s (236f, n=18) | 10.5s (211f, n=14) | 11.3s (226f, n=9) |
| TurnSinkSpout | 7.5s (151f, n=38) | 5.1s (101f, n=38) | 4.6s (93f, n=38) | 4.7s (95f, n=17) |

## 전체 평균 (모든 task의 성공 episode pool 평균)

| 모델 | 평균 길이 | 총 성공 episode |
|------|---------:|---------------:|
| baseline (16-step main) | 15.50s (310.0f) | 672 |
| baseline + client merge 16→8 | 10.55s (210.9f) | 713 |
| mh_m8 + ens→8 | 9.45s (188.9f) | 571 |
| mh_m8+econsist + ens→8 | 10.05s (200.9f) | 377 |
