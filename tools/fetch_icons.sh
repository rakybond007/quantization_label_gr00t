#!/bin/bash
# 논문 그림에 쓰는 일러스트를 받는다. Microsoft Fluent Emoji (MIT).
# 투명 배경 256px PNG 라 그대로 쓰면 되고, 이모지 **글자**와 달리 어느 기계에서
# 열어도 같게 보인다. 저장소에 커밋해 두므로 보통은 다시 받을 일이 없다.
set -eu
D="$(cd "$(dirname "$0")/.." && pwd)/vlm_gate/assets/icons"
B=https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets
mkdir -p "$D"
get() { curl -s -f -o "$D/$1.png" "$B/$2" && echo "ok   $1" || echo "MISS $1"; }
get robot     "Robot/3D/robot_3d.png"
get videogame "Video%20game/3D/video_game_3d.png"
get chart     "Chart%20increasing/3D/chart_increasing_3d.png"
get eyes      "Eyes/3D/eyes_3d.png"
get memo      "Memo/3D/memo_3d.png"
get balance   "Balance%20scale/3D/balance_scale_3d.png"
get stopwatch "Stopwatch/3D/stopwatch_3d.png"
get basket    "Basket/3D/basket_3d.png"
get plate     "Fork%20and%20knife%20with%20plate/3D/fork_and_knife_with_plate_3d.png"
get bowl      "Bowl%20with%20spoon/3D/bowl_with_spoon_3d.png"
get butter    "Butter/3D/butter_3d.png"
get canned    "Canned%20food/3D/canned_food_3d.png"
get check     "Check%20mark%20button/3D/check_mark_button_3d.png"
get cross     "Cross%20mark/3D/cross_mark_3d.png"
get mag       "Magnifying%20glass%20tilted%20left/3D/magnifying_glass_tilted_left_3d.png"
