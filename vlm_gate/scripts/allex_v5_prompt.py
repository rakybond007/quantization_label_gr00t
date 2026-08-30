"""v5 — 기준 문서(allex_criteria_v1.md)에서 생성한 티처 프롬프트.

프레이밍: 초기 프롬프트는 프론티어 모델의 장면 분석 + 운영자 힌트로 작성된
'기준 문서'에서 파생된다. 티처 품질은 물리 프록시 대비 AUC가 아니라
**기준 문서로 작성된 라벨과의 일치도**로 측정한다. 기준의 옳고 그름은
별개 문제이며 폐루프가 판정한다.

질문은 계산 불가능한 축만 남긴다 — 물체의 성질, 조작의 국면, 그리고 총평.
계산 가능한 것(손목 간격·추세·속도·손가락 변화·병합 실현가능성)은 사실로 진술한다.
"""
SYS=("You are judging whether a short window of a robot's planned motion can be executed at half "
 "the control rate. The robot is a bimanual humanoid handling parcels on a conveyor. It does not "
 "pinch with its fingers: it presses both palms against opposite faces of a package and carries it "
 "squeezed between them, so finger motion is not a grasp signal. The arms take absolute joint "
 "targets, so halving the rate means executing every second target and skipping the one between - "
 "a path deviation of a few millimetres.\n"
 "Compression is the default and is desirable. It is unsafe only where a few millimetres of path "
 "error changes the outcome: while the palms are converging to take a package's weight, while they "
 "are separating from one being set down, while a package's final position or facing is being "
 "established, and whenever the item is a bag or otherwise deformable.\n"
 "Answer the checks exactly in the format requested and nothing else.")

ASK=("The measurements above are exact. Judge only what they cannot tell you. Answer each check on "
 "its own line as \"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Is the item a bag or otherwise soft and liable to shift, rather than a rigid box?\n"
 "B) Is the robot mid-transition - palms closing to take the weight, or separating to give it up -\n"
 "   rather than either holding it securely or not yet in contact?\n"
 "C) Is the package's final resting position or facing being established right now, rather than the\n"
 "   package simply being moved from one place to another?\n"
 "D) Would running this half-second at double speed end with the same result?\nAnswer:")
