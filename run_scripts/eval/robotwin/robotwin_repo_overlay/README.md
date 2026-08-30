# RoboTwin repo overlay (GR00T eval)

These files are our modifications to the third-party RoboTwin 2.0 repo
(external_dependencies/RoboTwin, untracked because of ~22GB assets).

To reproduce eval on another server:
1. clone RoboTwin: `git clone https://github.com/RoboTwin-Platform/RoboTwin external_dependencies/RoboTwin`
2. set up `robotwin` conda env (sapien/mplib/pytorch3d/curobo). curobo needs **warp-lang==1.12.1**
   (1.13 drops wp.torch; 1.4.2 → CUDA illegal instruction). Build curobo with
   TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" + ninja on a GPU node.
3. download assets: `cd external_dependencies/RoboTwin && bash script/_download_assets.sh`
   (incl. background_texture.zip for randomized mode)
4. overlay these files:
   - script/eval_policy.py        → EP_RESULT marker, ROBOTWIN_EVAL_SAVE_DIR override,
                                     summary.txt + per_ep.csv + episode{N}_{success|fail}.mp4
   - policy/gr00t_zmq/            → our zmq policy adapter (inline RobotInferenceClient)
