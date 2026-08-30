import os
import collections
import dataclasses
import logging
import math
import pathlib
import imageio
import numpy as np
import tqdm
import tyro

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    # When using m8 (8-step compressed) for a chunk, replan more often to recover
    # from the 2x-speed effect of applying compressed actions one-per-env-step.
    replan_steps_m8: int = 3

    # Dynamic head selection (server must run with --head=main_and_m8)
    # Options:
    #   "main"        — always main 16-step (control, replan=replan_steps)
    #   "m8"          — always m8 8-step  (control, replan=replan_steps_m8)
    #   "agreement"   — use m8 if pair_sum(main first 16) ≈ m8 within tolerance,
    #                   else fall back to main
    #   "combined"    — agreement AND no gripper transition in next replan_steps
    decision: str = "agreement"
    # Threshold for "agreement" (per-step L2 norm of pair_sum(main) - m8 in
    # normalized continuous action space). Smaller = stricter = use m8 less.
    agreement_thresh: float = 0.05
    # Look-ahead window (in main-step units) for gripper-transition detection.
    grip_lookahead: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    task_idx: int = -1  # Task index to run

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, idx: {args.task_idx}")
    logging.info(f"Number of tasks in suite: {num_tasks_in_suite}")
    logging.info(f"Number of trials per task: {args.num_trials_per_task}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial" or args.task_suite_name == "libero_spatial_new" or args.task_suite_name == "libero_spatial_ood":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object" or args.task_suite_name == "libero_object_ood":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal" or args.task_suite_name == "libero_goal_new" or args.task_suite_name == "libero_goal_ood":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        if args.task_idx != -1 and task_id != args.task_idx:
            continue
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")
            task_segment = task_description.replace(" ", "_")
            if os.path.exists(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_failure.mp4"
            ):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                continue
            elif os.path.exists(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_success.mp4"
            ):
                logging.info(f"Video already exists, skipping episode {episode_idx}...")
                total_episodes += 1
                task_episodes += 1
                total_successes += 1
                task_successes += 1
                continue    
            if client is None:
                client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])


            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    # No need to resize for GR00T 
                    # img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    # )
                    # wrist_img = image_tools.convert_to_uint8(
                    #     image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    # )

                    #"""  # Save images for debugging
                    if t == args.num_steps_wait and episode_idx == 0:
                        task_segment = task_description.replace(" ", "_")
                        image_prefix = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}"
                        image_prefix.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(f"{image_prefix}_img.png", img)
                        imageio.imwrite(f"{image_prefix}_wrist.png", wrist_img)
                    #"""

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict

                        # element = {
                        #     "observation/image": img,
                        #     "observation/wrist_image": wrist_img,
                        #     "observation/state": np.concatenate(
                        #         (
                        #             obs["robot0_eef_pos"],
                        #             _quat2axisangle(obs["robot0_eef_quat"]),
                        #             obs["robot0_gripper_qpos"],
                        #         )
                        #     ),
                        #     **({"previous_actions": prev_actions_vec,
                        #         "previous_state": prev_state_vec,
                        #         "observation/previous_image": prev_img,
                        #         "observation/previous_wrist_image": prev_wrist_img} if args.use_previous_info else {}),
                        #     "prompt": str(task_description),
                        # }
                        element = {
                            "video.front_view": np.array([img]),
                            "video.left_wrist_view": np.array([wrist_img]),
                            "state.eef_pos_absolute": obs["robot0_eef_pos"], # GR00T requries [horizon, state_dim]
                            "state.eef_rot_absolute": _quat2axisangle(obs["robot0_eef_quat"]),
                            "state.gripper_close": obs["robot0_gripper_qpos"],
                            "annotation.human.action.task_description": [str(task_description)],
                        }
                        # Query model to get BOTH main (16-step) and m8 (8-step)
                        # outputs in a single forward pass (server head=main_and_m8).
                        out = client.infer(element)
                        main_chunk = np.concatenate([
                            out['action.eef_pos_delta'],
                            out['action.eef_rot_delta'],
                            out['action.gripper_close'].reshape(-1, 1)
                        ], axis=-1)                                          # (16, 7)
                        m8_chunk = None
                        if 'action.eef_pos_delta_m8' in out:
                            m8_chunk = np.concatenate([
                                out['action.eef_pos_delta_m8'],
                                out['action.eef_rot_delta_m8'],
                                out['action.gripper_close_m8'].reshape(-1, 1)
                            ], axis=-1)                                      # (8, 7)

                        # Decision: which head to use for this chunk?
                        use_m8 = False
                        if args.decision == "main" or m8_chunk is None:
                            use_m8 = False
                        elif args.decision == "m8":
                            use_m8 = True
                        elif args.decision in ("agreement", "combined"):
                            # pair-sum continuous of main first 16 → 8 steps to compare with m8 (continuous = first 6 dims)
                            paired_main_cont = (main_chunk[0::2, :6] + main_chunk[1::2, :6])
                            diff = np.linalg.norm(paired_main_cont - m8_chunk[:, :6], axis=-1)  # (8,)
                            agree = float(diff.mean()) < args.agreement_thresh
                            use_m8 = agree
                            if args.decision == "combined":
                                # Reject m8 if a gripper transition is upcoming in main's near horizon.
                                gw = args.grip_lookahead
                                grip = main_chunk[:gw, 6]
                                if len(grip) >= 2 and (np.abs(np.diff(grip)) > 0.5).any():
                                    use_m8 = False

                        if use_m8:
                            assert len(m8_chunk) >= args.replan_steps_m8, \
                                f"m8 chunk has {len(m8_chunk)} steps, want {args.replan_steps_m8}"
                            action_plan.extend(m8_chunk[: args.replan_steps_m8])
                            if not hasattr(args, "_n_m8"): args._n_m8 = 0
                            args._n_m8 += 1
                        else:
                            assert len(main_chunk) >= args.replan_steps, \
                                f"main chunk has {len(main_chunk)} steps, want {args.replan_steps}"
                            action_plan.extend(main_chunk[: args.replan_steps])
                        if not hasattr(args, "_n_total"): args._n_total = 0
                        args._n_total += 1

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    assert False
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=30,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    n_m8 = getattr(args, "_n_m8", 0)
    n_total = getattr(args, "_n_total", 1)
    m8_ratio = n_m8 / max(n_total, 1)
    logging.info(f"Decision mode: {args.decision} | m8 chunks: {n_m8}/{n_total} ({m8_ratio:.2%})")
    result_save_path = pathlib.Path(args.video_out_path) / f"{args.task_idx}_results.txt"
    with open(result_save_path, "w") as f:
        f.write(f"Total success rate: {float(total_successes) / float(total_episodes)}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Decision mode: {args.decision}\n")
        f.write(f"Agreement thresh: {args.agreement_thresh}\n")
        f.write(f"M8 chunks: {n_m8}/{n_total} ({m8_ratio:.4f})\n")
    logging.info(f"Results saved to {result_save_path}")
    # Hard-exit to skip slow libero env teardown (~9 min per variant). Results
    # are already written to disk. os._exit bypasses Python atexit handlers
    # (notably MuJoCo/EGL context teardown which is what hangs).
    import os as _os
    _os._exit(0)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
