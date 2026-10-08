import argparse
import logging
import os
import time
import cv2
import numpy as np
import torch
import tqdm

from typing import Any, Deque, MutableMapping, OrderedDict
from collections import deque
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from unitree_deploy.pep_bridge import TwinPEP
from unitree_deploy.real_unitree_env import make_real_env
from unitree_deploy.utils.eval_utils import (
    ACTTemporalEnsembler,
    LongConnectionClient,
    populate_queues,
)

# -----------------------------------------------------------------------------
# Network & environment defaults
# -----------------------------------------------------------------------------
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
# eval_utils switches on DEBUG logging for every library; the HTTP library would then print a line for each
# request to the model and to the twin's PEP (several per second). Keep only its warnings and errors.
logging.getLogger("urllib3").setLevel(logging.WARNING)
HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

# fmt: off
INIT_POSE = {
    # Starting state of the "pack camera" training episode
    # (examples/world_model_interaction_prompts/transitions/unitree_g1_pack_camera/0.h5, observation.state[0]),
    # so the robot starts in a pose the model has seen. Unitree's original value:
    # [0.10559805, 0.02726714, -0.01210221, -0.33341318, -0.22513399, -0.02627627, -0.15437093,  0.1273793 , -0.1674708 , -0.11544029, -0.40095493,  0.44332668,  0.11566751,  0.3936641, 5.4, 5.4]
    'g1_dex1': np.array([-0.50163078, 0.32051945, 0.18178487, 0.6838522, -0.085745335, -0.51033354, -0.25062084, -0.40424705, -0.26256371, -0.20069599, 0.21123028, 0.1912303, -0.20726478, 0.28874797, 5.4340272, 5.378758], dtype=np.float32),
    'z1_dual_dex1_realsense': np.array([-1.0262332,  1.4281361, -1.2149128,  0.6473399, -0.12425245, 0.44945636,  0.89584476,  1.2593982, -1.0737865,  0.6672816, 0.39730102, -0.47400007, 0.9894176, 0.9817477 ], dtype=np.float32),
    'z1_realsense': np.array([-0.06940782, 1.4751548, -0.7554075, 1.0501366, 0.02931615, -0.02810347, -0.99238837], dtype=np.float32),
}
ZERO_ACTION = {
    'g1_dex1': torch.zeros(16, dtype=torch.float32),
    'z1_dual_dex1_realsense': torch.zeros(14, dtype=torch.float32),
    'z1_realsense': torch.zeros(7, dtype=torch.float32),
}
CAM_KEY = {
    'g1_dex1': 'cam_right_high',
    'z1_dual_dex1_realsense': 'cam_high',
    'z1_realsense': 'cam_high',
}
# fmt: on


def prepare_observation(args: argparse.Namespace, obs: Any) -> OrderedDict:
    """
    Convert a raw env observation into the model's expected input dict.
    """
    # UnitreeEnv already converts the camera images from BGR to RGB; converting again here
    # would swap red and blue back before the image reaches the model.
    rgb_image = np.ascontiguousarray(obs.observation["images"][CAM_KEY[args.robot_type]])
    observation = {
        "observation.images.top":
        torch.from_numpy(rgb_image).permute(2, 0, 1),
        "observation.state":
        torch.from_numpy(obs.observation["qpos"]),
        "action": ZERO_ACTION[args.robot_type],
    }
    return OrderedDict(observation)


def run_policy(
    args: argparse.Namespace,
    env: Any,
    client: LongConnectionClient,
    temporal_ensembler: ACTTemporalEnsembler,
    cond_obs_queues: MutableMapping[str, Deque[torch.Tensor]],
    output_dir: Path,
    pep: TwinPEP | None = None,
) -> None:
    """
    Single rollout loop:
        1) warm start the robot,
        2) stream observations,
        3) fetch actions from the policy server,
        4) execute with temporal ensembling for smoother control.
    With `pep`, the start pose and every chunk are first validated by the twin's PEP.
    """

    if pep is not None:
        # The robot holds still until the PEP allows the move (e.g. after a person arms the twin).
        while True:
            state = env.get_observation(0).observation["qpos"]
            decision = pep.check_trajectory(INIT_POSE[args.robot_type][None], dt=1.0, start=state)
            print(f">>> PEP start pose: {decision.decision} {decision.rule_id} {decision.message}", flush=True)
            if decision.allowed:
                break
            time.sleep(3.0)
    _ = env.step(INIT_POSE[args.robot_type])
    time.sleep(2.0)
    t = 0

    while True:
        # Gapture observation
        obs = env.get_observation(t)
        # Format observation
        obs = prepare_observation(args, obs)
        cond_obs_queues = populate_queues(cond_obs_queues, obs)
        # Call server to get actions
        pred_actions = client.predict_action(args.language_instruction,
                                             cond_obs_queues).unsqueeze(0)
        # Keep only the next horizon of actions and apply temporal ensemble smoothing
        actions = temporal_ensembler.update(
            pred_actions[:, :args.action_horizon])[0]

        if pep is not None:
            decision = pep.check_trajectory(actions[:args.exe_steps].cpu().numpy(),
                                            dt=1 / args.control_freq,
                                            start=obs["observation.state"].numpy())
            print(f">>> PEP chunk: {decision.decision} {decision.rule_id} {decision.message}", flush=True)
            if not decision.allowed:
                # Hold still and ask the model again from a fresh observation.
                time.sleep(1.0)
                continue

        # Execute the actions
        for n in range(args.exe_steps):
            if pep is not None and not pep.may_continue():
                print(">>> PEP: twin no longer armed (E-stop, disarm or unreachable): chunk aborted", flush=True)
                break
            action = actions[n].cpu().numpy()
            print(f">>> Exec => step {n} action: {action}", flush=True)
            print("---------------------------------------------")

            # Maintain real-time loop at `control_freq` Hz
            t1 = time.time()
            obs = env.step(action)
            time.sleep(max(0, 1 / args.control_freq - time.time() + t1))
            t += 1

            # Prime the queue for the next action step (except after the last one in this chunk)
            if n < args.exe_steps - 1:
                obs = prepare_observation(args, obs)
                cond_obs_queues = populate_queues(cond_obs_queues, obs)


def run_eval(args: argparse.Namespace) -> None:
    client = LongConnectionClient(BASE_URL)

    pep = None
    if args.pep_url:
        if args.robot_type != "g1_dex1":
            raise ValueError("--pep_url supports only --robot_type g1_dex1")
        pep = TwinPEP(args.pep_url)
        twin = pep.state()  # fails here if the twin is not reachable
        print(f">>> Twin PEP at {args.pep_url}: robot {twin['id']} is {twin['mode']}", flush=True)

    # Initialize ACT temporal moving-averge smoother
    temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff=0.01,
                                              chunk_size=args.action_horizon,
                                              exe_steps=args.exe_steps)
    temporal_ensembler.reset()

    # Initialize observation and action horizon queue
    cond_obs_queues = {
        "observation.images.top": deque(maxlen=args.observation_horizon),
        "observation.state": deque(maxlen=args.observation_horizon),
        "action": deque(
            maxlen=16),  # NOTE: HAND CODE AS THE MODEL PREDCIT FUTURE 16 STEPS
    }

    if args.network_interface:
        # The first initialisation of the DDS factory wins (the arm controller's later ChannelFactoryInitialize(0)
        # then keeps it), so this pins the interface the client uses to talk to the robot or the simulator.
        ChannelFactoryInitialize(0, args.network_interface)
    env = make_real_env(
        robot_type=args.robot_type,
        dt=1 / args.control_freq,
    )
    if args.robot_type == "g1_dex1":
        # On connect (and on exit) the arm controller drives the arms to its init_pose, and until the first action
        # it holds its q_target; the grippers hold theirs. All are zeros by default, and with the table of the
        # "pack camera" task the zero pose goes through the table and the objects (the first action only comes once
        # the PEP allows it). So the arms and grippers start, wait and finish at the task's start pose instead.
        start = INIT_POSE[args.robot_type]
        for arm in env.robot.arm.values():
            arm.init_pose = start[:14].copy()
            arm.q_target = start[:14].copy()
        for gripper, opening in zip(env.robot.endeffector.values(), start[14:]):  # left, right
            gripper.q_target = np.array([opening])
    env.connect()

    try:
        for episode_idx in tqdm.tqdm(range(0, args.num_rollouts_planned)):
            output_dir = Path(args.output_dir) / f"episode_{episode_idx:03d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            run_policy(args, env, client, temporal_ensembler, cond_obs_queues,
                       output_dir, pep)
    finally:
        env.close()
    env.close()


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_type",
                        type=str,
                        default="g1_dex1",
                        help="The type of the robot embodiment.")
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=16,
        help="Number of future actions, predicted by the policy, to keep",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help=
        "Number of future actions to execute, which must be less than the above action horizon.",
    )
    parser.add_argument(
        "--observation_horizon",
        type=int,
        default=2,
        help="Number of most recent frames/states to consider.",
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default="Pack black camera into box",
        help="The language instruction provided to the policy server.",
    )
    parser.add_argument("--num_rollouts_planned",
                        type=int,
                        default=10,
                        help="The number of rollouts to run.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./results",
                        help="The directory for saving results.")
    parser.add_argument("--control_freq",
                        type=float,
                        default=30,
                        help="The Low-level control frequency in Hz.")
    parser.add_argument("--network_interface",
                        type=str,
                        default=None,
                        help="Network interface for DDS (e.g. lo, eth0). Default: chosen automatically. Give the "
                        "simulator the same --network_interface: if the two end up on different interfaces, the "
                        "client waits forever at 'Waiting to subscribe dds...'. Both on lo when they run on the "
                        "same machine.")
    parser.add_argument("--pep_url",
                        type=str,
                        default=None,
                        help="Base URL of the DGS G1-D Cyber Twin (e.g. http://127.0.0.1:3000). "
                        "If set, every chunk is validated by the twin's PEP before execution.")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run_eval(args)
