"""
Generate a simulator-rendered training set for fine-tuning UnifoLM-WMA-0 on "pack black camera into box": the real
model performs poorly on sim_g1_robot.py's camera images because they differ from the real dataset's, and that
gap is closed by fine-tuning on (sim video, state, action) sequences.

Each episode replays the full packing (replay_data/g1_pack_camera_ep0_cover.npz: the right hand puts the camera
into the box, the left hand puts the lid over it) in sim_g1_robot.py's scene, in this process, with the arm and
gripper control robot_client.py applies on the robot:
    - arms: the driver's PD law with the client's gains (kp 80 / kd 3, wrists 40 / 1.5) and its feed-forward
      gravity torques at the commanded pose (the simulated wrist and gripper weigh what the client's model
      assumes, so these are the same torques);
    - grippers: the Dex1 controller's command limit (never more than 0.18 past the measured opening), and the
      simulator's sticky grasp for the lid;
    - the recorded actions at their real rate, 30 per second, interpolated in between as the client's
      trajectory interpolator does.
The right head camera (the model's input, cam_right_high) is rendered every 1/30 s of simulated time, and the
state (measured joints and gripper openings) and action (the commanded targets) are recorded with each frame.
An episode therefore has the real dataset's shape: 858 frames of video at 30 fps, one state/action row per
frame, the task at its real speed (28.6 s) and no idle start or end.

Why not record the live demo instead (sim_g1_robot.py + replay_policy_server.py + robot_client.py, which a first
version of this script did): measured on a 12-core machine, the live loop runs the 28.6 s task in ~112 s (3.9x
slower than recorded) with ~30 pauses of ~1.2 s while the client waits for each 16-step chunk, and the camera
stream reaches only ~17 fps there (~2 fps on a 2-core machine). Videos written from it are slow-motion and
stop-and-go, at whatever rate the machine reached, while the model is conditioned on 30 fps real-time motion.
This in-process replay is exact in timing and runs faster than real time, several episodes in parallel.

Variation between episodes (the motion itself is the one recorded episode):
    - visual: lighting, a few object/table colours and the head camera's pose (sim_g1_robot.SceneRandomizer,
      as the first version);
    - physical: the camera's and the lid's starting position and heading on the table, by a few millimetres and
      degrees (--object_jitter, --yaw_jitter), so the measured states differ too.
An episode is kept only if it ends with the camera in the box and the lid over it; otherwise it is retried with
another seed (--max_retries).

Output, as prepare_data/prepare_training_data.py writes it (ready to train on):
    <out_dir>/videos/<dataset_name>/cam_right_high/{idx}.mp4    (H.264 via ffmpeg, else OpenCV mp4v)
    <out_dir>/transitions/<dataset_name>/{idx}.h5              (observation.state, action: N x 16)
    <out_dir>/transitions/<dataset_name>/meta_data/stats.safetensors   (over all episodes present)
    <out_dir>/transitions/<dataset_name>/meta_data/episodes.jsonl      (seed and jitter of each episode)
    <out_dir>/<dataset_name>.csv                                       (all episodes present)
Add `<dataset_name>: <weight>` to configs/train/config.yaml's dataset_and_weights, with data_dir pointing at
<out_dir>, to train on it alongside unitree_g1_pack_camera.

Usage:
    python record_sim_dataset.py --episodes 200                     # ~20 min on a 12-core machine
    python record_sim_dataset.py --episodes 100 --start_index 200   # add 100 more to the same dataset
"""

import argparse
import glob
import json
import os

# Rendering backend, before mujoco is imported (sim_g1_robot's own default is egl): EGL renders on the GPU where
# a render node exists; otherwise OSMesa (software). An explicit MUJOCO_GL in the environment wins.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl" if glob.glob("/dev/dri/renderD*") else "osmesa"

import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import sim_g1_robot as sim  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPTS_DIR.parents[1] / "data"
DEFAULT_EPISODE = SCRIPTS_DIR / "replay_data" / "g1_pack_camera_ep0_cover.npz"
VIEW_NAME = "cam_right_high"  # the model's input camera (robot_client.py's CAM_KEY for g1_dex1)
CAMERA = "head_right"
EMBODIMENT = "Unitree G1 Robot with Gripper"  # as in the real unitree_g1_pack_camera dataset
INSTRUCTION = "Pack black camera into box."
CSV_COLUMNS = ["videoid", "contentUrl", "duration", "data_dir", "instruction", "dynamic_confidence",
               "dynamic_wording", "dynamic_source_category", "embodiment"]

# robot_client.py's arm control (unitree_deploy/robot_devices/arm/configs.py, G1ArmConfig): kp_low / kd_low for
# shoulders and elbow, kp_wrist / kd_wrist for the three wrist joints, per arm.
ARM_KP = ([80.0] * 4 + [40.0] * 3) * 2
ARM_KD = ([3.0] * 4 + [1.5] * 3) * 2
GRIPPER_STEP = 0.18  # Dex1_Gripper_Controller.DELTA_GRIPPER_CMD: max command beyond the measured opening
SETTLE_BEFORE, SETTLE_AFTER = 1.0, 1.5  # s held still before recording, and after it before the success check


class EpisodeSimulator:
    """One episode of the packing replay in sim_g1_robot.py's scene, with the client's control, rendering the
    model's camera at the episode's frame rate."""

    def __init__(self, episode: Path, rng: np.random.Generator, visual: bool, object_jitter: float,
                 yaw_jitter: float):
        data = np.load(episode)
        self.fps = int(data["fps"])
        self.actions = data["action"].astype(np.float64)
        self.start_state = data["state"][0].astype(np.float64)

        self.model, body_joints = sim.build_model("g1d")
        m = self.model
        if visual:
            sim.SceneRandomizer(m, CAMERA).randomize(rng)
        self.data = mujoco.MjData(m)
        self.gravity_data = mujoco.MjData(m)  # for the feed-forward torques at the commanded pose
        names = [body_joints[i] for i in sim.ARM_JOINT_INDICES]
        self.arm_qpos = np.array([m.joint(n).qposadr[0] for n in names])
        self.arm_dof = np.array([m.joint(n).dofadr[0] for n in names])
        self.tau_limit = np.array([m.jnt_actfrcrange[m.joint(n).id, 1] for n in names])
        self.kp, self.kd = np.array(ARM_KP), np.array(ARM_KD)
        self.finger_qpos = {s: [m.joint(f"{s}_finger_{f}_joint").qposadr[0] for f in "ab"] for s in sim.GRIPPER_SIDES}
        self.finger_act = {s: [m.actuator(f"{s}_finger_{f}").id for f in "ab"] for s in sim.GRIPPER_SIDES}
        self.sticky = sim.StickyGrasp(m)
        self.tau_ff = np.zeros(14)

        # Start pose of the recording, grippers as recorded, objects jittered on the table.
        d = self.data
        d.qpos[self.arm_qpos] = self.start_state[:14]
        for side, opening in zip(sim.GRIPPER_SIDES, self.start_state[14:]):
            d.qpos[self.finger_qpos[side]] = opening / sim.GRIPPER_Q_MAX * sim.FINGER_TRAVEL
        self.jitter = {}
        for name in ("black_camera", "black_case"):
            adr = m.jnt_qposadr[m.body(name).jntadr[0]]
            shift = rng.uniform(-object_jitter, object_jitter, 2)
            yaw = rng.uniform(-yaw_jitter, yaw_jitter)
            d.qpos[adr : adr + 2] += shift
            turn = np.zeros(4)
            mujoco.mju_axisAngle2Quat(turn, [0, 0, 1], np.deg2rad(yaw))
            mujoco.mju_mulQuat(d.qpos[adr + 3 : adr + 7], turn, d.qpos[adr + 3 : adr + 7].copy())
            self.jitter[name] = {"dx": float(shift[0]), "dy": float(shift[1]), "yaw_deg": float(yaw)}
        mujoco.mj_forward(m, d)

        m.vis.global_.offwidth = max(m.vis.global_.offwidth, sim.CAMERA_WIDTH)
        m.vis.global_.offheight = max(m.vis.global_.offheight, sim.CAMERA_HEIGHT)
        self.renderer = mujoco.Renderer(m, sim.CAMERA_HEIGHT, sim.CAMERA_WIDTH)
        # As sim_g1_robot.py's camera stream renders them.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    def _gravity_torques(self, arm_target):
        """The client's feed-forward torques (G1_29_ArmIK.solve_tau): gravity at the commanded arm pose."""
        g = self.gravity_data
        g.qpos[:] = self.data.qpos
        g.qpos[self.arm_qpos] = arm_target
        g.qvel[:] = 0
        mujoco.mj_kinematics(self.model, g)
        mujoco.mj_comPos(self.model, g)
        bias = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, g, 0, bias)
        return bias[self.arm_dof]

    def opening(self, side):
        return float(np.mean(self.data.qpos[self.finger_qpos[side]]) * sim.GRIPPER_Q_MAX / sim.FINGER_TRAVEL)

    def state(self):
        return np.concatenate([self.data.qpos[self.arm_qpos], [self.opening(s) for s in sim.GRIPPER_SIDES]])

    def _step_towards(self, target):
        """One physics step with the client's control towards a 16-value target (14 arm joints, 2 grippers)."""
        d = self.data
        q, dq = d.qpos[self.arm_qpos], d.qvel[self.arm_dof]
        tau = self.kp * (target[:14] - q) - self.kd * dq + self.tau_ff
        d.qfrc_applied[self.arm_dof] = np.clip(tau, -self.tau_limit, self.tau_limit)
        commands = {}
        for side, goal in zip(sim.GRIPPER_SIDES, target[14:]):
            current = self.opening(side)
            command = float(np.clip(goal, current - GRIPPER_STEP, current + GRIPPER_STEP))
            commands[side] = command
            d.ctrl[self.finger_act[side]] = np.clip(command, 0, sim.GRIPPER_Q_MAX) / sim.GRIPPER_Q_MAX * sim.FINGER_TRAVEL
        self.sticky.update(d, commands)
        mujoco.mj_step(self.model, d)

    def _run_until(self, t_end, target_at):
        while self.data.time < t_end - 1e-9:
            self._step_towards(target_at(self.data.time))

    def run(self, write_frame):
        """Plays the episode; calls write_frame(rgb) for every frame. Returns (states, actions, in_box, covered)."""
        d, dt_frame = self.data, 1.0 / self.fps
        hold_start = self.actions[0]
        self.tau_ff = self._gravity_torques(hold_start[:14])
        self._run_until(SETTLE_BEFORE, lambda t: hold_start)
        t0 = d.time
        states, n = [], len(self.actions)
        for k in range(n):
            # Frame k: what the camera sees and the measured state now, and the command now (action k).
            self.renderer.update_scene(d, camera=CAMERA)
            write_frame(self.renderer.render())
            states.append(self.state())
            a, b = self.actions[k], self.actions[min(k + 1, n - 1)]
            self.tau_ff = self._gravity_torques(b[:14])
            t_k = t0 + k * dt_frame
            # Linear interpolation from action k to k+1 over the frame interval, as the client's interpolator.
            self._run_until(t_k + dt_frame, lambda t: a + (b - a) * min(1.0, (t - t_k) / dt_frame))
        last = self.actions[-1]
        self._run_until(d.time + SETTLE_AFTER, lambda t: last)
        in_box, covered = sim.pack_status(self.model, d, case_held=self.sticky.held["left"] is not None)
        self.renderer.close()
        return np.asarray(states, np.float32), self.actions.astype(np.float32), in_box, covered


class VideoWriter:
    """H.264 (yuv420p) through ffmpeg if available, as the real dataset's videos; else OpenCV's mp4v."""

    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path, self.proc, self.cv = path, None, None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            self.proc = subprocess.Popen(
                [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                 "-threads", "2", str(path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, rgb: np.ndarray):
        if self.proc:
            self.proc.stdin.write(rgb.tobytes())
        else:
            self.cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.path}")
        else:
            self.cv.release()


def generate_episode(task: dict) -> dict:
    """Worker: simulate and write one episode; files appear under their final names only if it succeeded."""
    started = time.perf_counter()
    rng = np.random.default_rng(task["seed"])
    video_path, h5_path = Path(task["video_path"]), Path(task["h5_path"])
    tmp_video = video_path.with_name(f".{video_path.stem}.tmp.mp4")
    episode = EpisodeSimulator(Path(task["episode"]), rng, task["visual"], task["object_jitter"], task["yaw_jitter"])
    writer = VideoWriter(tmp_video, sim.CAMERA_WIDTH, sim.CAMERA_HEIGHT, episode.fps)
    try:
        states, actions, in_box, covered = episode.run(writer.write)
    finally:
        writer.close()
    result = {"idx": task["idx"], "seed": task["seed"], "frames": len(states), "fps": episode.fps,
              "in_box": in_box, "covered": covered, "jitter": episode.jitter,
              "seconds": round(time.perf_counter() - started, 1)}
    if not (in_box and covered):
        tmp_video.unlink(missing_ok=True)
        return {**result, "ok": False}
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_dataset("observation.state", data=states)
        h5f.create_dataset("action", data=actions)
        h5f.attrs["action_type"] = "joint position"
        h5f.attrs["state_type"] = "joint position"
        h5f.attrs["robot_type"] = EMBODIMENT
    tmp_video.replace(video_path)
    return {**result, "ok": True}


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        items.extend(flatten_dict(v, key, sep).items() if isinstance(v, dict) else [(key, v)])
    return dict(items)


def write_metadata(out_dir: Path, dataset_name: str) -> int:
    """CSV and normalisation stats over every complete episode (h5 + mp4) in the dataset, including ones written
    by earlier runs (so appending with --start_index keeps both up to date)."""
    video_dir = out_dir / "videos" / dataset_name / VIEW_NAME
    transitions_dir = out_dir / "transitions" / dataset_name
    ids = sorted(int(p.stem) for p in transitions_dir.glob("*.h5") if p.stem.isdigit()
                 and (video_dir / f"{p.stem}.mp4").exists())
    if not ids:
        return 0
    actions, states = [], []
    for i in ids:
        with h5py.File(transitions_dir / f"{i}.h5", "r") as h5f:
            actions.append(torch.from_numpy(h5f["action"][()]))
            states.append(torch.from_numpy(h5f["observation.state"][()]))
    actions, states = torch.cat(actions), torch.cat(states)
    stats = {key: {"max": t.max(0).values, "min": t.min(0).values, "mean": t.mean(0), "std": t.std(0)}
             for key, t in (("action", actions), ("observation.state", states))}
    save_file(flatten_dict(stats), transitions_dir / "meta_data" / "stats.safetensors")
    rows = [{"videoid": i, "contentUrl": "x", "duration": "x", "data_dir": f"{dataset_name}/{VIEW_NAME}",
             "instruction": INSTRUCTION, "dynamic_confidence": "x", "dynamic_wording": "x",
             "dynamic_source_category": "x", "embodiment": EMBODIMENT} for i in ids]
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(out_dir / f"{dataset_name}.csv", index=False)
    return len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to generate.")
    parser.add_argument("--start_index", type=int, default=0, help="First episode id (to add to a dataset).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed: episode id i uses seed + i (+ 100000 x retry), so a run is reproducible.")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 1) - 2)),
                        help="Episodes simulated in parallel. Rendering (~9 ms a frame on an integrated GPU) is the "
                        "shared bottleneck: on a 12-core machine with an integrated GPU, 6 workers gave 11 "
                        "episodes/min and 10 workers 9. Default: 6, or all cores but 2 if fewer.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Dataset root directory.")
    parser.add_argument("--dataset_name", type=str, default="unitree_g1_pack_camera_sim")
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE,
                        help="Recorded episode to replay (.npz with action, state, fps).")
    parser.add_argument("--object_jitter", type=float, default=0.003,
                        help="Max shift (m) of the camera's and the lid's starting position on the table.")
    parser.add_argument("--yaw_jitter", type=float, default=3.0,
                        help="Max turn (degrees) of the camera's and the lid's starting heading.")
    parser.add_argument("--no_visual_randomization", action="store_true",
                        help="Render every episode with the same lighting, colours and camera pose.")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Seeds tried more per episode id if the camera is not in the box or the lid not on.")
    args = parser.parse_args()

    video_dir = args.out_dir / "videos" / args.dataset_name / VIEW_NAME
    transitions_dir = args.out_dir / "transitions" / args.dataset_name
    for d in (video_dir, transitions_dir / "meta_data"):
        d.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, args.episodes))
    print(f">>> Generating {args.episodes} episodes of {args.dataset_name!r} in {args.out_dir} with {workers} "
          f"workers (MUJOCO_GL={os.environ['MUJOCO_GL']}).", flush=True)

    def task(idx, attempt):
        return {"idx": idx, "attempt": attempt, "seed": args.seed + idx + 100000 * attempt,
                "episode": str(args.episode), "visual": not args.no_visual_randomization,
                "object_jitter": args.object_jitter, "yaw_jitter": args.yaw_jitter,
                "video_path": str(video_dir / f"{idx}.mp4"), "h5_path": str(transitions_dir / f"{idx}.h5")}

    started = time.perf_counter()
    written, given_up = [], []
    log = open(transitions_dir / "meta_data" / "episodes.jsonl", "a")
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pending = {pool.submit(generate_episode, task(i, 0)): task(i, 0)
                   for i in range(args.start_index, args.start_index + args.episodes)}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                t = pending.pop(future)
                try:
                    r = future.result()
                except Exception as exc:  # a crashed worker counts as a failed attempt
                    r = {"idx": t["idx"], "seed": t["seed"], "ok": False, "error": repr(exc)}
                status = "ok" if r["ok"] else (r.get("error") or
                                               f"failed (camera in box: {r.get('in_box')}, covered: {r.get('covered')})")
                print(f">>> episode {r['idx']} seed {r['seed']}: {status}"
                      + (f", {r['frames']} frames at {r['fps']} fps in {r['seconds']} s" if "frames" in r else ""),
                      flush=True)
                if r["ok"]:
                    written.append(r["idx"])
                    log.write(json.dumps(r) + "\n")
                    log.flush()
                elif t["attempt"] < args.max_retries:
                    retry = task(t["idx"], t["attempt"] + 1)
                    pending[pool.submit(generate_episode, retry)] = retry
                else:
                    given_up.append(t["idx"])
    log.close()

    total = write_metadata(args.out_dir, args.dataset_name)
    minutes = (time.perf_counter() - started) / 60
    print(f">>> Wrote {len(written)}/{args.episodes} episodes in {minutes:.1f} min"
          + (f"; gave up on {sorted(given_up)}" if given_up else "") + f". The dataset now has {total} episodes.",
          flush=True)
    print(f">>> To train on it: add '{args.dataset_name}: <weight>' to configs/train/config.yaml's "
          f"dataset_and_weights, with data_dir: {args.out_dir}", flush=True)
    if not written:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
