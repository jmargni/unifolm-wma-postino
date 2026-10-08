"""
Generate a simulator-rendered training set for fine-tuning UnifoLM-WMA-0 on "pack black camera into box", using
MuJoCo-rendered frames instead of real camera frames: the real model performs poorly when driven by
sim_g1_robot.py's cameras because their images differ enough from the real dataset's, and that gap is closed by
fine-tuning on (sim video, state, action) sequences rather than only real ones.

How each episode is produced, and why it is built this way: this launches the real, documented setup --
    python sim_g1_robot.py --robot g1d --headless --seed <per-episode>
    python replay_policy_server.py --cover
    UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15
-- as actual subprocesses and taps their live ZMQ camera stream and DDS state/commands, instead of replaying the
recorded episode's actions directly against MuJoCo physics in-process (which would be much faster: no real-time
wait, no subprocesses). That faster approach was tried first and reliably failed to grasp the camera or the case
at all, two different ways (PD-tracking the recorded actions with the real deployed controller's own gains; and
kinematically puppeting the recorded states). Both skip the real deployed controller's actual closed-loop,
continuously-compliant torque control, which turns out to matter: this is a delicate, contact-rich small-object
grasp, and only driving it for real -- the robot_client.py + replay_policy_server.py + sim_g1_robot.py stack
exactly as used interactively -- reproduced the task's real outcome (confirmed repeatedly: "IN THE BOX" and
"BOX COVERED" in sim_g1_robot.py's own status line). Note also that --robot g1 (not g1d) reproducibly stalled
partway through the cover episode in testing; g1d did not.

Each episode's camera stream is rendered with a freshly randomized scene (see sim_g1_robot.py's SceneRandomizer
and --seed): lighting, a few object/table colours and the head camera's own pose, so the *visuals* differ between
episodes while the motion -- the one proven trajectory replay_policy_server.py --cover plays back -- stays the
same.

Output matches what prepare_data/prepare_training_data.py produces, so it is ready to train on directly:
    <out_dir>/videos/<dataset_name>/cam_right_high/{idx}.mp4
    <out_dir>/transitions/<dataset_name>/{idx}.h5              (observation.state, action)
    <out_dir>/transitions/<dataset_name>/meta_data/stats.safetensors
    <out_dir>/<dataset_name>.csv
Add `<dataset_name>: <weight>` to configs/train/config.yaml's dataset_and_weights (data_dir pointing at <out_dir>)
to train on it alongside unitree_g1_pack_camera.

Each episode takes roughly as long as the live demo does (~1.5-2 minutes), so 50 episodes is on the order of an
hour or more -- run it unattended. Needs the unitree_deploy conda environment, plus h5py (not in its
pyproject.toml: pip install h5py if missing).

MUJOCO_GL defaults to osmesa (pure software; override with --mujoco-gl), not sim_g1_robot.py's own egl default:
on a machine with no working /dev/dri render node, egl falls back to an extremely slow (and sometimes blank)
path. This also has to be set *before* `import sim_g1_robot` below, not just when launching the subprocesses:
that import runs sim_g1_robot.py's own MUJOCO_GL default as a side effect in this process too, and on at least
one development machine, letting that take effect (even though this process never renders) was enough to make
the *subprocess's* independent osmesa rendering fail as well, via mujoco/__init__.py's own silently-swallowed
`except ImportError: pass` around importing Renderer -- surfacing only as a sim_g1_robot.py log full of
`AttributeError: module 'mujoco' has no attribute 'Renderer'` and zero frames captured. wait_for_camera() detects
this (or the plain slow-EGL case) within a few seconds and retries instead of wasting a whole --duration finding
out.

On a CPU-constrained machine (2 cores on the one this was developed on), the written video's fps can end up far
below sim_g1_robot.py's --camera_fps=30 default -- rendering the camera stream, robot_client.py and this
recorder are all competing for the same cores, and camera rendering alone can peg close to a full core. This is
a real limit of that hardware, not a bug, so stop_episode() measures the actual achieved fps from the frames'
own arrival times and that is what gets written (duration and state/action timing stay internally honest even
at, say, 2 fps instead of 30) rather than assuming the nominal rate.

Usage:
    python record_sim_dataset.py --episodes 50
"""

import argparse
import json
import os

# Before `import sim_g1_robot` below: that import runs sim_g1_robot.py's own
# os.environ.setdefault("MUJOCO_GL", "egl") as a side effect, in *this* process. On a machine with no working
# /dev/dri render node, letting that actually take effect here -- even though this process never renders -- was
# enough to break the *subprocess's* independent osmesa rendering a moment later (confirmed: reproducible with
# this import present, gone once MUJOCO_GL is already set before it runs). So this is set first, unconditionally.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import zmq
from safetensors.torch import save_file
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

import sim_g1_robot as sim

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPTS_DIR.parents[1] / "data"
VIEW_NAME = "cam_right_high"
CAMERA_COLUMN = sim.STREAM_CAMERAS.index("head_right") * sim.CAMERA_WIDTH
EMBODIMENT = "Unitree G1 Robot with Gripper"
ROBOT_TYPE = "g1_dex1"
INSTRUCTION = "Pack black camera into box."
IMAGE_PORT = 5555
VIDEO_FPS = 30  # matches sim_g1_robot.py's --camera_fps default


class Recorder:
    """Taps the live run's own ZMQ camera stream (cropped to head_right, i.e. ImageClient's cam_right_high) and
    DDS state/commands while it performs the task, pairing them by the camera's own frame arrivals. One instance
    is reused for every episode (subscriptions survive the publishing process restarting between episodes; DDS/
    ZMQ both reconnect transparently)."""

    def __init__(self, image_port: int = IMAGE_PORT, network_interface: str | None = None):
        ChannelFactoryInitialize(0, network_interface)
        self.lock = threading.Lock()
        self._arm_state = np.array(sim.START_ARM_POSE, dtype=np.float32)
        self._arm_action = np.array(sim.START_ARM_POSE, dtype=np.float32)
        self._grip_state = {"left": sim.GRIPPER_Q_MAX, "right": sim.GRIPPER_Q_MAX}
        self._grip_action = {"left": sim.GRIPPER_Q_MAX, "right": sim.GRIPPER_Q_MAX}
        self.frames, self.states, self.actions, self.times = [], [], [], []

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_lowstate, 10)
        self._lowcmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._lowcmd_sub.Init(self._on_lowcmd, 10)
        self._grip_subs = []
        for side in sim.GRIPPER_SIDES:
            s = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
            s.Init(self._make_grip_cb(side, self._grip_state), 10)
            c = ChannelSubscriber(f"rt/dex1/{side}/cmd", MotorCmds_)
            c.Init(self._make_grip_cmd_cb(side), 10)
            self._grip_subs += [s, c]

        self._recording = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._zmq_loop, args=(image_port,), daemon=True)
        self._thread.start()

    def _on_lowstate(self, msg):
        arm = [msg.motor_state[i].q for i in sim.ARM_JOINT_INDICES]
        with self.lock:
            self._arm_state = np.asarray(arm, dtype=np.float32)

    def _on_lowcmd(self, msg):
        arm = [msg.motor_cmd[i].q for i in sim.ARM_JOINT_INDICES]
        with self.lock:
            self._arm_action = np.asarray(arm, dtype=np.float32)

    def _make_grip_cb(self, side, store):
        def cb(msg):
            if msg.states:
                with self.lock:
                    store[side] = float(msg.states[0].q)
        return cb

    def _make_grip_cmd_cb(self, side):
        def cb(msg):
            if msg.cmds:
                with self.lock:
                    self._grip_action[side] = float(msg.cmds[0].q)
        return cb

    def _zmq_loop(self, image_port: int):
        socket = zmq.Context().socket(zmq.SUB)
        socket.connect(f"tcp://127.0.0.1:{image_port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop.is_set():
            try:
                buf = socket.recv()
            except zmq.Again:
                continue
            if not self._recording.is_set():
                continue
            frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            head_right = frame[:, CAMERA_COLUMN: CAMERA_COLUMN + sim.CAMERA_WIDTH].copy()
            with self.lock:
                state16 = np.concatenate([self._arm_state, [self._grip_state["left"], self._grip_state["right"]]])
                action16 = np.concatenate([self._arm_action, [self._grip_action["left"], self._grip_action["right"]]])
            with self._buffer_lock:
                self.frames.append(head_right)
                self.states.append(state16.astype(np.float32))
                self.actions.append(action16.astype(np.float32))
                self.times.append(time.monotonic())

    _buffer_lock = threading.Lock()

    def start_episode(self):
        with self._buffer_lock:
            self.frames, self.states, self.actions, self.times = [], [], [], []
        self._recording.set()

    def stop_episode(self):
        """Returns (frames, states (N,16), actions (N,16), fps), fps measured from the frames' own arrival times
        -- not assumed to be sim_g1_robot.py's --camera_fps, since on a CPU-constrained machine the camera
        subprocess can fall well behind its nominal rate (confirmed: ~2 fps actual against a 30 fps target when
        rendering competes with robot_client.py and this recorder for only 2 cores). Using the real rate keeps
        the written video's duration and the state/action timing honest; using a stale nominal 30 would silently
        teach a world model the wrong amount of motion per step."""
        self._recording.clear()
        with self._buffer_lock:
            frames, times = list(self.frames), list(self.times)
            states = np.stack(self.states) if self.states else np.empty((0, 16), dtype=np.float32)
            actions = np.stack(self.actions) if self.actions else np.empty((0, 16), dtype=np.float32)
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else VIDEO_FPS
        return frames, states, actions, fps

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)


class FFmpegWriter:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise FileNotFoundError("ffmpeg")
        self.proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
             str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray):
        self.proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg exited with an error (see its message above).")


class Cv2Writer:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame_bgr: np.ndarray):
        self.writer.write(frame_bgr)

    def release(self):
        self.writer.release()


def make_writer(path: Path, width: int, height: int, fps: int):
    try:
        return FFmpegWriter(path, width, height, fps)
    except FileNotFoundError:
        print(">>> ffmpeg not found on PATH: writing mp4v via OpenCV instead. If your training pipeline needs "
              f"H.264, re-encode afterwards: ffmpeg -i {path} -c:v libx264 {path}", flush=True)
        return Cv2Writer(path, width, height, fps)


def flatten_dict(d, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        items.extend(flatten_dict(v, new_key, sep).items() if isinstance(v, dict) else [(new_key, v)])
    return dict(items)


def run_live_episode(args, seed: int, log_path: Path) -> subprocess.Popen:
    """Starts sim_g1_robot.py (the only one whose output we need, for the success check) and returns it; the
    caller starts replay_policy_server.py and robot_client.py around the recorder's own start_episode() window."""
    # Force, not setdefault: importing sim_g1_robot above already ran its own os.environ.setdefault("MUJOCO_GL",
    # "egl") as a side effect, inside *this* process -- so by now MUJOCO_GL is already "egl" in os.environ
    # whether or not the user (or the --mujoco-gl default below) wanted osmesa, and setdefault here would be a
    # silent no-op. On a machine with no /dev/dri permission, egl falls back to an extremely slow (and possibly
    # blank) software path -- confirmed to cut a 30 fps, ~100 s capture down to ~130 total frames.
    env = dict(os.environ)
    env["MUJOCO_GL"] = args.mujoco_gl
    log = open(log_path, "w")
    return subprocess.Popen(
        ["python", "sim_g1_robot.py", "--robot", "g1d", "--headless", "--seed", str(seed)],
        cwd=SCRIPTS_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
    )


def terminate(*procs: subprocess.Popen):
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)


def check_success(log_path: Path, require_cover: bool) -> bool:
    text = log_path.read_text(errors="replace")[-3000:]
    return ("BOX COVERED" if require_cover else "IN THE BOX") in text


def wait_for_camera(log_path: Path, timeout: float) -> str | None:
    """Polls sim_g1_robot.py's own log for its camera subprocess coming up, instead of guessing a fixed delay:
    on a machine with only a couple of CPU cores (confirmed on the one this was developed on), mujoco's own
    Renderer import can transiently fail under the load of starting 3-4 heavy processes at once -- silently, via
    mujoco/__init__.py's own `except ImportError: pass` around it -- which otherwise only surfaces as zero
    frames captured after waiting out the whole --duration. Returns None once ready, or an error string if the
    camera subprocess crashed (so the caller can restart sim_g1_robot.py instead of recording a dead stream)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log_path.read_text(errors="replace")
        if "Traceback" in text:
            return text.strip().splitlines()[-1] if text.strip() else "unknown error"
        if "Camera stream on" in text:
            return None
        time.sleep(0.3)
    return f"camera stream did not start within {timeout:g} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50, help="Number of sequences to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode i is seeded with seed + i.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Target dataset root directory.")
    parser.add_argument("--dataset-name", type=str, default="unitree_g1_pack_camera_sim")
    parser.add_argument("--start-index", type=int, default=0, help="First episode id (for appending to a dataset).")
    parser.add_argument("--duration", type=float, default=100.0,
                        help="Seconds to record per episode (empirically enough margin for --cover's ~29 s task "
                        "plus the return-to-start-and-hold tail; shorten for the pack-only task).")
    parser.add_argument("--startup-delay", type=float, default=15.0,
                        help="Max seconds to wait for sim_g1_robot.py's camera stream to come up before the "
                        "replay server and client connect (polled, so usually much faster in practice; a low-"
                        "core machine under load may occasionally need the whole budget -- see wait_for_camera).")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Retries per episode slot if the task did not succeed (IN THE BOX / BOX COVERED not "
                        "seen) before giving up on it and moving on.")
    parser.add_argument("--log-dir", type=Path, default=Path(tempfile.gettempdir()) / "record_sim_dataset_logs")
    parser.add_argument("--mujoco-gl", type=str, default="osmesa",
                        help="MUJOCO_GL for the sim_g1_robot.py/robot_client.py subprocesses. osmesa (pure "
                        "software, default) is what was validated; egl needs a working /dev/dri render node or "
                        "it silently falls back to an extremely slow (and possibly blank) path.")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    target_dir = args.out_dir
    video_dir = target_dir / "videos" / args.dataset_name / VIEW_NAME
    transitions_dir = target_dir / "transitions" / args.dataset_name
    meta_dir = transitions_dir / "meta_data"
    for d in (video_dir, transitions_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Absolute floor only (not fps-proportional): the real achieved camera fps is whatever this machine's CPU
    # budget allows -- confirmed as low as ~2 fps against a 30 fps nominal target when sim_g1_robot.py's camera
    # subprocess, robot_client.py and this recorder all compete for just 2 cores -- so a fixed expected frame
    # count would reject a legitimately slow-but-working machine. This just catches a near-empty capture (crash,
    # wrong port, ...), with the written video's own fps measured from frame arrival times instead of assumed.
    min_frames = 10
    recorder = Recorder()
    rows, all_actions, all_states, failures = [], [], [], []
    try:
        for i in range(args.episodes):
            idx = args.start_index + i
            seed = args.seed + i
            ok, frames, states, actions, fps = False, [], None, None, VIDEO_FPS
            for attempt in range(args.max_retries + 1):
                log_path = args.log_dir / f"sim_{idx}_attempt{attempt}.log"
                sim_proc = replay_proc = client_proc = None
                camera_error = None
                try:
                    sim_proc = run_live_episode(args, seed, log_path)
                    camera_error = wait_for_camera(log_path, args.startup_delay)
                    if camera_error is None:
                        replay_proc = subprocess.Popen(["python", "replay_policy_server.py", "--cover"],
                                                       cwd=SCRIPTS_DIR, stdout=subprocess.DEVNULL,
                                                       stderr=subprocess.DEVNULL)
                        time.sleep(1.0)
                        client_env = dict(os.environ, UNITREE_IMAGE_SERVER="127.0.0.1", MUJOCO_GL=args.mujoco_gl)
                        client_proc = subprocess.Popen(["python", "robot_client.py", "--control_freq", "15"],
                                                       cwd=SCRIPTS_DIR, env=client_env, stdout=subprocess.DEVNULL,
                                                       stderr=subprocess.DEVNULL)

                        recorder.start_episode()
                        time.sleep(args.duration)
                        frames, states, actions, fps = recorder.stop_episode()
                finally:
                    terminate(*(p for p in (client_proc, replay_proc, sim_proc) if p is not None))

                if camera_error is not None:
                    print(f">>> episode {idx} attempt {attempt}: sim_g1_robot.py's camera stream never came up "
                          f"({camera_error}), retrying", flush=True)
                    continue

                success_marker = check_success(log_path, require_cover=True)
                ok = success_marker and len(frames) >= min_frames
                reason = ("SUCCESS" if ok else
                         "no success marker in sim_g1_robot.py log" if not success_marker else
                         f"only {len(frames)} frames captured (<{min_frames}; check MUJOCO_GL/CPU load in {log_path})")
                print(f">>> episode {idx} attempt {attempt}: {len(frames)} frames at {fps:.1f} fps (measured), "
                      f"{reason}", flush=True)
                if ok:
                    break

            if not ok:
                failures.append(idx)
                print(f">>> episode {idx}: giving up after {args.max_retries + 1} attempts, skipping.", flush=True)
                continue
            if fps < VIDEO_FPS * 0.8:
                print(f">>> episode {idx}: camera ran at only {fps:.1f} fps (target {VIDEO_FPS}) -- this "
                      f"machine's CPU is the bottleneck (see sim_g1_robot.py/robot_client.py CPU use), not a "
                      f"bug; writing the video at its real measured fps so duration and timing stay honest.",
                      flush=True)

            writer = make_writer(video_dir / f"{idx}.mp4", sim.CAMERA_WIDTH, sim.CAMERA_HEIGHT, round(fps))
            for frame in frames:
                writer.write(frame)
            writer.release()

            with h5py.File(transitions_dir / f"{idx}.h5", "w") as h5f:
                h5f.create_dataset("observation.state", data=states)
                h5f.create_dataset("action", data=actions)
                h5f.attrs["action_type"] = "joint position"
                h5f.attrs["state_type"] = "joint position"
                h5f.attrs["robot_type"] = ROBOT_TYPE

            rows.append({
                "videoid": idx, "contentUrl": "x", "duration": "x",
                "data_dir": f"{args.dataset_name}/{VIEW_NAME}", "instruction": INSTRUCTION,
                "dynamic_confidence": "x", "dynamic_wording": "x", "dynamic_source_category": "x",
                "embodiment": EMBODIMENT,
            })
            all_actions.append(torch.from_numpy(actions))
            all_states.append(torch.from_numpy(states))
    finally:
        recorder.close()

    if not rows:
        raise SystemExit("No episode succeeded -- nothing written. Check the logs in " + str(args.log_dir))

    actions_t = torch.cat(all_actions, dim=0)
    states_t = torch.cat(all_states, dim=0)
    stats = {
        "action": {"max": actions_t.max(0).values, "min": actions_t.min(0).values,
                   "mean": actions_t.mean(0), "std": actions_t.std(0)},
        "observation.state": {"max": states_t.max(0).values, "min": states_t.min(0).values,
                              "mean": states_t.mean(0), "std": states_t.std(0)},
    }
    save_file(flatten_dict(stats), meta_dir / "stats.safetensors")

    csv_path = target_dir / f"{args.dataset_name}.csv"
    columns = ["videoid", "contentUrl", "duration", "data_dir", "instruction", "dynamic_confidence",
              "dynamic_wording", "dynamic_source_category", "embodiment"]
    existing = pd.read_csv(csv_path) if csv_path.exists() and args.start_index > 0 else pd.DataFrame(columns=columns)
    pd.concat([existing, pd.DataFrame(rows, columns=columns)], ignore_index=True).to_csv(csv_path, index=False)

    print(f">>> Wrote {len(rows)}/{args.episodes} episodes of {args.dataset_name!r} under {target_dir} "
          f"({len(failures)} failed: {failures})", flush=True)
    print(f">>> Add '{args.dataset_name}: <weight>' to configs/train/config.yaml's dataset_and_weights, with "
          f"data_dir: {target_dir}", flush=True)


if __name__ == "__main__":
    main()
