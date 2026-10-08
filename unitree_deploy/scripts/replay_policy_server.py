"""
Replay policy server: plays back a real recorded episode instead of running the model.

It answers POST /predict_action in the same format as
`scripts/evaluation/real_eval_server.py` (and `mock_policy_server.py`), so
`robot_client.py` uses it unchanged. Instead of a model's prediction, it returns
the next 16 actions of a real "pack black camera into box" episode recorded on
a G1 with Dex1 grippers (`replay_data/g1_pack_camera_ep0.npz`, converted from
`examples/world_model_interaction_prompts/transitions/unitree_g1_pack_camera/0.h5`).
The robot then visibly performs the task's movements, e.g. for a demo of the
PEP checks.

- It follows the robot: each request it finds the point of the episode closest
  to the robot's current arm pose and returns the actions from there. If a chunk
  is denied by the PEP or cut by an E-stop, the replay does not run ahead.
- Only the episode's instruction ("Pack black camera into box", case and final
  full stop ignored) is played; for any other instruction it returns the current
  pose, so the robot holds still, as a model that does not know the task.
- By default it plays the first part of the episode, the camera packing: the
  right hand picks up the camera and puts it into the box (13.5 s recorded).
- With --cover it plays the whole sequence: after the camera packing, the left
  hand picks up the black case and lays it on the box as a cover. The case
  carry is re-planned for the simulator (replay_data/
  g1_pack_camera_ep0_cover.npz, made by make_cover_episode.py): the recorded
  joint angles would set it down beside the box there. The rest is as recorded.
- With --full it plays the whole recording unchanged (in the simulator the case
  ends next to the box).
- At the end it brings the arms back to the start pose at a gentle speed and
  holds them there. With --loop it waits there (--loop_pause, 8 s by default)
  and plays again: for a demo that repeats, start the simulator with
  --auto_reset, which puts the objects back while the arms wait.

The simulator's scene (sim_g1_robot.py) places the camera and the box where the
recorded grippers close and open, so the replay picks up the camera and drops
it into the box.
"""

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ACTION_CHUNK = 16  # the client hard-codes 16 predicted future steps
DEFAULT_EPISODE = Path(__file__).resolve().parent / "replay_data" / "g1_pack_camera_ep0.npz"
# The whole episode with the left hand's case carry re-planned for the simulator (make_cover_episode.py).
COVER_EPISODE = Path(__file__).resolve().parent / "replay_data" / "g1_pack_camera_ep0_cover.npz"
SEARCH_BACK, SEARCH_AHEAD = 8, 40  # episode steps around the last position searched for the robot's pose
GRIPPER_WEIGHT = 0.1  # rad per unit of gripper opening, when comparing the robot's pose with the episode's
RETURN_SPEED = 1.0  # rad/s, arm speed when moving back to the start pose
PACKING_END = 13.5  # s of the recording: camera released into the box and right hand lifted away
AT_START = 0.05  # rad, max joint distance to the start pose to restart the episode


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


class ReplayPolicy:
    def __init__(self, episode: Path, stride: int, loop: bool, accept: list[str], full: bool = False,
                 loop_pause: float = 8.0):
        data = np.load(episode)
        end = None if full else int(PACKING_END * int(data["fps"]))
        # The model's actions are 2 recorded frames apart (frame_stride 2 at 30 fps = 15 Hz, the client's
        # --control_freq 15): the replay uses the same spacing.
        self.actions = data["action"][:end:stride].astype(np.float32)
        self.states = data["state"][:end:stride].astype(np.float32)
        self.instruction = str(data["instruction"])
        self.accepted = {normalize(self.instruction), *(normalize(a) for a in accept)}
        self.control_freq = int(data["fps"]) / stride
        self.loop = loop
        self.loop_pause = loop_pause
        self.restart_at = None  # with --loop: when to play again, once back at the start pose
        self.index = 0  # episode step closest to the robot's pose at the last request
        self.returning = False
        self.finished = False  # back at the start pose after the episode (without --loop: hold there)
        self.num_requests = 0
        self.lock = threading.Lock()

    def predict_action(self, payload: dict) -> list:
        states = np.asarray(payload["observation.state"], dtype=np.float32)
        current = states[-1]
        instruction = payload["language_instruction"]
        with self.lock:
            self.num_requests += 1
            n = self.num_requests
            if normalize(instruction) not in self.accepted:
                actions = np.repeat(current[None], ACTION_CHUNK, axis=0)
                status = "unknown instruction: holding still"
            else:
                actions, status = self._next_chunk(current)
        print(f"[{n}] instruction: {instruction!r} | {status}", flush=True)
        return actions.tolist()

    def _next_chunk(self, current: np.ndarray):
        last = len(self.actions) - 1
        start = self.states[0]
        # Hold the recorded start pose itself: sending back the measured pose would let the arms sag a little
        # more at every request.
        hold = np.repeat(start[None], ACTION_CHUNK, axis=0)
        if self.finished:
            return hold, "episode done: holding the start pose"
        if self.restart_at is not None:
            # --loop: hold the start pose for the pause (the simulator's --auto_reset puts the objects back), then
            # play again from the start.
            if time.monotonic() < self.restart_at:
                return hold, f"episode done: playing again in {self.restart_at - time.monotonic():.0f} s"
            self.restart_at, self.index = None, 0
        elif self.returning:
            # Back to the start pose at a limited speed, then hold there or (--loop) play again after a pause.
            if np.abs(current[:14] - start[:14]).max() <= AT_START:
                self.returning, self.index = False, 0
                if not self.loop:
                    self.finished = True
                    return hold, "episode done: holding the start pose"
                self.restart_at = time.monotonic() + self.loop_pause
                return hold, f"episode done: playing again in {self.loop_pause:.0f} s"
            else:
                max_step = RETURN_SPEED / self.control_freq
                actions = np.empty((ACTION_CHUNK, current.shape[0]), dtype=np.float32)
                pose = current.copy()
                for k in range(ACTION_CHUNK):
                    pose[:14] += np.clip(start[:14] - pose[:14], -max_step, max_step)
                    pose[14:] = start[14:]
                    actions[k] = pose
                return actions, "returning to the start pose"

        # Where is the robot in the episode? Nearest recorded pose around the last position: the arm joints, and the
        # gripper openings (scaled down: 0-5.4) so that while an arm holds still, before and after a gripper opens
        # or closes are told apart.
        lo, hi = max(0, self.index - SEARCH_BACK), min(last, self.index + SEARCH_AHEAD)
        distances = np.maximum(np.abs(self.states[lo : hi + 1, :14] - current[:14]).max(axis=1),
                               GRIPPER_WEIGHT * np.abs(self.states[lo : hi + 1, 14:16] - current[14:16]).max(axis=1))
        self.index = lo + int(np.argmin(distances))

        if self.index + ACTION_CHUNK > last:
            # Play the last steps of the episode, then hold its last pose.
            actions = self.actions[self.index :]
            actions = np.concatenate([actions, np.repeat(actions[-1:], ACTION_CHUNK - len(actions), axis=0)])
            if self.index >= last - 1:
                self.returning = True
                return actions, "episode finished: returning to the start pose next"
            return actions, f"episode step {self.index}/{last}: last steps"
        actions = self.actions[self.index : self.index + ACTION_CHUNK]
        return actions, f"episode step {self.index}/{last} (pose error {distances.min():.3f} rad)"


def make_handler(policy: ReplayPolicy):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path != "/predict_action":
                self.send_error(404)
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                response = {"result": "ok", "action": policy.predict_action(payload), "desc": "success"}
            except Exception as e:
                response = {"result": "error", "desc": repr(e)}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE, help="Recorded episode (.npz).")
    parser.add_argument("--stride", type=int, default=2,
                        help="Recorded frames per action: 2 plays the 30 fps recording at the client's 15 Hz.")
    parser.add_argument("--loop", action="store_true",
                        help="Play the episode again after returning to the start pose and waiting --loop_pause s "
                        "(start the simulator with --auto_reset to put the objects back meanwhile).")
    parser.add_argument("--loop_pause", type=float, default=8.0,
                        help="With --loop: seconds to hold the start pose before playing again.")
    parser.add_argument("--full", action="store_true",
                        help="Play the whole recording unchanged (camera packing, then the left hand moving the black "
                        "case; in the simulator the case ends next to the box).")
    parser.add_argument("--cover", action="store_true",
                        help="Play the whole sequence with the case carry re-planned for the simulator: the camera "
                        "goes into the box and the case is laid on it as a cover.")
    parser.add_argument("--accept", action="append", default=[],
                        help="Another instruction to treat as the episode's task (repeatable).")
    args = parser.parse_args()

    if args.cover:
        args.episode, args.full = COVER_EPISODE, True
    policy = ReplayPolicy(args.episode, args.stride, args.loop, args.accept, args.full, args.loop_pause)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f">>> Replay policy server is ready on http://{args.host}:{args.port} ...", flush=True)
    print(f">>> Episode: {len(policy.actions)} actions at {policy.control_freq:g} Hz "
          f"({len(policy.actions) / policy.control_freq:.0f} s) for {sorted(policy.accepted)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(">>> Replay policy server stops ...")


if __name__ == "__main__":
    main()
