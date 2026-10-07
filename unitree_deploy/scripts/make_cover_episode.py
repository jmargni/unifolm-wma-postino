"""
Make the "cover" version of the replay episode: the whole "pack black camera into box" recording, where the left
hand's case carry is re-planned so that, in the simulator, it puts the black case (a lid) down over the tray.

On the real robot the recorded joint angles put the case over the tray; in the simulator (slightly different arm,
objects and grip) they carry it low, into the tray's side, and release it beside it. Here the recorded grasp and
first lift are kept; from 1.5 s after the left hand closes on the case, it instead follows waypoints for the case
centre: lift it clear of the tray, carry it over the tray turning it level and in line with it, lower it just above
the tray's rim, wait while the gripper opens at its recorded time (the lid drops down around the tray), lift away and
withdraw above it towards the start pose. After dropping the camera into the tray, the right hand moves up and away
from it (it could catch the case). The arms' joint angles are found by inverse kinematics on the simulator's arm
model. The camera packing and the gripper timing stay as recorded, except that the left gripper stays open after
the release (the recording closes it again as the real hand retracts).

    python make_cover_episode.py            # writes replay_data/g1_pack_camera_ep0_cover.npz
    python replay_policy_server.py --cover  # plays it

Positions are relative to torso_link (x forward, y left, z up), measured in the simulator's scene (sim_g1_robot.py).
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import sim_g1_robot as sim

DATA_DIR = Path(__file__).resolve().parent / "replay_data"
SOURCE = DATA_DIR / "g1_pack_camera_ep0.npz"
TARGET = DATA_DIR / "g1_pack_camera_ep0_cover.npz"
ARMS = {"left": slice(0, 7), "right": slice(7, 14)}  # arm joints in the 16-value state/action vectors
LEFT_ARM = ARMS["left"]

# Case centre and orientation in the left wrist frame once grasped (measured in the simulator, where the sticky grasp
# holds it fixed; the grasp varies by a few mm between runs).
CASE_IN_WRIST = np.array([0.132, -0.003, -0.026])
CASE_ROT_IN_WRIST = np.array([[0.0014, -0.924, -0.3825], [1.0, 0.0029, -0.0034], [0.0043, -0.3825, 0.924]])
# Tray centre (the tray is fixed on the table).
TRAY_XY = np.array(sim.PACK_BOX["pos"])
# Top of the tray's rim (above torso_link), just above the camera lying in it.
RIM_TOP = sim.PACK_TABLE_TOP + sim.PACK_BOX["wall_height"]
CARRY_CLEARANCE = 0.012  # lid bottom above the rim while carrying
PLACE_CLEARANCE = 0.005  # lid bottom above the rim when the gripper opens: it then drops down around the tray
LIFT_AWAY = 0.05  # lift of the open hand before it rejoins the recording
# After dropping the camera the right hand stays next to the tray (on the real robot it steadies it); in the simulator
# it can catch the case once the case lies on the tray. It is moved this far (m, torso x y z: up and away to the right)
# instead, from 1 s after it opens over the tray (sooner, the opening fingers rolled the camera over) to the end.
RIGHT_HAND_AWAY = np.array([0.0, -0.04, 0.10])


class ArmIK:
    """Damped least-squares IK of one arm on the simulator's model, for a point fixed in its wrist."""

    def __init__(self, point, side="left"):
        self.model, body_joints = sim.build_model("g1")
        self.data = mujoco.MjData(self.model)
        names = [body_joints[i] for i in sim.ARM_JOINT_INDICES][ARMS[side]]
        self.qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in names])
        self.dof_adr = np.array([self.model.joint(n).dofadr[0] for n in names])
        self.range = np.array([self.model.jnt_range[self.model.joint(n).id] for n in names])
        self.wrist = self.model.body(f"{side}_wrist_yaw_link").id
        self.point = np.asarray(point, dtype=float)
        mujoco.mj_kinematics(self.model, self.data)
        self.torso = self.data.xpos[self.model.body("torso_link").id].copy()

    def pose(self, q):
        """Point position (relative to torso_link) and wrist rotation for the arm's joints q."""
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jac
        rot = self.data.xmat[self.wrist].reshape(3, 3).copy()
        return self.data.xpos[self.wrist] + rot @ self.point - self.torso, rot

    def solve(self, q_start, target_pos, target_rot, iterations=200, damping=1e-4):
        q = q_start.copy()
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for _ in range(iterations):
            pos, rot = self.pose(q)
            err_rot = np.zeros(3)
            mujoco.mju_subQuat(err_rot, _quat(target_rot), _quat(rot))
            err_rot = rot @ err_rot  # from the wrist frame to the world frame of the Jacobian
            err = np.concatenate([target_pos - pos, 0.3 * err_rot])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err_rot) < 1e-3:
                break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, pos + self.torso, self.wrist)
            jac = np.vstack([jacp[:, self.dof_adr], 0.3 * jacr[:, self.dof_adr]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.1, 0.1), self.range[:, 0], self.range[:, 1])
        return q


def _quat(rot):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def _rot(quat):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    return mat.reshape(3, 3)


def _slerp(q0, q1, s):
    if np.dot(q0, q1) < 0:
        q1 = -q1
    angle = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    if angle < 1e-6:
        return q0
    return (np.sin((1 - s) * angle) * q0 + np.sin(s * angle) * q1) / np.sin(angle)


def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tray", type=float, nargs=2, default=TRAY_XY.tolist(),
                        help="Where to lay the case: tray centre (m, torso x y).")
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()

    data = dict(np.load(SOURCE))
    fps = int(data["fps"])
    actions = data["action"].astype(float)
    left_grip = actions[:, 14]
    # Steps of the recording (30 fps): the left hand closes on the case, then opens again.
    close = int(np.argmax((np.arange(len(left_grip)) > len(left_grip) // 2) & (left_grip < 4.0)))
    release = close + int(np.argmax(left_grip[close:] > 4.0))
    print(f">>> Left hand closes on the case at {close / fps:.1f} s and opens at {release / fps:.1f} s")

    ik = ArmIK(CASE_IN_WRIST)
    # The recorded grasp and first lift stay as they are (they hold the case in the simulator); the path is re-planned
    # from 1.5 s after the hand closes, keeping the hand's recorded orientation.
    start, back = close + 3 * fps // 2, len(actions) - 1
    p0, r0 = ik.pose(actions[start, LEFT_ARM])
    _, r_release = ik.pose(actions[release, LEFT_ARM])
    # Hand orientation that holds the case level and in line with the tray (case axes = tray axes, either way round),
    # the one nearer the recorded hand orientation.
    candidates = [_rot_z(yaw) @ CASE_ROT_IN_WRIST.T for yaw in (0, 180)]
    r_place = min(candidates, key=lambda r: np.arccos(np.clip((np.trace(r.T @ r_release) - 1) / 2, -1, 1)))
    home, r_home = ik.pose(data["state"][0, LEFT_ARM].astype(float))  # the point at the episode's start pose
    half_height = sim.PACK_CASE["half"][2]
    carry_z = RIM_TOP + CARRY_CLEARANCE + half_height
    place_z = RIM_TOP + PLACE_CLEARANCE + half_height
    tray = np.asarray(args.tray)
    # Waypoints of the case centre: (step, position, wrist rotation).
    waypoints = [
        (start, p0, r0),
        (start + fps, np.r_[p0[:2], carry_z], r0),  # lift clear of the tray and the camera in it
        (release - fps, np.r_[tray, carry_z], r_place),  # over the tray
        (release - fps // 3, np.r_[tray, place_z], r_place),  # lower onto the tray
        (release + fps // 2, np.r_[tray, place_z], r_place),  # hold while the gripper opens
        (release + 3 * fps // 2, np.r_[tray, place_z + LIFT_AWAY], r_place),  # lift the open hand away
        # Withdraw above the case towards where the hand starts (the replay server then returns the arms to the
        # start pose). Blending back into the recording instead swept the open fingers back through the case.
        (back, np.r_[home[:2], place_z + LIFT_AWAY], r_home),
    ]

    left = actions[:, LEFT_ARM].copy()
    q = left[start]
    for (k_a, p_a, r_a), (k_b, p_b, r_b) in zip(waypoints, waypoints[1:]):
        for k in range(k_a, k_b + 1):
            s = _min_jerk((k - k_a) / (k_b - k_a))
            target_pos = p_a + s * (p_b - p_a)
            target_rot = _rot(_slerp(_quat(r_a), _quat(r_b), s))
            q = ik.solve(q, target_pos, target_rot)
            reached, _ = ik.pose(q)
            if np.linalg.norm(reached - target_pos) > 0.003:
                raise SystemExit(f"IK did not reach the case waypoint at step {k}: {np.round(reached - target_pos, 4)}")
            left[k] = q

    # Right hand: lifted clear of the tray after it has dropped the camera.
    right_grip = actions[:, 15]
    camera_grasp = int(np.argmax(right_grip < 4.0))
    camera_release = camera_grasp + int(np.argmax(right_grip[camera_grasp:] > 4.0))
    lift_from, lift_full = camera_release + fps, camera_release + 5 * fps // 2
    ik_right = ArmIK([sim.FINGER_X, 0, 0], "right")
    right = actions[:, ARMS["right"]].copy()
    q = right[lift_from]
    for k in range(lift_from, len(actions)):
        pos, rot = ik_right.pose(actions[k, ARMS["right"]])
        away = RIGHT_HAND_AWAY * _min_jerk(min(1.0, (k - lift_from) / (lift_full - lift_from)))
        q = ik_right.solve(q, pos + away, rot)
        if np.linalg.norm(ik_right.pose(q)[0] - pos - away) > 0.003:
            raise SystemExit(f"IK did not reach the lifted right hand position at step {k}")
        right[k] = q
    print(f">>> Right hand moved by {RIGHT_HAND_AWAY} m from {lift_from / fps:.1f} s (camera dropped at "
          f"{camera_release / fps:.1f} s)")

    speed = max(np.abs(np.diff(left, axis=0)).max(), np.abs(np.diff(right, axis=0)).max()) * fps
    print(f">>> Left hand re-planned for steps {start}-{back}; max arm joint speed {speed:.2f} rad/s")
    # The replay server finds the robot in the episode by its state: use the same left arm values for the state.
    for key in ("action", "state"):
        values = data[key].copy()
        values[start : back + 1, LEFT_ARM] = left[start : back + 1]
        values[lift_from:, ARMS["right"]] = right[lift_from:]
        # The recording closes the left gripper again ~1.5 s after the release, as the real hand retracts; in the
        # simulator the hand is still over the case then and would grasp it again. It stays open instead.
        values[release:, 14] = values[release:, 14].max()
        data[key] = values.astype(np.float32)
    data["source"] = (f"{data['source']} | left hand re-planned to lay the case on the tray at {tray.tolist()} "
                      "(make_cover_episode.py)")
    np.savez(args.output, **data)
    print(f">>> Wrote {args.output}")


if __name__ == "__main__":
    main()
