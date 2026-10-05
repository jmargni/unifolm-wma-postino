"""Bridge from robot_client.py to the policy enforcement point (PEP) of the DGS G1-D Cyber Twin.

Before a chunk of actions is executed, it is sent to the twin as one `trajectory` command
(POST /api/command). The twin checks it (identity, armed / E-stop state, replay, URDF joint
limits, joint speed), records the decision in its audit log / SIEM, and mirrors the end pose
in its 3D console. The chunk may be executed only if the twin answers ALLOW, or MONITOR when
the twin's enforcement is switched off (the violation is logged but not blocked).

The bridge fails closed: if the twin cannot be reached or rejects the request, nothing runs.
It never arms the robot or clears an E-stop; that is done by a person in the twin's consoles.
"""

import time
import uuid
from dataclasses import dataclass

import numpy as np
import requests

# Order of the 14 arm values in the g1_dex1 state and action vectors (see robot_configs.g1_motors),
# by the joint names of the G1-D URDF used by the twin.
G1_DEX1_ARM_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# Dex1 gripper opening: 0 = closed, about 5.4 = open. The twin's hands are only open or closed.
GRIPPER_CLOSED_BELOW = 2.7
PEP_PRINCIPAL = "operator"
PEP_TARGET = "g1d-01"


@dataclass
class PepDecision:
    allowed: bool
    decision: str  # ALLOW, MONITOR, DENY, or REJECTED / UNREACHABLE when the twin gave no decision
    message: str
    rule_id: str = ""
    event_id: str = ""


class TwinPEP:
    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def state(self) -> dict:
        """The twin's current robot state (mode, joints, gripper)."""
        response = self.session.get(f"{self.base_url}/api/robot", timeout=self.timeout)
        response.raise_for_status()
        return response.json()["robot"]

    def may_continue(self) -> bool:
        """Checked before every step of an allowed chunk: False once the twin is no longer armed
        (E-stop or disarm from a console), or if it cannot be reached."""
        try:
            return self.state()["mode"] == "armed"
        except (requests.RequestException, ValueError, KeyError):
            return False

    def check_trajectory(self, actions: np.ndarray, dt: float, start: np.ndarray | None = None) -> PepDecision:
        """Ask the PEP whether a chunk of g1_dex1 actions (N x 16) may be executed.

        `dt` is the time between actions, `start` the measured state (16 values) before the first one:
        the PEP limits the joint speed from `start` to the first action and between actions.
        """
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2 or actions.shape[1] < 16:
            raise ValueError(f"Expected N x 16 g1_dex1 actions, got shape {actions.shape}")
        params = {
            "joints": list(G1_DEX1_ARM_JOINTS),
            "points": np.round(actions[:, :14], 5).tolist(),
            "dt": float(dt),
            # Closed if either gripper ends closed.
            "gripper": "closed" if actions[-1, 14:16].min() < GRIPPER_CLOSED_BELOW else "open",
        }
        if start is not None:
            params["start"] = np.round(np.asarray(start, dtype=float)[:14], 5).tolist()
        # Identity, route and target are set here, never taken from the model's output.
        payload = {
            "action": "trajectory",
            "params": params,
            "request_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "identity": PEP_PRINCIPAL,
            "via": "pep",
            "target": PEP_TARGET,
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/command",
                json={"payload": payload, "principal": PEP_PRINCIPAL},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return PepDecision(False, "UNREACHABLE", f"Twin PEP not reachable: {exc}")
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code != 200:
            return PepDecision(False, "REJECTED", f"HTTP {response.status_code}: {body.get('error', response.text[:200])}")
        decision = body.get("decision", "")
        return PepDecision(
            allowed=decision in ("ALLOW", "MONITOR"),
            decision=decision,
            message=body.get("message", ""),
            rule_id=body.get("rule_id", ""),
            event_id=body.get("event_id", ""),
        )
