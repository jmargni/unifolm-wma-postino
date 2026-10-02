"""
Fake G1 + Dex1 grippers for local testing of `robot_client.py` without a robot.

It speaks the same DDS topics as the real robot:
    publishes   rt/lowstate              (joint positions of the 29 body motors)
    subscribes  rt/lowcmd                (joint targets sent by the client)
    publishes   rt/dex1/{left,right}/state
    subscribes  rt/dex1/{left,right}/cmd

The robot is kinematic: each joint is moved straight to its commanded position
and shown in a MuJoCo viewer. There is no physics (no gravity, contacts or
motor dynamics) and no camera, so the client receives black images.
"""

import argparse
import os
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

import unitree_deploy

XML_PATH = Path(unitree_deploy.__path__[0]) / "robot_devices" / "assets" / "g1" / "g1_body29.xml"
NUM_BODY_MOTORS = 29
GRIPPER_SIDES = ("left", "right")


class MockG1:
    def __init__(self, network_interface: str | None = None):
        self.lock = threading.Lock()
        self.q = np.zeros(NUM_BODY_MOTORS)
        self.gripper_q = dict.fromkeys(GRIPPER_SIDES, 0.0)
        self.num_lowcmd = 0
        self.num_gripper_cmd = 0

        ChannelFactoryInitialize(0, network_interface)

        self.lowstate_publisher = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher.Init()
        self.lowcmd_subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.lowcmd_subscriber.Init(self._on_lowcmd, 10)

        self.gripper_publishers = {}
        self.gripper_subscribers = {}
        for side in GRIPPER_SIDES:
            self.gripper_publishers[side] = ChannelPublisher(f"rt/dex1/{side}/state", MotorStates_)
            self.gripper_publishers[side].Init()
            self.gripper_subscribers[side] = ChannelSubscriber(f"rt/dex1/{side}/cmd", MotorCmds_)
            self.gripper_subscribers[side].Init(lambda msg, side=side: self._on_gripper_cmd(side, msg), 10)

    def _on_lowcmd(self, msg: LowCmd_):
        with self.lock:
            for i in range(NUM_BODY_MOTORS):
                self.q[i] = msg.motor_cmd[i].q
            self.num_lowcmd += 1

    def _on_gripper_cmd(self, side: str, msg: MotorCmds_):
        if len(msg.cmds) == 0:
            return
        with self.lock:
            self.gripper_q[side] = msg.cmds[0].q
            self.num_gripper_cmd += 1

    def snapshot(self):
        with self.lock:
            return self.q.copy(), dict(self.gripper_q), self.num_lowcmd, self.num_gripper_cmd

    def publish_state(self):
        q, gripper_q, _, _ = self.snapshot()

        lowstate = unitree_hg_msg_dds__LowState_()
        for i in range(NUM_BODY_MOTORS):
            lowstate.motor_state[i].q = float(q[i])
            lowstate.motor_state[i].dq = 0.0
        self.lowstate_publisher.Write(lowstate)

        for side in GRIPPER_SIDES:
            state = unitree_go_msg_dds__MotorState_()
            state.q = float(gripper_q[side])
            state.dq = 0.0
            self.gripper_publishers[side].Write(MotorStates_(states=[state]))

    def publish_loop(self, stop_event: threading.Event, rate_hz: float):
        while not stop_event.is_set():
            start = time.perf_counter()
            self.publish_state()
            time.sleep(max(0, 1 / rate_hz - (time.perf_counter() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Do not open the MuJoCo viewer window.")
    parser.add_argument("--network_interface", type=str, default=None, help="DDS network interface (default: auto).")
    parser.add_argument("--state_freq", type=float, default=500, help="State publishing frequency in Hz.")
    args = parser.parse_args()

    robot = MockG1(args.network_interface)

    stop_event = threading.Event()
    threading.Thread(target=robot.publish_loop, args=(stop_event, args.state_freq), daemon=True).start()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    # qpos address of each body motor, in G1_29_JointIndex order (the free joint of the base is skipped).
    qpos_adr = [model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(qpos_adr) == NUM_BODY_MOTORS, f"Expected {NUM_BODY_MOTORS} hinge joints, got {len(qpos_adr)}"

    viewer = None if args.headless else mujoco.viewer.launch_passive(model, data)
    print(">>> Mock G1 is publishing state on DDS. Start robot_client.py now. Ctrl+C to stop.", flush=True)

    last_log = time.monotonic()
    try:
        while viewer is None or viewer.is_running():
            q, gripper_q, num_lowcmd, num_gripper_cmd = robot.snapshot()
            data.qpos[qpos_adr] = q
            mujoco.mj_forward(model, data)
            if viewer is not None:
                viewer.sync()

            if time.monotonic() - last_log > 2.0:
                last_log = time.monotonic()
                print(
                    f"lowcmd msgs: {num_lowcmd} | gripper cmd msgs: {num_gripper_cmd} | "
                    f"arm q: {np.round(q[15:], 2)} | grippers: "
                    f"{[round(gripper_q[side], 2) for side in GRIPPER_SIDES]}",
                    flush=True,
                )
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if viewer is not None:
            viewer.close()
        print(">>> Mock G1 stopped.", flush=True)
        # Skip interpreter teardown: the DDS threads crash the process on a normal exit.
        os._exit(0)


if __name__ == "__main__":
    main()
