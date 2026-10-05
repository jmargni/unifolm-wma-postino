"""
Physics simulation of a G1 (or G1-D) with two Dex1 grippers, for testing `robot_client.py` without a robot.

It replaces the robot side of the real setup, using the same interfaces:
    DDS  publishes   rt/lowstate              (positions, velocities and torques of the 29 body motors)
         subscribes  rt/lowcmd                (per-motor q, dq, kp, kd, tau sent by the client)
         publishes   rt/dex1/{left,right}/state
         subscribes  rt/dex1/{left,right}/cmd
    ZMQ  publishes   tcp://*:5555             (JPEG camera stream, same layout as the robot's image server)

Unlike `mock_g1_robot.py`, the robot is simulated with MuJoCo physics:
    - every motor is driven by the PD law of the real motor driver,
      tau = kp * (q_cmd - q) + kd * (dq_cmd - dq) + tau_ff, clipped to the motor torque limit;
    - gravity, contacts and friction act on the arms, the grippers and the objects;
    - the pelvis is fixed in place (the real robot balances with Unitree's own controller, not simulated here);
    - the head stereo camera and the two wrist cameras are rendered and streamed.

The scene is a table with a black "camera" and an open box. The Dex1 grippers are simplified two-finger
grippers; their opening (0 = closed, 5.45 = open) maps linearly to the finger travel.

With --robot g1d the robot is the G1-D of the DGS Cyber Twin (Unitree's g1_d_description URDF, in
assets/g1d): wheeled base, lifting column and torso held fixed, the same 14-joint arms as the G1, and the same
Dex1-like grippers in place of its three-finger hands. Its arm motors use the G1's indices 15-28 on rt/lowstate
and rt/lowcmd; the other 15 indices (G1 legs and waist) do not exist on it and are reported at 0. This mapping
is this simulator's choice: how a real G1-D numbers its motors over DDS has not been checked.

To make the client read the simulated cameras, start it with:
    UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py
"""

import argparse
import multiprocessing as mp
import os
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Offscreen rendering of the cameras uses EGL. The viewer window uses GLFW and is not affected.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_, unitree_hg_msg_dds__LowState_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402

import unitree_deploy  # noqa: E402

ASSETS = Path(unitree_deploy.__path__[0]) / "robot_devices" / "assets"
XML_PATH = ASSETS / "g1" / "g1_body29.xml"
G1D_URDF_PATH = ASSETS / "g1d" / "g1_d.urdf"
NUM_BODY_MOTORS = 29
ARM_JOINT_INDICES = range(15, 29)  # G1_29_JointIndex of the 14 arm joints, the only ones the G1-D shares with the G1
# G1-D joints held fixed: wheels, lifting column (retracted, as in the twin's default) and torso yaw/pitch.
G1D_FIXED_JOINTS = ("Left_Wheel_Joint", "Right_Wheel_Joint", "LZ_mt_Joint", "LZ_it_Joint", "Yaw_Joint", "torso_Joint")
# The G1-D URDF has no joint armature or friction; these are the values of Unitree's G1 MJCF (class "g1"),
# whose arms are identical to the G1-D's (same meshes, joint origins, axes and limits).
ARM_ARMATURE = 0.01
ARM_FRICTIONLOSS = 0.3
# Table and objects are placed for the G1's torso; for the G1-D they follow its torso.
G1_TORSO_POS = (-0.004, 0.837)  # x, z of torso_link with the G1's pelvis fixed at its standing height
GRIPPER_SIDES = ("left", "right")
MODE_MACHINE = 5  # g1_29dof_rev_1_0, see assets/g1/README.md

# Gripper: the Dex1 command range [0, GRIPPER_Q_MAX] maps to a travel of [0, FINGER_TRAVEL] metres per finger.
GRIPPER_Q_MAX = 5.45
FINGER_TRAVEL = 0.04

# Gains used until the first rt/lowcmd arrives, so the arms hold the zero pose instead of falling.
DEFAULT_KP = 100.0
DEFAULT_KD = 3.0

CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640
# Stream layout expected by ImageClient: [head left | head right | left wrist | right wrist], side by side.
STREAM_CAMERAS = ("head_left", "head_right", "left_wrist", "right_wrist")

TABLE_TOP_Z = 0.76


def _xyaxes_quat(x_axis, y_axis):
    """Quaternion of a frame whose x and y axes are given (MuJoCo cameras look along their -z axis)."""
    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_axis, dtype=float)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    mat = np.column_stack([x, y, np.cross(x, y)])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def _camera_quat(pitch_down_deg):
    """Camera looking along +x (forward), tilted down by `pitch_down_deg`, image right = -y."""
    pitch = np.deg2rad(pitch_down_deg)
    return _xyaxes_quat([0, -1, 0], [np.sin(pitch), 0, np.cos(pitch)])


def _add_gripper(spec, side):
    """Replace the rubber hand of one wrist by a two-finger gripper and a wrist camera."""
    wrist = spec.body(f"{side}_wrist_yaw_link")
    for geom in list(wrist.geoms):
        if geom.meshname.endswith("rubber_hand"):
            spec.delete(geom)

    wrist.add_geom(
        name=f"{side}_gripper_palm",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.075, 0, 0],
        size=[0.02, 0.04, 0.022],
        mass=0.3,
        rgba=[0.15, 0.15, 0.15, 1],
    )
    for finger, sign in (("a", 1), ("b", -1)):
        body = wrist.add_body(name=f"{side}_finger_{finger}", pos=[0.125, sign * 0.006, 0])
        body.add_joint(
            name=f"{side}_finger_{finger}_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, sign, 0],
            range=[0, FINGER_TRAVEL],
            damping=1.0,
        )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.03, 0.005, 0.018],
            mass=0.05,
            friction=[1.5, 0.01, 0.001],
            condim=4,
            rgba=[0.3, 0.3, 0.3, 1],
        )
        actuator = spec.add_actuator(
            name=f"{side}_finger_{finger}",
            target=f"{side}_finger_{finger}_joint",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[0, FINGER_TRAVEL],
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            forcerange=[-20, 20],
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        actuator.set_to_position(kp=500, kv=10)

    # Above the palm, looking past the fingers.
    wrist.add_camera(name=f"{side}_wrist", pos=[0.04, 0, 0.045], quat=_camera_quat(15), fovy=90)


def _add_scene(spec, dx=0.0, table_top_z=TABLE_TOP_Z):
    """Table, camera and box. `dx` shifts the scene forward, `table_top_z` sets the table height."""
    world = spec.worldbody
    world.add_light(pos=[0.6, 0, 2.5], dir=[0, 0, -1], diffuse=[0.3, 0.3, 0.3])

    # Table in front of the robot.
    table_size = [0.25, 0.6, 0.02]
    world.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[dx + 0.25 + table_size[0], 0, table_top_z - table_size[2]],
        size=table_size,
        rgba=[0.55, 0.4, 0.28, 1],
    )
    for x in (0.27, 0.73):
        for y in (-0.57, 0.57):
            world.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[dx + x, y, (table_top_z - 0.04) / 2],
                size=[0.02, 0.02, (table_top_z - 0.04) / 2],
                rgba=[0.4, 0.3, 0.2, 1],
            )

    # Black "camera" to pick up.
    camera = world.add_body(name="black_camera", pos=[dx + 0.42, 0.15, table_top_z + 0.035])
    camera.add_freejoint()
    camera.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.03, 0.035], mass=0.3, rgba=[0.05, 0.05, 0.05, 1])
    camera.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[-0.045, 0, 0],
        quat=[0.7071068, 0, 0.7071068, 0],
        size=[0.022, 0.02],
        mass=0.05,
        rgba=[0.1, 0.1, 0.1, 1],
    )

    # Open cardboard box to put it in.
    box = world.add_body(name="box", pos=[dx + 0.45, -0.18, table_top_z])
    box.add_freejoint()
    half, height, wall = 0.09, 0.08, 0.005
    cardboard = [0.72, 0.55, 0.35, 1]
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, wall], size=[half, half, wall], mass=0.1, rgba=cardboard)
    for pos, size in (
        ([half, 0, height / 2], [wall, half, height / 2]),
        ([-half, 0, height / 2], [wall, half, height / 2]),
        ([0, half, height / 2], [half, wall, height / 2]),
        ([0, -half, height / 2], [half, wall, height / 2]),
    ):
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, mass=0.05, rgba=cardboard)


def _g1_spec():
    """Unitree's G1 with the pelvis fixed in place. Returns the spec and the 29 body joint names."""
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    # Names of the body joints, in G1_29_JointIndex order (same order as the actuators of the file).
    body_joints = [actuator.target for actuator in spec.actuators]
    assert len(body_joints) == NUM_BODY_MOTORS, f"Expected {NUM_BODY_MOTORS} actuators, got {len(body_joints)}"

    # The motors are simulated by applying the PD torque directly, so the position actuators are removed.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    spec.delete(spec.joint("floating_base_joint"))
    for key in list(spec.keys):
        spec.delete(key)

    # Grey background instead of black, so the camera images look like a room.
    for texture in spec.textures:
        if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            texture.rgb1 = [0.75, 0.78, 0.8]
            texture.rgb2 = [0.45, 0.48, 0.5]
            texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    return spec, body_joints


# World of the G1-D: the same floor, sky and lights as the G1's MJCF file.
G1D_WORLD_XML = """
<mujoco model="g1d_world">
  <option integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.78 0.8" rgb2="0.45 0.48 0.5" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def _g1d_robot_spec():
    """The G1-D URDF as a spec, without its hands and with its base, column and torso joints removed (fixed)."""
    urdf = ET.parse(G1D_URDF_PATH).getroot()
    for tag in ("joint", "link"):
        for element in list(urdf.findall(tag)):
            if "hand" in element.get("name"):
                urdf.remove(element)
    compiler = urdf.find("mujoco/compiler")
    compiler.set("meshdir", str(G1D_URDF_PATH.parent))
    compiler.set("fusestatic", "false")  # keep torso_link and head_link as bodies for the cameras
    robot = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    for name in G1D_FIXED_JOINTS:
        robot.delete(robot.joint(name))
    return robot


def _g1d_spec():
    """Unitree's G1-D on the floor, with its base, column and torso held fixed and its three-finger hands removed
    (Dex1-style grippers are mounted instead). Returns the spec, the 29 body joint names in G1_29_JointIndex
    order (the 14 arm joints have the G1's names and indices, the others do not exist: None), and the x, z
    position of its torso_link."""
    # Measure on a separate copy: a spec must not be compiled and then modified (body references get mixed up).
    model = _g1d_robot_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Height that puts the lowest point of the base and wheels on the floor.
    lowest = np.inf
    for g in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        mesh = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        lowest = min(lowest, (verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])[:, 2].min())
    torso_x, _, torso_z = data.xpos[model.body("torso_link").id]

    spec = mujoco.MjSpec.from_string(G1D_WORLD_XML)
    spec.attach(_g1d_robot_spec(), frame=spec.worldbody.add_frame(pos=[0, 0, -lowest]), prefix="")

    body_joints = [None] * NUM_BODY_MOTORS
    g1_joint_names = _g1_spec()[1]
    for i in ARM_JOINT_INDICES:
        body_joints[i] = g1_joint_names[i]
        joint = spec.joint(g1_joint_names[i])
        joint.armature = ARM_ARMATURE
        joint.frictionloss = ARM_FRICTIONLOSS
    return spec, body_joints, (torso_x, torso_z - lowest)


def _set_contact_group(body, contype, conaffinity):
    """Set the contact bits of every colliding geom of `body` and its descendants."""
    for geom in body.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = contype, conaffinity
    for child in body.bodies:
        _set_contact_group(child, contype, conaffinity)


def build_model(robot="g1"):
    """G1 (fixed pelvis) or G1-D (fixed base) with Dex1-like grippers, head and wrist cameras, and a table scene."""
    dx, table_top_z = 0.0, TABLE_TOP_Z
    if robot == "g1d":
        spec, body_joints, (torso_x, torso_z) = _g1d_spec()
        # Place the table and objects relative to the torso, as for the G1.
        dx, table_top_z = torso_x - G1_TORSO_POS[0], TABLE_TOP_Z + torso_z - G1_TORSO_POS[1]
    else:
        spec, body_joints = _g1_spec()

    # Stereo camera in the face, 6 cm apart, looking down at the table.
    torso = spec.body("torso_link")
    for name, y in (("head_left", 0.03), ("head_right", -0.03)):
        torso.add_camera(name=name, pos=[0.08, y, 0.38], quat=_camera_quat(50), fovy=65)

    for side in GRIPPER_SIDES:
        _add_gripper(spec, side)
    _add_scene(spec, dx, table_top_z)
    if robot == "g1d":
        # The G1-D URDF's collision meshes overlap at the wrist joints. The robot's own parts (grippers
        # included) do not collide with each other, but still collide with the table, objects and floor.
        _set_contact_group(spec.body("AGV_link"), contype=2, conaffinity=1)

    spec.option.timestep = 0.002
    return spec.compile(), body_joints


class SimG1:
    def __init__(self, model, body_joints, network_interface: str | None = None):
        self.model = model
        self.data = mujoco.MjData(model)

        # Motors that exist in this robot (all 29 on the G1, the 14 arm motors on the G1-D). The others are
        # reported at 0 and their commands are ignored.
        self.motors = np.array([i for i, name in enumerate(body_joints) if name is not None])
        names = [body_joints[i] for i in self.motors]
        self.qpos_adr = np.array([model.joint(name).qposadr[0] for name in names])
        self.dof_adr = np.array([model.joint(name).dofadr[0] for name in names])
        self.tau_limit = np.array([model.jnt_actfrcrange[model.joint(name).id, 1] for name in names])
        self.finger_qpos_adr = {
            side: [model.joint(f"{side}_finger_{f}_joint").qposadr[0] for f in "ab"] for side in GRIPPER_SIDES
        }
        self.finger_dof_adr = {
            side: [model.joint(f"{side}_finger_{f}_joint").dofadr[0] for f in "ab"] for side in GRIPPER_SIDES
        }
        self.finger_actuators = {side: [model.actuator(f"{side}_finger_{f}").id for f in "ab"] for side in GRIPPER_SIDES}
        mujoco.mj_forward(model, self.data)

        # Latest commands, written by the DDS callbacks and read by the physics loop.
        self.lock = threading.Lock()
        self.q_cmd = np.zeros(NUM_BODY_MOTORS)
        self.dq_cmd = np.zeros(NUM_BODY_MOTORS)
        self.kp = np.full(NUM_BODY_MOTORS, DEFAULT_KP)
        self.kd = np.full(NUM_BODY_MOTORS, DEFAULT_KD)
        self.tau_ff = np.zeros(NUM_BODY_MOTORS)
        self.gripper_q_cmd = dict.fromkeys(GRIPPER_SIDES, 0.0)
        self.num_lowcmd = 0
        self.num_gripper_cmd = 0
        self.tau = np.zeros(NUM_BODY_MOTORS)

        ChannelFactoryInitialize(0, network_interface)

        self.lowstate = unitree_hg_msg_dds__LowState_()
        self.lowstate.mode_machine = MODE_MACHINE
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
        cmds = msg.motor_cmd[:NUM_BODY_MOTORS]
        with self.lock:
            self.q_cmd[:] = [c.q for c in cmds]
            self.dq_cmd[:] = [c.dq for c in cmds]
            self.kp[:] = [c.kp for c in cmds]
            self.kd[:] = [c.kd for c in cmds]
            self.tau_ff[:] = [c.tau for c in cmds]
            self.num_lowcmd += 1

    def _on_gripper_cmd(self, side: str, msg: MotorCmds_):
        if len(msg.cmds) == 0:
            return
        with self.lock:
            self.gripper_q_cmd[side] = float(np.clip(msg.cmds[0].q, 0.0, GRIPPER_Q_MAX))
            self.num_gripper_cmd += 1

    def step(self):
        """Apply the motor torques and advance the physics by one timestep."""
        data = self.data
        q = data.qpos[self.qpos_adr]
        dq = data.qvel[self.dof_adr]
        m = self.motors
        with self.lock:
            tau = self.kp[m] * (self.q_cmd[m] - q) + self.kd[m] * (self.dq_cmd[m] - dq) + self.tau_ff[m]
            gripper_q_cmd = dict(self.gripper_q_cmd)
        self.tau[m] = np.clip(tau, -self.tau_limit, self.tau_limit)
        data.qfrc_applied[self.dof_adr] = self.tau[m]

        for side in GRIPPER_SIDES:
            data.ctrl[self.finger_actuators[side]] = gripper_q_cmd[side] / GRIPPER_Q_MAX * FINGER_TRAVEL

        mujoco.mj_step(self.model, data)

    def gripper_q(self, side):
        """Gripper opening and its velocity in Dex1 units (mean of the two fingers)."""
        scale = GRIPPER_Q_MAX / FINGER_TRAVEL
        q = np.mean(self.data.qpos[self.finger_qpos_adr[side]]) * scale
        dq = np.mean(self.data.qvel[self.finger_dof_adr[side]]) * scale
        return float(q), float(dq)

    def joint_q(self):
        """Positions of the 29 body motors, in G1_29_JointIndex order (0 for motors this robot does not have)."""
        q = np.zeros(NUM_BODY_MOTORS)
        q[self.motors] = self.data.qpos[self.qpos_adr]
        return q

    def publish_state(self):
        q = self.joint_q()
        dq = np.zeros(NUM_BODY_MOTORS)
        dq[self.motors] = self.data.qvel[self.dof_adr]
        for i in range(NUM_BODY_MOTORS):
            motor = self.lowstate.motor_state[i]
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau_est = float(self.tau[i])
        self.lowstate.tick += 1
        self.lowstate_publisher.Write(self.lowstate)

        for side in GRIPPER_SIDES:
            state = unitree_go_msg_dds__MotorState_()
            state.q, state.dq = self.gripper_q(side)
            self.gripper_publishers[side].Write(MotorStates_(states=[state]))

    def counters(self):
        with self.lock:
            return self.num_lowcmd, self.num_gripper_cmd


def camera_server(shared_qpos, port: int, fps: float, jpeg_quality: int, robot: str):
    """Child process: render the cameras from the latest joint positions and stream them like the robot does."""
    model, _ = build_model(robot)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, CAMERA_HEIGHT, CAMERA_WIDTH)
    # Shadows and reflections make rendering about 5x slower on an integrated GPU.
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

    socket = zmq.Context().socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    print(f">>> Camera stream on tcp://*:{port} ({len(STREAM_CAMERAS)} cameras at {fps:g} fps).", flush=True)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    parent_pid = os.getppid()
    try:
        # Stop if the simulator dies without terminating this process (e.g. a crash).
        while os.getppid() == parent_pid:
            start = time.perf_counter()
            with shared_qpos.get_lock():
                data.qpos[:] = shared_qpos[:]
            mujoco.mj_forward(model, data)
            images = []
            for camera in STREAM_CAMERAS:
                renderer.update_scene(data, camera=camera)
                images.append(renderer.render())
            frame = cv2.cvtColor(np.hstack(images), cv2.COLOR_RGB2BGR)  # the client expects BGR
            ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                socket.send(jpg.tobytes())
            time.sleep(max(0, 1 / fps - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass


def viewer_loop(viewer, view_data, shared_qpos, rate_hz: float = 60):
    """Show the latest pose. Runs in its own thread because viewer.sync() blocks while the window redraws."""
    while viewer.is_running():
        with shared_qpos.get_lock():
            view_data.qpos[:] = shared_qpos[:]
        mujoco.mj_forward(viewer.m, view_data)
        viewer.sync()
        time.sleep(1 / rate_hz)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="Do not open the MuJoCo viewer window.")
    parser.add_argument("--network_interface", type=str, default=None, help="DDS network interface (default: auto).")
    parser.add_argument(
        "--state_freq",
        type=float,
        default=100,
        help="State publishing frequency in Hz. Higher rates slow the simulation below real time: "
        "each DDS message costs about 0.7 ms of Python time.",
    )
    parser.add_argument("--no_cameras", action="store_true", help="Do not render and stream the cameras.")
    parser.add_argument("--image_port", type=int, default=5555, help="TCP port of the camera stream.")
    parser.add_argument("--camera_fps", type=float, default=30, help="Frame rate of the camera stream.")
    parser.add_argument("--jpeg_quality", type=int, default=80, help="JPEG quality of the camera stream (0-100).")
    parser.add_argument(
        "--robot",
        choices=("g1", "g1d"),
        default="g1",
        help="g1: humanoid G1 with fixed pelvis. g1d: G1-D (wheeled base and column, held fixed), the robot of the "
        "DGS Cyber Twin. Both with the same arms and Dex1-like grippers.",
    )
    args = parser.parse_args()

    model, body_joints = build_model(args.robot)
    robot = SimG1(model, body_joints, args.network_interface)
    data = robot.data

    # Latest joint positions, read by the camera process and the viewer thread.
    ctx = mp.get_context("spawn")
    shared_qpos = ctx.Array("d", model.nq)
    shared_qpos[:] = data.qpos

    camera_process = None
    if not args.no_cameras:
        camera_process = ctx.Process(
            target=camera_server,
            args=(shared_qpos, args.image_port, args.camera_fps, args.jpeg_quality, args.robot),
            daemon=True,
        )
        camera_process.start()

    viewer = None
    if not args.headless:
        # The viewer gets its own copy of the data: it cannot read the physics data while a step writes it.
        # As a consequence, objects cannot be dragged with the mouse.
        view_data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, view_data)
        threading.Thread(target=viewer_loop, args=(viewer, view_data, shared_qpos), daemon=True).start()
    print(f">>> Simulated {'G1-D' if args.robot == 'g1d' else 'G1'} is running. Start robot_client.py now. "
          "Ctrl+C to stop.", flush=True)

    dt = model.opt.timestep
    publish_every = max(1, round(1 / (args.state_freq * dt)))
    wall_start = time.perf_counter()
    sim_start = data.time
    next_share = time.monotonic()
    next_log = next_share + 2.0
    log_sim_time, log_wall_time = data.time, time.perf_counter()
    try:
        while viewer is None or viewer.is_running():
            # Keep the simulation in step with the wall clock; if it falls behind by more than
            # 0.1 s, drop the backlog instead of running faster than real time to catch up.
            lag = (time.perf_counter() - wall_start) - (data.time - sim_start)
            if lag > 0.1:
                wall_start += lag
            elif lag < 0:
                time.sleep(-lag)

            robot.step()
            if round(data.time / dt) % publish_every == 0:
                robot.publish_state()

            now = time.monotonic()
            if now >= next_share:
                next_share = now + 1 / 60
                with shared_qpos.get_lock():
                    shared_qpos[:] = data.qpos

            if now >= next_log:
                next_log = now + 2.0
                wall = time.perf_counter()
                realtime = (data.time - log_sim_time) / max(wall - log_wall_time, 1e-9)
                log_sim_time, log_wall_time = data.time, wall
                num_lowcmd, num_gripper_cmd = robot.counters()
                arm_q = robot.joint_q()[ARM_JOINT_INDICES]
                grippers = [round(robot.gripper_q(side)[0], 2) for side in GRIPPER_SIDES]
                print(
                    f"real-time x{realtime:.2f} | lowcmd msgs: {num_lowcmd} | gripper cmd msgs: {num_gripper_cmd} | "
                    f"arm q: {np.round(arm_q, 2)} | grippers: {grippers}",
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        if camera_process is not None:
            camera_process.terminate()
            camera_process.join(timeout=2)
        print(">>> Simulated G1 stopped.", flush=True)
        # Skip interpreter teardown: the DDS threads crash the process on a normal exit.
        os._exit(0)


if __name__ == "__main__":
    main()
