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

The scene is laid out from the real "pack black camera into box" episode replayed by replay_policy_server.py:
a black camera (robot's right), an open box (centre) and a black case (left) on a table, where the recorded
grippers close and open, so the replay really packs the camera. The Dex1 grippers are simplified two-finger
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
import json
import multiprocessing as mp
import os
import threading
import time
import urllib.request
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
# With --twin_url, the G1-D's base (planar joints added here), wheels, column and torso follow the Cyber Twin's
# state kinematically, at most at these speeds (the twin changes column height and joints instantly).
G1D_BASE_JOINTS = ("base_x", "base_y", "base_yaw")
G1D_FOLLOW_SPEED = {"base_x": 1.0, "base_y": 1.0, "base_yaw": 2.0, "LZ_mt_Joint": 0.1, "LZ_it_Joint": 0.1,
                    "Yaw_Joint": 1.0, "torso_Joint": 1.0}  # m/s or rad/s; wheels follow the twin directly
# Parts of the G1-D below the torso: they do not collide when the base moves (they would scrape the floor).
G1D_BASE_BODIES = ("AGV_link", "Left_Wheel_Link", "RIght_Wheel_Link", "LZ_ot_Link", "LZ_mt_Link", "LZ_it_Link",
                   "Pitching_Link", "Yaw_Link")
# The G1-D URDF has no joint armature or friction; these are the values of Unitree's G1 MJCF (class "g1"),
# whose arms are identical to the G1-D's (same meshes, joint origins, axes and limits).
ARM_ARMATURE = 0.01
ARM_FRICTIONLOSS = 0.3
# Table and objects are placed for the G1's torso; for the G1-D they follow its torso.
G1_TORSO_POS = (-0.004, 0.837)  # x, z of torso_link with the G1's pelvis fixed at its standing height
GRIPPER_SIDES = ("left", "right")
MODE_MACHINE = 5  # g1_29dof_rev_1_0, see assets/g1/README.md

# Gripper: the Dex1 command range [0, GRIPPER_Q_MAX] maps to a travel of [0, FINGER_TRAVEL] metres per finger.
# Finger pads along the wrist's x axis, from FINGER_X - FINGER_HALF_LENGTH to FINGER_X + FINGER_HALF_LENGTH (m).
FINGER_X, FINGER_HALF_LENGTH, FINGER_TIP_X = 0.1225, 0.0225, 0.133
GRIPPER_YELLOW = [0.95, 0.75, 0.1, 1]
FINGER_TIP_RED = [0.85, 0.1, 0.08, 1]
GRIPPER_Q_MAX = 5.45
FINGER_TRAVEL = 0.04

# Head stereo camera (torso frame), calibrated from the G1_Dex1_MountCameraRedGripper dataset: the red finger tips
# in 33 real frames of each view against the simulated arms at the recorded joint angles (median error 8 px).
HEAD_CAMERAS = {
    "head_left": ([0.0798, 0.0038, 0.4744], [0.7059, 0.2019, -0.1938, -0.6506], 60.3),
    "head_right": ([0.0789, -0.0048, 0.4728], [0.6868, 0.1829, -0.2218, -0.6675], 60.4),
}

# Arm pose at start-up, held until the first rt/lowcmd: the first pose of the recorded "pack black camera into
# box" episode, the same as robot_client.py's INIT_POSE for g1_dex1. (The zero pose would go through the table.)
START_ARM_POSE = np.array([-0.50163078, 0.32051945, 0.18178487, 0.6838522, -0.085745335, -0.51033354, -0.25062084,
                           -0.40424705, -0.26256371, -0.20069599, 0.21123028, 0.1912303, -0.20726478, 0.28874797])

# Gains used until the first rt/lowcmd arrives, so the arms hold the start pose instead of falling.
DEFAULT_KP = 100.0
DEFAULT_KD = 3.0

CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640
# Stream layout expected by ImageClient: [head left | head right | left wrist | right wrist], side by side.
STREAM_CAMERAS = ("head_left", "head_right", "left_wrist", "right_wrist")



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
    """Replace the rubber hand of one wrist by a two-finger gripper and a wrist camera. Proportions of the real
    Dex1: the red finger tips are 0.133 m from the wrist, as calibrated from the dataset's head-camera images."""
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
        rgba=GRIPPER_YELLOW,
    )
    for finger, sign in (("a", 1), ("b", -1)):
        body = wrist.add_body(name=f"{side}_finger_{finger}", pos=[FINGER_X, sign * 0.006, 0])
        body.add_joint(
            name=f"{side}_finger_{finger}_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, sign, 0],
            range=[0, FINGER_TRAVEL],
            damping=1.0,
        )
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[FINGER_HALF_LENGTH, 0.005, 0.018],
            mass=0.05,
            friction=[1.5, 0.01, 0.001],
            condim=4,
            rgba=[0.12, 0.12, 0.12, 1],
        )
        # Red rubber tip (visual only), where the real Dex1 has it.
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, pos=[FINGER_TIP_X - FINGER_X, 0, 0],
                      size=[0.014, 0.0065, 0.012], contype=0, conaffinity=0, mass=0.001, rgba=FINGER_TIP_RED)
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


# Scene of the "pack black camera into box" task. Objects have the size and look of the real ones (measured in the
# G1_Dex1_MountCameraRedGripper dataset images with the calibrated head camera) and are placed where the grippers of
# the recorded episode replayed by replay_policy_server.py (replay_data/g1_pack_camera_ep0.npz) close and open.
# Positions are relative to torso_link, which is at the same height on the G1 and G1-D (x forward, y left, z up;
# m and degrees; yaw = turn about z of the object's x axis).
PACK_TABLE_TOP = 0.036  # just below the lowest the finger pads reach over the table
PACK_CAMERA = dict(pos=(0.286, -0.144), yaw=-27.6, half=(0.015, 0.039, 0.011))  # 7.8 x 3.0 x 2.2 cm; right hand closes
PACK_BOX = dict(pos=(0.248, -0.035), inner_half=(0.019, 0.042), wall_height=0.025, wall=0.004)  # white tray, 9.2 x 4.6 cm
PACK_CASE = dict(pos=(0.284, 0.149), yaw=15.8, half=(0.026, 0.052, 0.020))  # 10.4 x 5.2 x 4 cm; left hand closes
TABLE_WHITE = [0.86, 0.86, 0.84, 1]
SCENE_HEADLIGHT, SCENE_LIGHT = 0.2, 0.1
OBJECT_BLACK = [0.04, 0.04, 0.05, 1]


def _yaw_quat(deg):
    return [np.cos(np.deg2rad(deg) / 2), 0, 0, np.sin(np.deg2rad(deg) / 2)]


def _add_open_box(body, half_x, half_y, height, wall, mass, rgba, inside_rgba=None):
    """Open-top box made of a floor and four walls, with its bottom at the body origin."""
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, wall / 2], size=[half_x + wall, half_y + wall, wall / 2],
                  mass=mass / 5, rgba=inside_rgba or rgba)
    for pos, size in (
        ([half_x + wall / 2, 0, height / 2], [wall / 2, half_y + wall, height / 2]),
        ([-half_x - wall / 2, 0, height / 2], [wall / 2, half_y + wall, height / 2]),
        ([0, half_y + wall / 2, height / 2], [half_x, wall / 2, height / 2]),
        ([0, -half_y - wall / 2, height / 2], [half_x, wall / 2, height / 2]),
    ):
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, mass=mass / 5, rgba=rgba)


def _add_scene(spec, torso_x, torso_z):
    """White table with the black camera (right), the white tray (centre) and the black case (left), placed
    relative to the robot's torso as in the recorded "pack black camera into box" episode."""
    world = spec.worldbody
    world.add_light(pos=[0.6, 0, 2.5], dir=[0, 0, -1], diffuse=[0.3, 0.3, 0.3])
    top = torso_z + PACK_TABLE_TOP

    # Table in front of the robot.
    table_size = [0.3, 0.6, 0.02]
    table_x = torso_x + 0.15 + table_size[0]
    world.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[table_x, 0, top - table_size[2]],
                   size=table_size, rgba=TABLE_WHITE)
    # Visual-only extension towards the robot: in the real images the table runs under the arms.
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[torso_x + 0.1, 0, top - 0.001], size=[0.05, table_size[1], 0.001],
                   contype=0, conaffinity=0, rgba=TABLE_WHITE)
    for x in (table_x - table_size[0] + 0.03, table_x + table_size[0] - 0.03):
        for y in (-table_size[1] + 0.03, table_size[1] - 0.03):
            world.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, (top - 0.04) / 2],
                           size=[0.02, 0.02, (top - 0.04) / 2], rgba=[0.3, 0.3, 0.32, 1])

    def place(spec_):
        return [torso_x + spec_["pos"][0], spec_["pos"][1]]

    # Black camera, picked up by the right hand: its narrow side faces the closing fingers. Screen at one end.
    half = PACK_CAMERA["half"]
    camera = world.add_body(name="black_camera", pos=[*place(PACK_CAMERA), top + half[2]],
                            quat=_yaw_quat(PACK_CAMERA["yaw"]))
    camera.add_freejoint()
    camera.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.06, friction=[1.2, 0.01, 0.001],
                    rgba=OBJECT_BLACK)
    camera.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -half[1] + 0.012, half[2]], size=[half[0] - 0.003, 0.010, 0.0006],
                    contype=0, conaffinity=0, mass=0.001, rgba=[0.65, 0.72, 1.0, 1])

    # White tray with a black rim, where the right hand releases the camera.
    box = world.add_body(name="box", pos=[*place(PACK_BOX), top])
    box.add_freejoint()
    _add_open_box(box, *PACK_BOX["inner_half"], PACK_BOX["wall_height"], PACK_BOX["wall"], mass=0.1,
                  rgba=[0.95, 0.95, 0.95, 1], inside_rgba=[0.97, 0.97, 0.97, 1])
    hx, hy = (h + PACK_BOX["wall"] for h in PACK_BOX["inner_half"])
    for pos, size in (([hx, 0], [0.0025, hy]), ([-hx, 0], [0.0025, hy]), ([0, hy], [hx, 0.0025]), ([0, -hy], [hx, 0.0025])):
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[*pos, PACK_BOX["wall_height"]], size=[*size, 0.0015],
                     contype=0, conaffinity=0, mass=0.001, rgba=OBJECT_BLACK)

    # Black case (the lid), with the blue window strip of the real one on top.
    half = PACK_CASE["half"]
    case = world.add_body(name="black_case", pos=[*place(PACK_CASE), top + half[2]], quat=_yaw_quat(PACK_CASE["yaw"]))
    case.add_freejoint()
    case.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, mass=0.1, friction=[1.2, 0.01, 0.001],
                  rgba=[0.06, 0.06, 0.08, 1])
    case.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, half[2]], size=[half[0] * 0.45, half[1] * 0.85, 0.0006],
                  contype=0, conaffinity=0, mass=0.001, rgba=[0.25, 0.3, 0.55, 1])


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
        elif texture.name == "groundplane":
            # Plain dark grey floor, like the lab floor in the dataset images.
            texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_FLAT
            texture.rgb1 = texture.rgb2 = [0.17, 0.17, 0.17]
            texture.mark = mujoco.mjtMark.mjMARK_NONE
    spec.material("groundplane").reflectance = 0
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
    <texture type="2d" name="groundplane" builtin="flat" rgb1="0.17 0.17 0.17" rgb2="0.17 0.17 0.17" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def _g1d_robot_spec(movable_base=False):
    """The G1-D URDF as a spec, without its hands. Its wheel, column and torso joints are removed (fixed),
    unless `movable_base`."""
    urdf = ET.parse(G1D_URDF_PATH).getroot()
    for tag in ("joint", "link"):
        for element in list(urdf.findall(tag)):
            if "hand" in element.get("name"):
                urdf.remove(element)
    compiler = urdf.find("mujoco/compiler")
    compiler.set("meshdir", str(G1D_URDF_PATH.parent))
    compiler.set("fusestatic", "false")  # keep torso_link and head_link as bodies for the cameras
    robot = mujoco.MjSpec.from_string(ET.tostring(urdf, encoding="unicode"))
    if not movable_base:
        for name in G1D_FIXED_JOINTS:
            robot.delete(robot.joint(name))
    return robot


def _g1d_spec(movable_base=False):
    """Unitree's G1-D on the floor, with its base, column and torso held fixed and its three-finger hands removed
    (Dex1-style grippers are mounted instead). Returns the spec, the 29 body joint names in G1_29_JointIndex
    order (the 14 arm joints have the G1's names and indices, the others do not exist: None), and the x, z
    position of its torso_link. With `movable_base`, the base gets planar joints (x, y, yaw) and the wheel,
    column and torso joints are kept, to follow the Cyber Twin."""
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
    spec.attach(_g1d_robot_spec(movable_base), frame=spec.worldbody.add_frame(pos=[0, 0, -lowest]), prefix="")
    if movable_base:
        base = spec.body("AGV_link")
        base.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
        base.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
        base.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
        # These joints are set kinematically from the twin. A very large armature keeps the arms' motion from
        # nudging them within a physics step, so the arms behave as on the fixed base.
        for name in (*G1D_BASE_JOINTS, *G1D_FIXED_JOINTS):
            spec.joint(name).armature = 1e4

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


def build_model(robot="g1", movable_base=False):
    """G1 (fixed pelvis) or G1-D (fixed base, or following the Cyber Twin with `movable_base`) with Dex1-like
    grippers, head and wrist cameras, and a table scene."""
    if robot == "g1d":
        spec, body_joints, (torso_x, torso_z) = _g1d_spec(movable_base)
    else:
        spec, body_joints = _g1_spec()
        torso_x, torso_z = G1_TORSO_POS

    # Stereo camera in the face, 6 cm apart, looking down at the table.
    torso = spec.body("torso_link")
    for name, (pos, quat, fovy) in HEAD_CAMERAS.items():
        torso.add_camera(name=name, pos=pos, quat=quat, fovy=fovy)

    for side in GRIPPER_SIDES:
        _add_gripper(spec, side)
    _add_scene(spec, torso_x, torso_z)
    if robot == "g1d":
        # The G1-D URDF's collision meshes overlap at the wrist joints. The robot's own parts (grippers
        # included) do not collide with each other, but still collide with the table, objects and floor.
        _set_contact_group(spec.body("AGV_link"), contype=2, conaffinity=1)
        if movable_base:
            for name in G1D_BASE_BODIES:
                for geom in spec.body(name).geoms:
                    geom.contype = geom.conaffinity = 0

    # Softer lighting: the white table renders light grey as in the dataset images instead of saturating.
    spec.visual.headlight.diffuse = [SCENE_HEADLIGHT] * 3
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.specular = [0.1, 0.1, 0.1]
    for light in spec.lights:
        light.diffuse = [SCENE_LIGHT] * 3
        light.specular = [0.1, 0.1, 0.1]

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
        # G1-D following the Cyber Twin: base, wheel, column and torso joints set kinematically each step.
        follow = [n for n in (*G1D_BASE_JOINTS, *G1D_FIXED_JOINTS) if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0]
        self.follow_names = follow
        self.follow_qpos_adr = np.array([model.joint(n).qposadr[0] for n in follow], dtype=int)
        self.follow_dof_adr = np.array([model.joint(n).dofadr[0] for n in follow], dtype=int)
        self.follow_speed = np.array([G1D_FOLLOW_SPEED.get(n, np.inf) for n in follow])
        self.follow_target = np.zeros(len(follow))

        # Start with the arms at the task's start pose and the grippers open.
        arm_qpos = [model.joint(body_joints[i]).qposadr[0] for i in ARM_JOINT_INDICES]
        self.data.qpos[arm_qpos] = START_ARM_POSE
        for side in GRIPPER_SIDES:
            self.data.qpos[self.finger_qpos_adr[side]] = FINGER_TRAVEL
        mujoco.mj_forward(model, self.data)

        # Latest commands, written by the DDS callbacks and read by the physics loop.
        self.lock = threading.Lock()
        self.q_cmd = np.zeros(NUM_BODY_MOTORS)
        self.q_cmd[ARM_JOINT_INDICES] = START_ARM_POSE
        self.dq_cmd = np.zeros(NUM_BODY_MOTORS)
        self.kp = np.full(NUM_BODY_MOTORS, DEFAULT_KP)
        self.kd = np.full(NUM_BODY_MOTORS, DEFAULT_KD)
        self.tau_ff = np.zeros(NUM_BODY_MOTORS)
        self.gripper_q_cmd = dict.fromkeys(GRIPPER_SIDES, GRIPPER_Q_MAX)  # open
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

    def set_twin_state(self, robot: dict):
        """Target pose of the base, wheels, column and torso from the Cyber Twin's /api/robot state."""
        values = {"base_x": robot["x"], "base_y": robot["y"], "base_yaw": robot["yaw"], **robot["joints"]}
        with self.lock:
            self.follow_target[:] = [values.get(n, 0.0) for n in self.follow_names]

    def objects_status(self):
        """Where the black camera is, and whether it lies inside the box (for the status line)."""
        m, d = self.model, self.data
        cam = d.xpos[m.body("black_camera").id]
        box = d.xpos[m.body("box").id]
        rot = d.xmat[m.body("box").id].reshape(3, 3)
        local = rot.T @ (cam - box)
        inside = all(abs(local[:2]) < PACK_BOX["inner_half"]) and local[2] < PACK_BOX["wall_height"] + 0.03
        return f"camera at ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f}){' IN THE BOX' if inside else ''}"

    def base_pose(self):
        """x, y, yaw of the base and column height (0 if this robot does not follow the twin)."""
        q = dict(zip(self.follow_names, self.data.qpos[self.follow_qpos_adr]))
        return q.get("base_x", 0.0), q.get("base_y", 0.0), q.get("base_yaw", 0.0), q.get("LZ_mt_Joint", 0.0) + q.get("LZ_it_Joint", 0.0)

    def step(self):
        """Apply the motor torques and advance the physics by one timestep."""
        data = self.data
        if len(self.follow_names):
            # Kinematic: move towards the twin's pose at a limited speed, without dynamics.
            with self.lock:
                target = self.follow_target.copy()
            current = data.qpos[self.follow_qpos_adr]
            max_step = self.follow_speed * self.model.opt.timestep
            data.qpos[self.follow_qpos_adr] = current + np.clip(target - current, -max_step, max_step)
            data.qvel[self.follow_dof_adr] = 0.0
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


def camera_server(shared_qpos, port: int, fps: float, jpeg_quality: int, robot: str, movable_base: bool):
    """Child process: render the cameras from the latest joint positions and stream them like the robot does."""
    model, _ = build_model(robot, movable_base)
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


def twin_follower(robot, twin_url: str, stop_event: threading.Event, rate_hz: float = 20):
    """Thread: poll the Cyber Twin's state and make the base, column and torso follow it."""
    url = twin_url.rstrip("/") + "/api/robot"
    warned = False
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                robot.set_twin_state(json.load(response)["robot"])
            warned = False
        except (OSError, ValueError, KeyError) as exc:
            if not warned:
                print(f">>> Cyber Twin not reachable at {url} ({exc}); base holds its pose.", flush=True)
                warned = True
        time.sleep(1 / rate_hz)


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
    parser.add_argument(
        "--twin_url",
        type=str,
        default=None,
        help="G1-D only: URL of the DGS Cyber Twin (e.g. http://127.0.0.1:3000). The base, wheels, column and "
        "torso then follow the twin's state, so base and column commands allowed by its PEP move the robot.",
    )
    args = parser.parse_args()
    if args.twin_url and args.robot != "g1d":
        parser.error("--twin_url needs --robot g1d")

    model, body_joints = build_model(args.robot, movable_base=bool(args.twin_url))
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
            args=(shared_qpos, args.image_port, args.camera_fps, args.jpeg_quality, args.robot, bool(args.twin_url)),
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
    stop_event = threading.Event()
    if args.twin_url:
        threading.Thread(target=twin_follower, args=(robot, args.twin_url, stop_event), daemon=True).start()
        print(f">>> Base, column and torso follow the Cyber Twin at {args.twin_url}.", flush=True)
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
                    f"arm q: {np.round(arm_q, 2)} | grippers: {grippers} | {robot.objects_status()}"
                    + (" | base x {:.2f} y {:.2f} yaw {:.2f} column {:.3f}".format(*robot.base_pose()) if args.twin_url else ""),
                    flush=True,
                )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
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
