# Running `robot_client.py` locally without a robot or a GPU

This folder contains the robot client and programs that stand in for the real
robot and the real model, so the whole control loop can be run on a laptop.
For the robot there are two choices: the lightweight mock `mock_g1_robot.py`
(sections 1-8) or the physics simulator `sim_g1_robot.py`, which also provides
camera images (section 9).

| File | What it is | Replaces |
| --- | --- | --- |
| `robot_client.py` | The client (the "postino"). Reads the robot state, asks the model server for actions, sends them to the robot. | — (Unitree's client, with a colour fix, the training start pose and the optional `--pep_url`) |
| `mock_g1_robot.py` | A fake Unitree G1 with two Dex1 grippers, shown in a MuJoCo window. | The real G1 |
| `sim_g1_robot.py` | A physics simulation of the G1 with grippers, cameras and a table scene (see section 9). | The real G1 and its cameras |
| `mock_policy_server.py` | A fake model server on `http://127.0.0.1:8000`. | `scripts/evaluation/real_eval_server.py` (needs a GPU and the 16.7 GB checkpoint) |
| `view_camera_stream.py` | Shows the simulator's camera stream in a window (Esc to close). | — |

```
                 joint state (DDS)                 observation (HTTP POST)
mock_g1_robot.py ─────────────────► robot_client.py ─────────────────────► mock_policy_server.py
                 ◄─────────────────                 ◄─────────────────────
                 joint commands (DDS)               16 future actions
```

## Quick start

Three ready-to-run setups. All run on the laptop except the real model (EC2).

| Setup | Robot | Model | PEP |
| --- | --- | --- | --- |
| **A** | simulator (G1) | mock | — |
| **B** | simulator (G1-D) | mock | Cyber Twin |
| **C** | simulator (G1-D) | real model on EC2 | Cyber Twin |

In **every laptop terminal**, first run:

```bash
conda activate unitree_deploy
cd ~/projects/unifolm-wma-postino/unitree_deploy/scripts
```

The mock model and the SSH tunnel to EC2 both use port 8000: run **one or
the other**, never both.

### A. Simulator + mock model (3 terminals)

```bash
# Terminal 1 - simulator (a MuJoCo window opens with the robot)
python sim_g1_robot.py

# Terminal 2 - mock model
python mock_policy_server.py

# Terminal 3 - client
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15
```

After a few seconds the client prints `All Device Connect Success`, then
`>>> Exec => step N action: [...]` lines, and the arms move slowly in the
MuJoCo window. The mock model answers instantly with a gentle wave, so the
motion is continuous.

### B. G1-D simulator + mock model + Cyber Twin PEP (4 terminals)

The simulator runs the **G1-D**, the twin's robot (wheeled base and lifting
column held fixed, same arms as the G1, Dex1-like grippers), and the client
asks the twin's PEP before every move.

```bash
# Terminal 1 - simulator with the G1-D
python sim_g1_robot.py --robot g1d

# Terminal 2 - mock model
python mock_policy_server.py

# Terminal 3 - the twin (PEP, and consoles on :3000 and :4000)
cd ~/projects/DGS-CyberTwin-G1D
python3 server.py

# Terminal 4 - client, with every chunk validated by the PEP
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15 --pep_url http://127.0.0.1:3000
```

Start the twin before the client: without it the robot never moves.
Then arm the robot in the browser (see *Using the PEP* below).

### C. G1-D simulator + real model on EC2 + Cyber Twin PEP

Same as B, with the real model in place of the mock. Server set-up, first
start and costs are in [README_EC2.md](../../README_EC2.md).

**1. Start the model server on EC2.** Start the instance in the AWS console
and copy its public IP (it changes at every start; below: `EC2_IP`). Then:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@EC2_IP
tmux new -s model                 # or: tmux attach -t model, if it is already running
conda activate unifolm-wma
cd ~/unifolm-world-model-action
bash scripts/run_real_eval_server.sh
```

Wait for `>>> Inference server is ready ...` (a few minutes), then leave tmux
with `Ctrl+B`, then `D`, and close this SSH session.

**2. On the laptop, four terminals:**

```bash
# Terminal 1 - simulator with the G1-D
python sim_g1_robot.py --robot g1d

# Terminal 2 - SSH tunnel to the model (instead of the mock model); leave it open
ssh -i ~/.ssh/id_ed25519 -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -L 8000:127.0.0.1:8000 ubuntu@EC2_IP

# Terminal 3 - the twin
cd ~/projects/DGS-CyberTwin-G1D
python3 server.py

# Terminal 4 - client
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15 --pep_url http://127.0.0.1:3000 \
    --language_instruction "pack black camera into box"
```

Check the tunnel before starting the client (in any terminal):

```bash
curl -s -m 10 -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs     # 200 = model reachable
```

Then arm the robot in the browser (see *Using the PEP* below).

With the real model each request takes about 30 s (upload plus the model's
time on the GPU): the robot moves for about 1 s, then holds still until the
next answer. The model's "imagined" videos are saved on EC2 in
`~/results/unitree_g1_pack_camera/testing/videos/`.

### Using the PEP (setups B and C)

1. Open **http://127.0.0.1:3000**. The client prints
   `PEP start pose: DENY G1D-107 ... arm richiesto` every 3 s: the robot waits.
2. Click **Arma robot**. The client prints `PEP start pose: ALLOW`, the robot
   moves to the start pose, and then `PEP chunk: ALLOW ...` before every chunk.
   The 3D robot on the page follows the arms.
3. Open **http://127.0.0.1:4000** to see every decision in the event log.

| In the browser | Effect |
| --- | --- |
| **E-STOP** (3000 or 4000) | robot stops within one step; every chunk denied |
| **Reset** in Defense (4000), then **Arma robot** (3000) | loop resumes |
| **Disarma** (3000) | robot stops; chunks denied until armed again |
| Enforce → monitor (4000) | violations logged as `MONITOR` but executed |
| Twin closed | robot holds; nothing moves until the twin is back and armed |

How the PEP check works: section 10.

### Stopping

- `Ctrl+C` in each laptop terminal: client first, then the others.
- Setup C: **stop the EC2 instance** in the AWS console when you are done
  (it bills while running). The model server in tmux stops with it.

### Optional: see the simulated cameras

In another laptop terminal (same `conda activate` and `cd`):

```bash
python view_camera_stream.py      # Esc to close
```

### Common messages

| Client prints | Meaning |
| --- | --- |
| `PEP start pose: DENY G1D-107 ... arm richiesto` | normal: arm the robot on :3000 |
| `PEP start pose: UNREACHABLE ...` | the twin is not running, or `--pep_url` is wrong |
| `[G1_29_ArmController] Waiting to subscribe dds...` | the simulator is not running |
| `An error occurred: ... Connection refused` (repeated) | no model on port 8000: start the mock or the tunnel |
| `DEBUG:urllib3...Resetting dropped connection` | harmless: the twin closes each connection |

## 1. Start everything

Open three terminals. In each one:

```bash
conda activate unitree_deploy
cd ~/pyspace/unifolm-world-model-action/unitree_deploy/scripts
```

Then start the programs **in this order**, one per terminal:

```bash
# Terminal 1 - the fake robot (a MuJoCo window opens)
python mock_g1_robot.py

# Terminal 2 - the fake model server
python mock_policy_server.py

# Terminal 3 - the client
python robot_client.py
```

Stop each program with `Ctrl+C`. Stop the client first: it then drives the arms
back to the zero pose before exiting.

## 2. What you should see

**Terminal 1 (robot)** prints a status line every 2 seconds. The message
counters stay at 0 until the client connects, then grow:

```
>>> Mock G1 is publishing state on DDS. Start robot_client.py now. Ctrl+C to stop.
lowcmd msgs: 749 | gripper cmd msgs: 1220 | arm q: [ 0.04  0.01 ...] | grippers: [5.4, 5.4]
```

**Terminal 2 (server)** prints one line per request from the client:

```
>>> Mock policy server is ready on http://127.0.0.1:8000 ...
[1] instruction: 'Pack black camera into box' | state (2, 16) | images (2, 3, 480, 640) (all black) -> actions (16, 16)
```

**Terminal 3 (client)** connects, then prints every action it executes:

```
[SUCCESS] 🚀 All Device Connect Success!!!.✅
>>> Exec => step 0 action: [ 0.06069  0.01567 ... 5.4  5.4 ]
>>> Exec => step 1 action: [ 0.06854  0.01974 ... 5.40087  5.3939 ]
```

In the MuJoCo window the two arms sway slowly.

Each action is 16 numbers: 7 left-arm joints, 7 right-arm joints (radians),
then the left and right gripper openings (0 = closed, about 5.4 = open).

## 3. Sending commands with `robot_client.py`

The client does not take joint commands on the command line. What you send is
a **language instruction**; the model server turns it (plus the camera image and
joint state) into actions, and the client executes them. The other options
control how those actions are executed.

| Option | Default | Meaning |
| --- | --- | --- |
| `--language_instruction` | `"Pack black camera into box"` | The text command sent to the model server. |
| `--robot_type` | `g1_dex1` | Robot embodiment. Only `g1_dex1` works with the mock robot and the simulator. |
| `--control_freq` | `30` | Actions executed per second (Hz). |
| `--action_horizon` | `16` | How many of the predicted future actions to keep (max 16). |
| `--exe_steps` | `16` | How many of those to execute before asking the server again. Must be ≤ `--action_horizon`. |
| `--observation_horizon` | `2` | How many recent frames/states are sent to the server. |
| `--num_rollouts_planned` | `10` | Number of episodes. Has no visible effect: the first episode runs until `Ctrl+C`. |
| `--output_dir` | `./results` | Folder for results. Only empty `episode_NNN` folders are created. |

### Examples

Send a different instruction:

```bash
python robot_client.py --language_instruction "pick up the red cup"
```

The same command as in the main project README (15 Hz control):

```bash
python robot_client.py --robot_type "g1_dex1" --action_horizon 16 --exe_steps 16 \
    --observation_horizon 2 --language_instruction "pack black camera into box" \
    --output_dir ./results --control_freq 15
```

Ask the server for new actions twice as often (execute 8 of the 16 predicted
steps, then re-plan):

```bash
python robot_client.py --language_instruction "wave both arms" --exe_steps 8
```

Slow the robot down to 10 actions per second:

```bash
python robot_client.py --control_freq 10
```

See all options:

```bash
python robot_client.py --help
```

> **With the mock server the instruction changes nothing in the motion.** The
> mock only prints the instruction it received and always returns the same sine
> wave. The instruction only has an effect with the real model server.

## 4. Options of the mock programs

Make the mock motion bigger or faster:

```bash
# +-0.3 rad instead of +-0.15, one full wave every 60 steps instead of 120
python mock_policy_server.py --amplitude 0.3 --period_steps 60
```

Run the server on another port (the client's address is fixed in
`robot_client.py`, variables `HOST` and `PORT`, so change it there too):

```bash
python mock_policy_server.py --port 8001
```

Run the fake robot without the MuJoCo window (e.g. over SSH):

```bash
python mock_g1_robot.py --headless
```

Bind the fake robot's DDS traffic to one network interface:

```bash
python mock_g1_robot.py --network_interface wlo1
```

## 5. Sending a request to the server by hand

To see exactly what the client sends and receives, call the server directly.
With the mock server running:

```bash
python - <<'EOF'
import requests

payload = {
    "language_instruction": "pick up the red cup",
    "observation.state": [[0.0] * 16] * 2,                          # 2 frames x 16 joints
    "observation.images.top": [[[[0] * 64] * 48] * 3] * 2,          # 2 frames x 3 x H x W
    "action": [[0.0] * 16] * 16,                                    # always zeros
}
reply = requests.post("http://127.0.0.1:8000/predict_action", json=payload).json()
print(reply["result"], len(reply["action"]), "actions of", len(reply["action"][0]), "values")
print(reply["action"][0])
EOF
```

Expected output: `ok 16 actions of 16 values`, followed by the first action.

## 6. Limits of the mock setup

- **No physics.** The fake robot moves each joint straight to the commanded
  position: no gravity, no contacts, no motor behaviour. `sim_g1_robot.py`
  (section 9) has physics.
- **No camera.** The client sends all-black images. By default the client looks
  for the real robot's camera server (`192.168.123.164`); the environment
  variable `UNITREE_IMAGE_SERVER` changes that address. `sim_g1_robot.py`
  provides simulated cameras.
- **No intelligence.** The actions are a fixed sine wave; the instruction and
  the images are ignored.
- **DDS is broadcast on the local network.** Do not run the fake robot on a
  network where a real Unitree robot is connected: both would answer on the
  same topics.

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Client loops on `[G1_29_ArmController] Waiting to subscribe dds...` | No robot state is arriving. | Start `mock_g1_robot.py` (or `sim_g1_robot.py`) first. If it is running, check that both use the same network interface (`--network_interface`). |
| Client prints `An error occurred: ... Connection refused` repeatedly | The model server is not running. | Start `mock_policy_server.py`. The client retries on its own. |
| `OSError: [Errno 98] Address already in use` in the server | Port 8000 is taken (often an old server). | Stop the other program or use `--port`. |
| MuJoCo window does not open / GLFW error | No display available. | Use `python mock_g1_robot.py --headless` (or `sim_g1_robot.py --headless`). |
| `ModuleNotFoundError: unitree_deploy` or `unitree_sdk2py` | Wrong Python environment. | `conda activate unitree_deploy`. |
| `ZMQError: Address already in use (addr='tcp://*:5555')` in `sim_g1_robot.py` | An old simulator is still streaming. | Stop it (`pkill -f sim_g1_robot`), or use `--image_port` (the client always connects to 5555). |
| The model server says `all black` with `sim_g1_robot.py` | The client is not reading the simulator's cameras. | Start the client with `UNITREE_IMAGE_SERVER=127.0.0.1`. |

## 8. Switching to the real model or the real robot

- **Real model:** start `scripts/evaluation/real_eval_server.py` on the GPU
  machine instead of `mock_policy_server.py`, and set `HOST`/`PORT` in
  `robot_client.py` to its address (or forward port 8000 with SSH).
- **Real robot:** do not start `mock_g1_robot.py` or `sim_g1_robot.py`, and
  do not set `UNITREE_IMAGE_SERVER`; connect the PC to the
  robot's network and follow the main project README.

## 9. Physics simulator instead of the mock robot

`sim_g1_robot.py` replaces `mock_g1_robot.py` (start one or the other, not
both). It speaks the same DDS topics, but simulates the robot with MuJoCo
physics instead of copying the commanded positions:

- **Motors:** each joint gets the torque the real motor driver would apply,
  `kp * (q_cmd - q) + kd * (dq_cmd - dq) + tau`, using the gains and the
  gravity-compensation torque the client sends, clipped to the motor limits.
  Gravity, contacts and friction act on everything.
- **Fixed pelvis:** the real G1 keeps its balance with Unitree's own walking
  controller, which is not simulated, so the pelvis is fixed in place.
- **Grippers:** simplified two-finger grippers. The Dex1 opening (0 = closed,
  5.45 = open) maps to 0-4 cm of travel per finger. They stop when they close
  on an object.
- **Cameras:** a stereo head camera and one camera per wrist, streamed on
  `tcp://*:5555` in the same format as the robot's image server. The model
  receives the right head image.
- **Scene:** a table with a black "camera" (on the robot's left) and an open
  box (on its right), matching the default instruction "Pack black camera into
  box".

- **Robot:** `--robot g1` (default) is the humanoid G1 with its pelvis fixed.
  `--robot g1d` is the **G1-D** of the DGS Cyber Twin (Unitree's
  `g1_d_description`, copied in `unitree_deploy/robot_devices/assets/g1d`):
  wheeled base, lifting column (retracted) and torso held fixed, the same
  14-joint arms as the G1 (identical meshes, joint positions and limits) and
  the same Dex1-like grippers instead of its three-finger hands. Its arm motors
  use the G1's indices 15-28 on DDS, so the client works unchanged; the other
  indices are reported at 0. This numbering is the simulator's choice: how a
  real G1-D numbers its motors over DDS has not been checked.

Start it like the mock, and tell the client where the cameras are:

```bash
# Terminal 1 - the simulated robot (a MuJoCo window opens)
python sim_g1_robot.py

# Terminal 2 - the fake model server
python mock_policy_server.py

# Terminal 3 - the client, reading the simulator's cameras
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py
```

Without `UNITREE_IMAGE_SERVER` the client looks for the real robot's camera
server (`192.168.123.164`) and sends black images, as with the mock.

Terminal 1 prints a status line every 2 seconds. `real-time x1.00` means the
simulation keeps up with the clock. Lower values mean the laptop is too slow
and the robot moves in slow motion.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--headless` | off | No MuJoCo window. |
| `--no_cameras` | off | Do not render or stream the cameras (the client then gets black images). |
| `--state_freq` | `100` | Joint-state messages per second. Higher values slow the simulation below real time. |
| `--camera_fps` | `30` | Target frame rate of the camera stream. An integrated GPU reaches about 20-25. |
| `--jpeg_quality` | `80` | JPEG quality of the camera stream. |
| `--image_port` | `5555` | Port of the camera stream. |
| `--network_interface` | auto | DDS network interface, as for the mock. |

Limits:

- **The images do not look like the lab.** The model was trained on real
  camera images, so do not expect it to solve the task in this scene. The
  simulator checks the plumbing: the images reach the model, actions stay in
  range, the arms move smoothly, and timing holds.
- **No walking or balance.** The pelvis is fixed.
- **Approximate grippers and objects.** Shapes, masses and friction are
  guesses, not measured from the real Dex1 or the lab objects.
- **DDS is broadcast on the local network**, as with the mock: do not run the
  simulator on a network where a real Unitree robot is connected.
- **Objects cannot be dragged with the mouse** in the MuJoCo window, and
  camera images are rendered without shadows (to keep the frame rate up).

## 10. Validating every chunk with the Cyber Twin's PEP

With `--pep_url`, the client asks the policy enforcement point (PEP) of the
DGS G1-D Cyber Twin for permission before moving:

```
model ──16 actions──► robot_client.py ──"trajectory"──► twin PEP (:3000) ──ALLOW──► execute in the simulator
                                                              │
                                                              └─► audit log / SIEM, 3D console mirrors the pose
```

- **Start pose:** the robot holds still until the PEP allows it, for example
  until a person presses **Arma robot** in the twin's console (`:3000`).
- **Each chunk:** the 16 actions go to the PEP as one `trajectory` command. It
  checks identity, armed / E-stop state, replay, the G1-D URDF joint limits and
  a lab joint-speed limit of 3 rad/s (also from the measured pose to the first
  action). Denied chunks are not executed: the robot holds and the client asks
  the model again.
- **During a chunk:** before every step the client checks the twin is still
  armed. An E-stop or disarm from the consoles stops the robot within one step.
- **Fail closed:** if the twin cannot be reached, nothing moves.
- **Monitor mode:** if enforcement is switched off in the Defense console
  (`:4000`), violations are recorded as `MONITOR` and the chunk is executed.

The client never arms the twin and never clears an E-stop.

Start the twin (its own folder, plain Python), then the client with the flag:

```bash
# Terminal 1 - the twin and its consoles (:3000 robot, :4000 Red Team / Defense)
cd ~/projects/DGS-CyberTwin-G1D && python3 server.py

# Terminals 2 and 3 - simulator and model server as in sections 1 or 9

# Terminal 4 - the client
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py --control_freq 15 \
    --pep_url http://127.0.0.1:3000
```

The client prints one line per decision:

```
>>> PEP start pose: DENY G1D-107 Sequenza non valida: arm richiesto prima del movimento
>>> PEP chunk: ALLOW G1D-200 Eseguito: trajectory
>>> PEP chunk: DENY G1D-102 Traiettoria: left_shoulder_roll_joint oltre 3.0 rad/s al punto 0
>>> PEP: twin no longer armed (E-stop, disarm or unreachable): chunk aborted
```

Requirements on the twin side: the `trajectory` command in `engine.py` and the
`/api/robot` endpoint in `server.py` (added for this bridge). Only
`--robot_type g1_dex1` is supported: its 14 arm joints map by name to the
G1-D's arms; the two Dex1 grippers map to the G1-D hands' open / closed state.
The bridge code is in `unitree_deploy/unitree_deploy/pep_bridge.py`.
