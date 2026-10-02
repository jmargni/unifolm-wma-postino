# Running `robot_client.py` locally without a robot or a GPU

This folder contains the robot client and two mock programs that stand in for
the real robot and the real model, so the whole control loop can be run on a
laptop.

| File | What it is | Replaces |
| --- | --- | --- |
| `robot_client.py` | The client. Reads the robot state, asks the model server for actions, sends them to the robot. | — (this is the real one, unmodified) |
| `mock_g1_robot.py` | A fake Unitree G1 with two Dex1 grippers, shown in a MuJoCo window. | The real G1 |
| `mock_policy_server.py` | A fake model server on `http://127.0.0.1:8000`. | `scripts/evaluation/real_eval_server.py` (needs a GPU and the 16.7 GB checkpoint) |

```
                 joint state (DDS)                 observation (HTTP POST)
mock_g1_robot.py ─────────────────► robot_client.py ─────────────────────► mock_policy_server.py
                 ◄─────────────────                 ◄─────────────────────
                 joint commands (DDS)               16 future actions
```

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
| `--robot_type` | `g1_dex1` | Robot embodiment. Only `g1_dex1` works with the mock robot. |
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
  position: no gravity, no contacts, no motor behaviour.
- **No camera.** The client sends all-black images. The camera address is fixed
  to the real robot's IP (`192.168.123.164`) in
  `unitree_deploy/robot_devices/cameras/imageclient.py`.
- **No intelligence.** The actions are a fixed sine wave; the instruction and
  the images are ignored.
- **DDS is broadcast on the local network.** Do not run the fake robot on a
  network where a real Unitree robot is connected: both would answer on the
  same topics.

## 7. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Client loops on `[G1_29_ArmController] Waiting to subscribe dds...` | No robot state is arriving. | Start `mock_g1_robot.py` first. If it is running, check that both use the same network interface (`--network_interface`). |
| Client prints `An error occurred: ... Connection refused` repeatedly | The model server is not running. | Start `mock_policy_server.py`. The client retries on its own. |
| `OSError: [Errno 98] Address already in use` in the server | Port 8000 is taken (often an old server). | Stop the other program or use `--port`. |
| MuJoCo window does not open / GLFW error | No display available. | Use `python mock_g1_robot.py --headless`. |
| `ModuleNotFoundError: unitree_deploy` or `unitree_sdk2py` | Wrong Python environment. | `conda activate unitree_deploy`. |

## 8. Switching to the real model or the real robot

- **Real model:** start `scripts/evaluation/real_eval_server.py` on the GPU
  machine instead of `mock_policy_server.py`, and set `HOST`/`PORT` in
  `robot_client.py` to its address (or forward port 8000 with SSH).
- **Real robot:** do not start `mock_g1_robot.py`; connect the PC to the
  robot's network and follow the main project README.
