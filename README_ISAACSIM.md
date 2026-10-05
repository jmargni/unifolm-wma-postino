# Running Unitree's Isaac Sim simulator, the model and the client on one EC2 machine

This guide sets up a single Amazon EC2 machine (`g6e.2xlarge`) that runs all three
parts of the loop:

```
                         EC2 g6e.2xlarge (one L40S GPU, 48 GB)
┌───────────────────────────────────────────────────────────────────────────┐
│ unitree_sim_isaaclab         robot_client.py            real_eval_server  │
│ (Isaac Sim, G1 + Dex1)  ◄──► (the "postino")  ──HTTP──► (UnifoLM-WMA-0)   │
│   DDS: joint state/commands     127.0.0.1:8000             ~19 GB GPU     │
│   ZMQ: camera images                                                      │
└───────────────────────────────────────────────────────────────────────────┘
             ▲ SSH (and optionally WebRTC) from the laptop, to watch and control
```

- **Simulator:** [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab),
  Unitree's own simulator of the G1, built on NVIDIA Isaac Sim / Isaac Lab. It uses
  the same DDS topics as the real robot and streams cameras over ZMQ, so
  `robot_client.py` talks to it as to the real robot.
- **Model:** the same server as in [README_EC2.md](README_EC2.md).
- **Client:** `unitree_deploy/scripts/robot_client.py`, unchanged.

Because everything runs on one machine, the 8.4 MB image upload from the laptop
disappears from every model request.

> **Status of this guide.** The steps follow Unitree's and NVIDIA's documentation
> but have **not been run end to end yet**. Section 7 lists what must be checked
> on the first run. Two points are known unknowns: whether the simulator's
> camera stream has the layout the client expects, and whether its gripper DDS
> topic names match the client's.

> **Expectations.** The simulator's tasks (pick-and-place a cylinder or a red
> block, stack blocks) are not the task the model was trained on ("pack black
> camera into box"), and its images are rendered, not real. As found with the
> MuJoCo simulator, the model may not perform the task well. The purpose of
> this setup is a realistic robot simulation that receives and executes the
> commands, for the security tests.

---

## 1. Choose and launch the machine

### 1.1 Why `g6e.2xlarge`

| Resource | `g6e.2xlarge` | Needed |
| --- | --- | --- |
| GPU | 1× NVIDIA L40S, 48 GB | model ~19 GB + Isaac Sim (rendering and physics) |
| RAM | 64 GB | model checkpoint (16.7 GB, loaded into RAM first) + Isaac Sim (32 GB is its usual minimum) |
| vCPU | 8 | physics, cameras, model server and the client's control threads |

`g6e.xlarge` has the same GPU but only 4 vCPUs and 32 GB RAM: too tight for
both programs at once.

### 1.2 GPU quota

The quota **Running On-Demand G and VT instances** must be at least **8** (it
counts vCPUs) in the region you use. If you already raised it for the model
server, check the value; otherwise follow [README_EC2.md](README_EC2.md),
section 1.1.

### 1.3 Launch settings

AWS console → **EC2** → **Launch instance**:

| Setting | Value |
| --- | --- |
| Name | `unifolm-isaacsim` |
| Image (AMI) | **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)** |
| Instance type | **`g6e.2xlarge`** |
| Key pair | your key (e.g. the one used in README_EC2.md) |
| Security group | **SSH (TCP 22) from "My IP"** only. Optional WebRTC ports: section 6.2. |
| Storage | **300 GB gp3** |

Why 300 GB: Isaac Sim with its extension cache is tens of GB, plus three conda
environments, the 16.7 GB checkpoint, the CLIP weights and the simulator assets.

Ubuntu **22.04** is required by Isaac Sim (it needs a recent GLIBC).

Connect and check the GPU:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@EC2_IP
nvidia-smi        # must show "NVIDIA L40S" and a driver version of 535 or newer
```

---

## 2. Base tools (on EC2)

```bash
sudo apt update
sudo apt install -y git git-lfs tmux cmake build-essential libvulkan1 vulkan-tools
git lfs install

# Conda (Miniforge)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc
```

Isaac Sim renders with Vulkan even without a screen. Check that the NVIDIA
Vulkan driver is visible:

```bash
vulkaninfo --summary 2>/dev/null | grep -i "deviceName"     # must list the NVIDIA L40S
```

If it lists nothing, see Troubleshooting (section 9).

The machine runs **three separate conda environments**, one per program:

| Environment | Program | Python |
| --- | --- | --- |
| `unifolm-wma` | model server | 3.10 |
| `unitree_sim_env` | Isaac Sim simulator | 3.10 (Isaac Sim 4.5) |
| `unitree_deploy` | client | 3.10 |

---

## 3. Model server (on EC2)

Follow [README_EC2.md](README_EC2.md) **sections 3.2 to 4** on this machine:
clone the fork to `~/unifolm-world-model-action`, create `unifolm-wma`,
download the checkpoint, set the paths, start the server in tmux.

Skip the swap file of section 3.1: with 64 GB of RAM it is not needed.

Start the server first and wait for `>>> Inference server is ready ...`
before starting the simulator, so the two never load at the same time.

Check GPU memory:

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv     # about 19 GB of 46 GB used
```

---

## 4. Isaac Sim and Unitree's simulator (on EC2)

### 4.1 Get the simulator

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
cd unitree_sim_isaaclab
```

### 4.2 Install Isaac Sim 4.5 and Isaac Lab

Unitree provides a script that creates the `unitree_sim_env` environment and
installs Isaac Sim, Isaac Lab and the Unitree SDK:

```bash
bash auto_setup_env.sh 4.5 unitree_sim_env
```

This downloads several GB and takes a while.

If the script fails, do the same by hand following
`doc/isaacsim4.5_install.md` in the simulator repository. In summary it is:

```bash
conda create -y -n unitree_sim_env python=3.10
conda activate unitree_sim_env
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# Isaac Lab at the commit Unitree tested (see doc/isaacsim4.5_install.md), then:
#   cd IsaacLab && ./isaaclab.sh --install
# Unitree SDK, the simulator's requirements and its image server (teleimager):
#   git clone https://github.com/unitreerobotics/unitree_sdk2_python && pip install -e unitree_sdk2_python
#   pip install -r requirements.txt
#   cd teleimager && pip install -e .
```

Use the exact Isaac Lab commit from Unitree's document: newer Isaac Lab
versions may not work with the simulator.

### 4.3 Accept the NVIDIA licence

Isaac Sim asks to accept its licence (EULA) on first start. To accept it
without an interactive prompt, add this to `~/.bashrc`:

```bash
echo 'export OMNI_KIT_ACCEPT_EULA=YES' >> ~/.bashrc
source ~/.bashrc
```

### 4.4 Download the simulator's assets

```bash
cd ~/unitree_sim_isaaclab
. fetch_assets.sh
```

### 4.5 First start (test the simulator alone)

```bash
tmux new -s sim
conda activate unitree_sim_env
cd ~/unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
    --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
    --enable_dex1_dds --robot_type g129 --no_render
```

- `--no_render`: no window (the machine has no screen); the view can be
  streamed over WebRTC (section 6.2).
- `--enable_dex1_dds`: the Dex1 grippers listen on DDS, like the real ones.
- `--device cpu`: physics on the CPU, as in Unitree's examples. The GPU is used
  for rendering.

The **first start is slow** (Isaac Sim compiles its shaders, possibly 10+
minutes). Later starts are much faster.

Other tasks with the G1 and Dex1 grippers:

| Task | What happens |
| --- | --- |
| `Isaac-PickPlace-Cylinder-G129-Dex1-Joint` | pick a cylinder and place it |
| `Isaac-PickPlace-RedBlock-G129-Dex1-Joint` | pick a red block and place it |
| `Isaac-Stack-RgyBlock-G129-Dex1-Joint` | stack red, green and yellow blocks |
| `Isaac-Move-Cylinder-G129-Dex1-Wholebody` | whole-body task, the robot can move |

Leave tmux without stopping it: `Ctrl+B`, then `D`.

---

## 5. The client (on EC2)

The fork is already cloned in `~/unifolm-world-model-action` (section 3).
Build the client's environment with the repository's script
(see [README_REPRODUCE.md](README_REPRODUCE.md)):

```bash
cd ~/unifolm-world-model-action
bash unitree_deploy/setup_env.sh
```

It creates `unitree_deploy` with a CPU build of PyTorch, which is all the
client needs.

---

## 6. Run the full loop

### 6.1 Start order

Three tmux windows on EC2, started in this order:

```bash
# 1. Model server (if not already running)
tmux new -s model
conda activate unifolm-wma && cd ~/unifolm-world-model-action
bash scripts/run_real_eval_server.sh                 # wait for "Inference server is ready"

# 2. Simulator (section 4.5)
tmux new -s sim
conda activate unitree_sim_env && cd ~/unitree_sim_isaaclab
python sim_main.py --device cpu --enable_cameras \
    --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
    --enable_dex1_dds --robot_type g129 --no_render

# 3. Client
tmux new -s client
conda activate unitree_deploy && cd ~/unifolm-world-model-action/unitree_deploy/scripts
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py \
    --robot_type g1_dex1 --language_instruction "pick up the cylinder" --control_freq 15
```

No SSH tunnel is needed for the model: client and server are on the same
machine, and the client already uses `127.0.0.1:8000`.

`UNITREE_IMAGE_SERVER=127.0.0.1` makes the client read the simulator's camera
stream instead of the real robot's address.

The client's start pose (`INIT_POSE` in `robot_client.py`) is the start of
the "pack camera" training episode. In Isaac Sim's scenes it may touch the
table or objects: watch the first seconds (section 6.2).

### 6.2 Watching the robot from the laptop

**Option A: the camera stream (simplest).** Forward the simulator's image port
over SSH and show it with the viewer script from the fork:

```bash
# on the laptop, terminal 1
ssh -i ~/.ssh/id_ed25519 -N -o ServerAliveInterval=15 -L 5555:127.0.0.1:5555 ubuntu@EC2_IP

# on the laptop, terminal 2
conda activate unitree_deploy
python ~/projects/unifolm-wma-postino/unitree_deploy/scripts/view_camera_stream.py   # Esc to close
```

This only works if the stream has the format checked in section 7, item 1.

**Option B: the full Isaac Sim view (WebRTC).** Install NVIDIA's *Isaac Sim
WebRTC Streaming Client* on the laptop and connect it to `EC2_IP`. This needs
extra inbound rules in the security group, **from "My IP" only**: the ports
are listed in NVIDIA's Isaac Sim documentation on livestream clients (for Isaac
Sim 4.5: TCP 49100 and UDP 47998). WebRTC uses UDP, so it cannot go through
the SSH tunnel.

### 6.3 Stopping

Stop in reverse order: client (`Ctrl+C`), then simulator, then model server.
Then **stop the instance** in the EC2 console (see [README_EC2.md](README_EC2.md),
section 7): a `g6e.2xlarge` bills for every running hour.

---

## 7. Checks on the first run

These are the points this guide could not verify in advance.

1. **Camera stream format.** The client expects, on ZMQ port 5555, **one JPEG
   per message** with the cameras side by side: the head stereo pair as the
   first 1280×480 pixels, then the wrist cameras. It gives the model the right
   half of the head pair (`cam_right_high`). Check:

   ```bash
   conda activate unitree_deploy
   python -c "
   import zmq, cv2, numpy as np
   s = zmq.Context().socket(zmq.SUB); s.connect('tcp://127.0.0.1:5555'); s.setsockopt_string(zmq.SUBSCRIBE, '')
   s.setsockopt(zmq.RCVTIMEO, 10000)
   m = s.recv_multipart(); print('parts:', len(m))
   img = cv2.imdecode(np.frombuffer(m[-1], np.uint8), cv2.IMREAD_COLOR); print('image:', None if img is None else img.shape)
   cv2.imwrite('isaac_frame.jpg', img)"
   ```

   Expected: `parts: 1` and a width of at least 1280 with height 480. If the
   stream uses another port, layout or message format (Unitree's newer image
   server, `teleimager`, may), the client cannot read it as is: a small
   adapter that re-publishes the frames in the expected layout is needed.

2. **Gripper topics.** The client sends gripper commands on
   `rt/dex1/left/cmd` and `rt/dex1/right/cmd` and reads
   `rt/dex1/left/state` / `rt/dex1/right/state`. If the client stays on
   `[Dex1_Gripper_Controller] Waiting to subscribe dds...`, the simulator uses
   other names: compare with the Dex1 DDS code in `unitree_sim_isaaclab`.

3. **Arm state.** If the client stays on
   `[G1_29_ArmController] Waiting to subscribe dds...`, the simulator is not
   publishing `rt/lowstate` (check it started with `--robot_type g129`).

4. **Memory.** With all three running:

   ```bash
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv
   free -h
   ```

   GPU below ~44 GB and RAM without heavy swap use are fine.

5. **What the model sees.** The model's imagined videos are saved in
   `~/results/unitree_g1_pack_camera/testing/videos/`; the first frame of each
   is the image it received. Check the colours and that the robot's grippers
   are in view.

---

## 8. Costs

- A `g6e.2xlarge` bills by the hour while **running**; stopped, only the
  300 GB disk is billed. Check current prices for your region on the EC2
  On-Demand pricing page.
- Stop the instance whenever you are not testing, and set a budget alert
  (AWS console → Billing → Budgets).
- A stopped instance gets a **new public IP** when started again: update
  `EC2_IP` in your commands.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Launch fails: "vCPU limit" | Quota below 8. | Section 1.2. |
| `vulkaninfo` lists no NVIDIA device; Isaac Sim fails with Vulkan / `VK_ERROR` messages | The NVIDIA Vulkan driver is not installed or not registered. | Check `ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/` contains an `nvidia_icd.json`; if not, install the NVIDIA driver's GL/Vulkan package matching `nvidia-smi`'s driver version (e.g. `sudo apt install libnvidia-gl-535`), then reboot. |
| `GLIBCXX_3.4.30 not found` | Ubuntu older than 22.04, or conda's older libstdc++ is used. | Use the Ubuntu 22.04 AMI; inside the env, `conda install -c conda-forge libstdcxx-ng`. |
| First start of the simulator seems stuck | Shader compilation. | Wait (10+ minutes the first time); `nvidia-smi` shows GPU activity meanwhile. |
| Model server: `CUDA out of memory` after the simulator started | Simulator started first, or both loaded at once. | Stop both; start the model server first, then the simulator. |
| `Killed` while loading, or very slow machine | Out of RAM. | `free -h`; close other programs; add the swap file of README_EC2.md section 3.1. |
| Client waits forever on `Waiting to subscribe dds...` | Simulator not running, or different topic names. | Section 7, items 2 and 3. |
| Client shows black or garbled images, or crashes in `ImageClient` | Camera stream in another format. | Section 7, item 1. |
| Two simulators answer / joints jump | Another simulator or robot on the same network uses the same DDS topics. | Run only one simulator; on EC2 this cannot come from your laptop. |

---

## Sources

- Unitree simulator: <https://github.com/unitreerobotics/unitree_sim_isaaclab>
  (install documents in its `doc/` folder)
- NVIDIA Isaac Sim documentation: requirements, livestream / WebRTC clients
- Model server set-up: [README_EC2.md](README_EC2.md)
- Client environment: [README_REPRODUCE.md](README_REPRODUCE.md)
