# Running the real model on an Amazon EC2 GPU server

This guide sets up the **UnifoLM-WMA-0-Dual** model server on an Amazon EC2
machine with an NVIDIA GPU. The laptop keeps running the client
(`robot_client.py`) and the simulator; only the model moves to the cloud.

```
        Laptop                                         EC2 (GPU)
┌───────────────────────────┐   SSH tunnel    ┌──────────────────────────┐
│ sim_g1_robot.py           │   port 8000     │ real_eval_server.py      │
│ robot_client.py ──────────┼────────────────►│   UnifoLM-WMA-0-Dual     │
│   (127.0.0.1:8000)        │◄────────────────┼── 16 actions             │
└───────────────────────────┘                 └──────────────────────────┘
```

The client and the server talk exactly as before. Nothing in the client changes:
the SSH tunnel makes the EC2 server appear at `127.0.0.1:8000` on the laptop.

**Before you start:** check with your organisation that this project's data may
be processed on AWS, and who approves the cost.

---

## 1. One-time AWS preparation

### 1.1 Ask for GPU quota

New AWS accounts are not allowed to start GPU machines (the quota is 0).

1. AWS console → **Service Quotas** → **Amazon Elastic Compute Cloud (Amazon EC2)**.
2. Search **Running On-Demand G and VT instances**.
3. **Request increase** to at least **8** (the quota counts vCPUs; one `g5.2xlarge` uses 8).
4. Do this in the region you will use (see 2.1). Approval can take a day or more.

### 1.2 Create an SSH key pair

AWS console → **EC2** → **Key Pairs** → **Create key pair**:

- Name: `unifolm-ec2`, type RSA, format `.pem`.
- The file downloads once. Move it and protect it:

```bash
mv ~/Downloads/unifolm-ec2.pem ~/.ssh/
chmod 400 ~/.ssh/unifolm-ec2.pem
```

---

## 2. Launch the machine

AWS console → **EC2** → **Launch instance**.

### 2.1 Settings

| Setting | Value | Why |
| --- | --- | --- |
| Region (top right) | The closest one with GPU machines, e.g. **Europe (Frankfurt) `eu-central-1`** | Lower delay per request. |
| Name | `unifolm-model-server` | |
| Image (AMI) | Search **"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"** | NVIDIA driver already installed. |
| Instance type | **`g5.2xlarge`** (NVIDIA A10G, 24 GB GPU, 32 GB RAM) | See the table below. |
| Key pair | `unifolm-ec2` | |
| Network → Security group | Allow **SSH (port 22) from "My IP"** only. Nothing else. | The model server has no password; it is reached only through SSH. |
| Storage | **150 GB gp3** | Checkpoint 17 GB + CLIP weights 4 GB + environment ~15 GB + saved videos. |

Instance type: two memories matter.

- **GPU memory** holds the model while it runs. 24 GB is expected to be enough
  but has not been tested with this model; if it is not, the server fails with
  CUDA "out of memory".
- **Normal memory (RAM)**: the server first reads the whole 16.7 GB checkpoint
  into RAM, then builds the model there, then moves it to the GPU. While
  loading, it needs more RAM than the checkpoint size.

| Type | GPU | GPU memory | RAM | Note |
| --- | --- | --- | --- | --- |
| `g5.xlarge` | A10G | 24 GB | 16 GB | **Not recommended:** too little RAM to load the checkpoint; the server is likely to be killed while starting. |
| `g5.2xlarge` | A10G | 24 GB | 32 GB | **Chosen.** Same GPU, enough RAM with the swap file of section 3.1. |
| `g5.4xlarge` | A10G | 24 GB | 64 GB | Only if `g5.2xlarge` still runs out of RAM. |
| `g6e.xlarge` | L40S | 48 GB | 32 GB | Fallback if 24 GB of GPU memory is not enough. |

Prices change and differ per region: check the **EC2 On-Demand pricing** page
before launching.

Click **Launch instance**. When it shows **Running**, copy its **Public IPv4
address** (below: `EC2_IP`).

### 2.2 Connect

```bash
ssh -i ~/.ssh/unifolm-ec2.pem ubuntu@EC2_IP
```

Check the GPU:

```bash
nvidia-smi        # must list the GPU ("NVIDIA A10G") and a driver version
```

---

## 3. Install the model (on EC2)

All commands in this section run **on the EC2 machine**.

### 3.1 Tools

```bash
sudo apt update && sudo apt install -y git tmux

# Conda (Miniforge)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
$HOME/miniforge3/bin/conda init bash
source ~/.bashrc

# Swap file: extra (slow) memory on disk, so loading the checkpoint
# cannot run out of RAM. Survives reboots.
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h           # "Swap:" now shows 32G
```

### 3.2 Code

```bash
git clone --recurse-submodules https://github.com/jmargni/unifolm-wma-postino.git ~/unifolm-world-model-action
cd ~/unifolm-world-model-action
```

If the repository is private, GitHub asks for a username and password: use
your GitHub username and a **personal access token** (GitHub → Settings →
Developer settings → Personal access tokens, read-only access to this
repository) as the password.

`--recurse-submodules` is needed here: the model uses `external/dlimp`.

### 3.3 Environment

This is the model's environment (`unifolm-wma`), not the `unitree_deploy`
one used on the laptop.

```bash
conda create -y -n unifolm-wma python=3.10
conda activate unifolm-wma

conda install -y -c conda-forge pinocchio=3.2.0 ffmpeg=7.1.1

pip install -e .
pip install -e external/dlimp
```

`pip install -e .` installs PyTorch 2.3.1 with its own CUDA 12.1 libraries and
xformers 0.0.27; only the NVIDIA driver from the AMI is needed. It takes
several minutes.

Check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA A10G
```

### 3.4 Model checkpoint (16.7 GB)

```bash
mkdir -p ~/checkpoints
python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('unitreerobotics/UnifoLM-WMA-0-Dual', 'unifolm_wma_dual.ckpt', local_dir='$HOME/checkpoints'))
"
ls -lh ~/checkpoints/unifolm_wma_dual.ckpt     # about 16.7 GB
```

On the first start the server also downloads the CLIP model it uses
(`ViT-H-14`, about 4 GB) from Hugging Face. That happens automatically.

### 3.5 Configure the server

Two files contain placeholder paths. Replace them:

```bash
cd ~/unifolm-world-model-action

# 1. Normalisation data for the "pack camera" task (ships with the repository)
sed -i "s#data_dir: '/path/to/[^']*'#data_dir: '$HOME/unifolm-world-model-action/examples/world_model_interaction_prompts'#" \
    configs/inference/world_model_decision_making.yaml

# 2. Checkpoint and results folder
sed -i "s#^ckpt=.*#ckpt=$HOME/checkpoints/unifolm_wma_dual.ckpt#" scripts/run_real_eval_server.sh
sed -i "s#^res_dir=.*#res_dir=$HOME/results#" scripts/run_real_eval_server.sh
```

Check:

```bash
grep -n "data_dir:" configs/inference/world_model_decision_making.yaml
grep -n -E "^(ckpt|res_dir)=" scripts/run_real_eval_server.sh
```

The `dataset_and_weights` entry in the config is already set to
`unitree_g1_pack_camera`, which matches the G1 client. Leave it.

---

## 4. Start the server (on EC2)

Run it inside `tmux`, so it keeps running if your SSH connection drops:

```bash
tmux new -s model
conda activate unifolm-wma
cd ~/unifolm-world-model-action
bash scripts/run_real_eval_server.sh
```

Wait for:

```
>>> Inference server is ready ...
```

The first start takes several minutes (loading 16.7 GB and downloading CLIP).
Later starts are faster.

- Leave tmux without stopping the server: `Ctrl+B`, then `D`.
- Return to it later: `tmux attach -t model`.
- Stop the server: inside tmux, `Ctrl+C`.

The server prints `images shape`, `states shape`, `actions shape` for every
request, and saves the video the model imagines for each request in
`~/results/unitree_g1_pack_camera/testing/videos/`.

---

## 5. Connect the laptop (on the laptop)

Open the SSH tunnel in its own terminal and leave it open:

```bash
ssh -i ~/.ssh/unifolm-ec2.pem -CNg -L 8000:127.0.0.1:8000 ubuntu@EC2_IP
```

The command prints nothing and does not return: that is normal. While it runs,
`127.0.0.1:8000` on the laptop is the model server on EC2.

**Do not run `mock_policy_server.py` at the same time**: it uses the same port
8000 and the tunnel would fail with "Address already in use".

### 5.1 Quick test with a single request

```bash
conda activate unitree_deploy
python - <<'EOF'
import time, requests

payload = {
    "language_instruction": "pack black camera into box",
    "observation.state": [[0.0] * 16] * 2,                       # 2 frames x 16 joints
    "observation.images.top": [[[[0] * 640] * 480] * 3] * 2,     # 2 frames x 3 x 480 x 640 (black)
    "action": [[0.0] * 16] * 16,                                 # always zeros
}
t = time.time()
reply = requests.post("http://127.0.0.1:8000/predict_action", json=payload).json()
print(f"{reply['result']} in {time.time() - t:.1f} s")
if reply["result"] == "ok":
    print(len(reply["action"]), "actions of", len(reply["action"][0]), "values")
else:
    print(reply["desc"])
EOF
```

Expected: `ok in N s` and `16 actions of 16 values`. The time `N` is the full
round trip (upload + model + reply); note it, it is what every step of the loop
will cost.

### 5.2 Full loop with the simulator

Same as with the mock server ([unitree_deploy/scripts/README.md](unitree_deploy/scripts/README.md)),
with the tunnel taking the place of `mock_policy_server.py`:

```bash
# Terminal 1 - simulator
conda activate unitree_deploy
cd ~/projects/unifolm-wma-postino/unitree_deploy/scripts
python sim_g1_robot.py

# Terminal 2 - SSH tunnel (section 5)

# Terminal 3 - client
conda activate unitree_deploy
cd ~/projects/unifolm-wma-postino/unitree_deploy/scripts
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py \
    --robot_type g1_dex1 --language_instruction "pack black camera into box" --control_freq 15
```

The model was trained on real camera images of this task; the simulator's
scene is different, so expect the movements to be only loosely related to the
instruction.

---

## 6. Speed: what to expect

Every request uploads about **8.4 MB** (two camera images as JSON). Upload time
from the laptop alone:

| Upload speed | Time per request |
| --- | --- |
| 10 Mbit/s | ~7 s |
| 20 Mbit/s | ~3.4 s |
| 100 Mbit/s | ~0.7 s |

Then add the model's own time on the GPU (16 diffusion steps plus the imagined
video). The robot executes 16 actions (~1 s at 15 Hz), then waits for the next
answer. Long pauses between movements are expected with a cloud server.

Measure your upload speed with any speed test site to know which row applies.

---

## 7. Stop paying when you are done

The GPU machine is billed for every hour it is **running**, even when idle.

- **Pause:** EC2 console → select the instance → **Instance state → Stop**.
  Everything on its disk is kept; you only pay for the disk (a few dollars per
  month for 150 GB). **Start** it again later; it gets a **new public IP**, so
  update `EC2_IP` in your commands.
- **Delete everything:** **Instance state → Terminate**. The disk, the
  environment and the checkpoint are gone.

A safety net: AWS console → **Billing → Budgets** → create a monthly budget with
an e-mail alert.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| "You have requested more vCPU capacity than your current vCPU limit" at launch | GPU quota still 0. | Section 1.1; wait for approval. |
| `ssh: connect to host ... Connection timed out` | Security group does not allow your current IP (it changes at home or with VPN). | EC2 → Security groups → edit the SSH rule → "My IP" again. |
| `nvidia-smi: command not found` | Wrong AMI. | Relaunch with the Deep Learning Base OSS Nvidia Driver AMI. |
| `torch.cuda.is_available()` is `False` | Wrong environment, or a CPU-only PyTorch was installed. | `conda activate unifolm-wma`; reinstall with `pip install --force-reinstall torch==2.3.1 torchvision==0.18.1`. |
| `CUDA out of memory` when the server starts or on the first request | 24 GB of GPU memory is not enough for this model. | Stop the instance, change the type to `g6e.xlarge` (Actions → Instance settings → Change instance type), start again. Request quota for it first if needed. |
| Server stops while loading with just `Killed`, or the machine freezes | Ran out of RAM while loading the checkpoint. | Check the swap file is active (`free -h`). If it is, change the type to `g5.4xlarge`. |
| Server start takes very long, disk busy | Loading is using the swap file. | Normal on `g5.2xlarge`; it only happens at start-up. |
| Server error mentioning `/path/to/...` | A placeholder path is left. | Section 3.5. |
| Laptop request: `Connection refused` | Tunnel not open, or server not ready yet. | Check the tunnel terminal; on EC2, `tmux attach -t model` and wait for "Inference server is ready". |
| Tunnel: `bind ... Address already in use` | Something on the laptop already uses port 8000 (often `mock_policy_server.py`). | Stop it, then open the tunnel again. |
| Downloads from Hugging Face fail | Network / rate limit. | Retry; optionally log in with `huggingface-cli login`. |
