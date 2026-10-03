# Hosting this work in your own git repository and reproducing the environment

This document explains how to:

1. put this working copy (Unitree's code plus our local changes) into a git
   repository of your own, and
2. rebuild the same running environment on another machine from that
   repository.

How to *run* the programs once the environment exists is in
[`unitree_deploy/scripts/README.md`](unitree_deploy/scripts/README.md).

---

## 1. What the environment is made of

| Piece | Where it comes from | Stored in our repository? |
| --- | --- | --- |
| This repository (Unitree's `unifolm-world-model-action` + our changes) | Your git host | **Yes** |
| Conda environment `unitree_deploy` (Python 3.10, pinocchio, PyTorch, MuJoCo, ...) | Built by `unitree_deploy/setup_env.sh` | No (about 2.7 GB, rebuilt on each machine) |
| CycloneDDS 0.10.x, the C library for the robot's network protocol | Cloned from GitHub at a pinned commit and compiled by the script | No |
| `unitree_sdk2_python`, Unitree's Python SDK | Cloned from GitHub at a pinned commit by the script | No |
| Model checkpoint (16.7 GB) | Hugging Face, see the main `README.md` | No (only needed on the GPU machine) |

The repository therefore holds everything that is *ours*, plus the recipe to
fetch and build everything that is not.

### Our changes on top of Unitree's code

Unitree's code is at commit `3e198de` of
`https://github.com/unitreerobotics/unifolm-world-model-action`.

| File | Change |
| --- | --- |
| `pyproject.toml` | Accept any Python 3.10.x instead of exactly 3.10.18. |
| `unitree_deploy/pyproject.toml` | Pinned and added missing dependencies. |
| `unitree_deploy/unitree_deploy/robot_devices/arm/g1_arm_ik.py`, `z1_arm_ik.py` | Robot model files are found regardless of the folder you start from. |
| `unitree_deploy/scripts/mock_g1_robot.py` | New: fake G1 robot. |
| `unitree_deploy/scripts/mock_policy_server.py` | New: fake model server. |
| `unitree_deploy/scripts/sim_g1_robot.py` | New: physics simulation of the G1 with grippers, cameras and a table scene (MuJoCo). |
| `unitree_deploy/unitree_deploy/robot_devices/cameras/imageclient.py` | The camera server address can be set with the `UNITREE_IMAGE_SERVER` environment variable (default: the real robot, `192.168.123.164`). |
| `unitree_deploy/scripts/README.md` | New: how to run the client, the mocks and the simulator. |
| `unitree_deploy/setup_env.sh` | New: builds the environment. |
| `unitree_deploy/constraints.txt` | New: exact versions of all Python packages of the working environment. |
| `README_REPRODUCE.md` | New: this file. |

---

## 2. Create the repository and push the code

### 2.1 Before you start: the licence

Unitree's code is under **CC BY-NC-SA 4.0** (see `LICENSE`):

- **Attribution:** keep the `LICENSE` file and Unitree's copyright notices.
- **NonCommercial:** the code may not be used for commercial purposes.
- **ShareAlike:** if you distribute a modified version, it must stay under the
  same licence.

Check with whoever is responsible for licensing in your organisation that the
intended use is allowed, and create the repository as **private** unless you
have decided otherwise.

### 2.2 Create an empty repository on your git host

On GitHub, GitLab, Bitbucket or your company's server, create a new repository:

- Name: for example `unifolm-wma-postino`.
- Visibility: **private**.
- Do **not** tick "add a README", "add .gitignore" or "add a licence": the
  repository must be completely empty, because the code already has them.

Copy the repository address it shows you, for example
`git@github.com:YOUR-ORG/unifolm-wma-postino.git`.

### 2.3 Point this working copy to your repository

The working copy is currently linked to Unitree's repository under the name
`origin`. Rename that link to `upstream` (so you can still fetch Unitree's
updates) and add yours as the new `origin`:

```bash
cd ~/pyspace/unifolm-world-model-action

git remote rename origin upstream
git remote add origin git@github.com:YOUR-ORG/unifolm-wma-postino.git
git remote -v        # check: 'origin' is yours, 'upstream' is unitreerobotics
```

### 2.4 Commit our changes

```bash
git status           # review what will be saved

git add pyproject.toml \
        README_REPRODUCE.md \
        unitree_deploy/pyproject.toml \
        unitree_deploy/setup_env.sh \
        unitree_deploy/constraints.txt \
        unitree_deploy/scripts/README.md \
        unitree_deploy/scripts/mock_g1_robot.py \
        unitree_deploy/scripts/mock_policy_server.py \
        unitree_deploy/unitree_deploy/robot_devices/arm/g1_arm_ik.py \
        unitree_deploy/unitree_deploy/robot_devices/arm/z1_arm_ik.py

git commit -m "Add mock robot, mock policy server and reproducible setup"
```

Listing the files one by one (instead of `git add .`) makes sure nothing
unintended is saved.

### 2.5 Push

```bash
git push -u origin main
```

This uploads Unitree's history plus your commit (about 120 MB). Open the
repository page in the browser and check that `README_REPRODUCE.md` is there.

### 2.6 What must never be committed

- **Model checkpoints** (`*.ckpt`, 16.7 GB): too large for git. They are
  downloaded from Hugging Face.
- **Passwords, VPN profiles, tokens, private keys.**
- **The conda environment and the `cyclonedds` / `unitree_sdk2_python`
  folders:** they are rebuilt by the setup script.
- **`results/` folders** created by `robot_client.py`. Add them to the ignore
  list once:

  ```bash
  echo "results/" >> .gitignore
  git add .gitignore && git commit -m "Ignore results folders"
  ```

### 2.7 Later: getting Unitree's updates

```bash
git fetch upstream
git merge upstream/main     # may ask you to resolve conflicts in the files listed in section 1
git push
```

---

## 3. Reproduce the environment on another machine

Tested on Ubuntu 24.04 (x86-64). A GPU is **not** needed for the client and the
mocks.

### 3.1 Prerequisites

```bash
# System tools: git, C compiler, cmake (needed to compile CycloneDDS)
sudo apt update
sudo apt install -y git build-essential cmake

# Conda (Miniforge). Skip if 'conda' already works.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh        # accept the defaults, then open a new terminal
```

You need about 5 GB of free disk space.

### 3.2 Get the code

```bash
mkdir -p ~/pyspace && cd ~/pyspace
git clone git@github.com:YOUR-ORG/unifolm-wma-postino.git unifolm-world-model-action
cd unifolm-world-model-action
```

The `external/dlimp` submodule is only used for training data preparation; it
is not needed here, so no `--recurse-submodules`.

### 3.3 Build the environment

```bash
bash unitree_deploy/setup_env.sh
```

The script takes roughly 5 to 15 minutes, depending on the network. It:

1. creates the conda environment `unitree_deploy` (Python 3.10 + pinocchio),
2. clones CycloneDDS next to the repository (`~/pyspace/cyclonedds`) and compiles it,
3. installs PyTorch 2.3.1 (CPU build),
4. installs `unitree_deploy` with the exact package versions in `constraints.txt`,
5. clones `unitree_sdk2_python` next to the repository and installs it,
6. checks that everything imports and prints `All imports OK`.

It can be run again safely: steps already done are reused.

Options, set in front of the command:

| Variable | Default | Use |
| --- | --- | --- |
| `ENV_NAME` | `unitree_deploy` | Another name for the conda environment. |
| `DEPS_DIR` | folder containing the repository | Where `cyclonedds` and `unitree_sdk2_python` are cloned. |
| `TORCH_INDEX_URL` | CPU wheels | Use `https://download.pytorch.org/whl/cu121` for a machine with an NVIDIA GPU. |

Example: `ENV_NAME=postino DEPS_DIR=~/deps bash unitree_deploy/setup_env.sh`

### 3.4 Check that it works

Three terminals, each with:

```bash
conda activate unitree_deploy
cd ~/pyspace/unifolm-world-model-action/unitree_deploy/scripts
```

```bash
python mock_g1_robot.py        # terminal 1 (add --headless if there is no screen)
python mock_policy_server.py   # terminal 2
python robot_client.py         # terminal 3
```

The client must print `All Device Connect Success` and then
`>>> Exec => step N action: [...]` lines. Details and examples are in
[`unitree_deploy/scripts/README.md`](unitree_deploy/scripts/README.md).

To check the physics simulator with cameras instead of the mock robot, replace
terminals 1 and 3 with:

```bash
python sim_g1_robot.py                                   # terminal 1
UNITREE_IMAGE_SERVER=127.0.0.1 python robot_client.py    # terminal 3
```

The simulator prints `real-time x1.00` every 2 seconds, and the mock server
prints `has content` (not `all black`) for each request. The camera rendering
needs OpenGL with EGL, which the graphics drivers of a normal Ubuntu desktop
provide.

---

## 4. Keeping it reproducible

- **You added or upgraded a Python package:** regenerate the version list and
  commit it.

  ```bash
  conda activate unitree_deploy
  pip freeze --exclude-editable | grep -v -E "^(torch|torchvision)|@ " > unitree_deploy/constraints.txt
  git add unitree_deploy/constraints.txt && git commit -m "Update pinned package versions"
  ```

- **You need a newer CycloneDDS or Unitree SDK:** change `CYCLONEDDS_COMMIT` or
  `SDK_COMMIT` at the top of `unitree_deploy/setup_env.sh`, delete the
  corresponding folder, and run the script again.

- **Work on a change:** use a branch, so `main` always stays in a working state.

  ```bash
  git checkout -b my-change
  # ... edit, test ...
  git add <files> && git commit -m "Describe the change"
  git push -u origin my-change
  ```

---

## 5. Not covered here

- **The real model server** (`scripts/evaluation/real_eval_server.py`): needs an
  NVIDIA GPU, a second environment and the checkpoint. Follow the main
  `README.md`.
- **The real robot or a remote simulator** (such as the lab's digital twin):
  network set-up and access are specific to each site. The local simulator
  `sim_g1_robot.py` is covered in section 3.4.
