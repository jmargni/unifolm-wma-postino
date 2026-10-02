#!/usr/bin/env bash
# Build the conda environment needed to run unitree_deploy/scripts (robot client + mocks).
#
# Usage:
#   bash unitree_deploy/setup_env.sh
#
# Optional variables:
#   ENV_NAME   name of the conda environment            (default: unitree_deploy)
#   DEPS_DIR   where cyclonedds and unitree_sdk2_python
#              are cloned                               (default: the folder next to this repository)
#   TORCH_INDEX_URL  PyTorch wheel index                (default: CPU-only wheels)
set -euo pipefail

ENV_NAME="${ENV_NAME:-unitree_deploy}"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEPLOY_DIR")"
DEPS_DIR="${DEPS_DIR:-$(dirname "$REPO_ROOT")}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# Pinned versions of the two dependencies that are built from source.
CYCLONEDDS_REPO="https://github.com/eclipse-cyclonedds/cyclonedds.git"
CYCLONEDDS_COMMIT="5041f3560c088c99e5088b2b8520b69169621196"  # branch releases/0.10.x
SDK_REPO="https://github.com/unitreerobotics/unitree_sdk2_python.git"
SDK_COMMIT="814556d15970dd2ecf1c9984e845ca02ab07e206"

# Clone a repository at one exact commit (no-op if the folder already exists).
clone_at_commit() {
    local repo="$1" commit="$2" dir="$3"
    if [ -d "$dir/.git" ]; then
        echo ">>> $dir already exists, keeping it."
        return
    fi
    git init -q "$dir"
    git -C "$dir" remote add origin "$repo"
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
}

for tool in conda git cmake gcc; do
    command -v "$tool" >/dev/null || { echo "ERROR: '$tool' not found. See README_REPRODUCE.md, section 'Prerequisites'."; exit 1; }
done

mkdir -p "$DEPS_DIR"

echo ">>> [1/5] Creating conda environment '$ENV_NAME' ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Environment '$ENV_NAME' already exists, reusing it."
else
    conda create -y -n "$ENV_NAME" -c conda-forge python=3.10 pinocchio=4.1
fi
conda activate "$ENV_NAME"

echo ">>> [2/5] Building CycloneDDS (C library) ..."
CYCLONEDDS_DIR="$DEPS_DIR/cyclonedds"
clone_at_commit "$CYCLONEDDS_REPO" "$CYCLONEDDS_COMMIT" "$CYCLONEDDS_DIR"
if [ ! -d "$CYCLONEDDS_DIR/install/lib" ]; then
    cmake -S "$CYCLONEDDS_DIR" -B "$CYCLONEDDS_DIR/build" -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_DIR/install" \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF
    cmake --build "$CYCLONEDDS_DIR/build" --target install --parallel
fi
export CYCLONEDDS_HOME="$CYCLONEDDS_DIR/install"

echo ">>> [3/5] Installing PyTorch from $TORCH_INDEX_URL ..."
pip install torch==2.3.1 torchvision==0.18.1 --index-url "$TORCH_INDEX_URL"

echo ">>> [4/5] Installing unitree_deploy ..."
pip install -c "$DEPLOY_DIR/constraints.txt" -e "$DEPLOY_DIR"

echo ">>> [5/5] Installing unitree_sdk2_python ..."
SDK_DIR="$DEPS_DIR/unitree_sdk2_python"
clone_at_commit "$SDK_REPO" "$SDK_COMMIT" "$SDK_DIR"
pip install -c "$DEPLOY_DIR/constraints.txt" -e "$SDK_DIR"

echo ">>> Checking the installation ..."
python -c "import pinocchio, mujoco, torch, cyclonedds, unitree_sdk2py, unitree_deploy; print('All imports OK')"

echo ">>> Done. Next:  conda activate $ENV_NAME  and follow unitree_deploy/scripts/README.md"
