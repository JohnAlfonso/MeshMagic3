#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_FILE="${SCRIPT_DIR}/conda_env.yml"
WORKDIR=$(pwd)

# Install miniconda if not present
if [ -z "$(which conda)" ]; then
    echo "Installing Miniconda..."
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init --all
    apt update
    apt install nano vim -y
    apt install npm -y
    npm install -g pm2@6.0.12
fi

export PATH="$HOME/miniconda3/bin:$PATH"
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create or update environment
if conda env list | grep -q '^404-base-miner-env\s'; then
    echo "Environment 404-base-miner-env exists; updating"
    echo -e "a\na" | conda env update -f "$CONDA_ENV_FILE" --prune
else
    echo "Creating 404-base-miner-env environment..."
    echo -e "a\na" | conda env create -f "$CONDA_ENV_FILE"
fi

conda activate 404-base-miner-env

# Setup CUDA activation hooks
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<'SH'
export CUDA_ENV_PREFIX="$CONDA_PREFIX"
if [ -z "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_SAVED_CUDA_HOME="${CUDA_HOME:-}"; fi
if [ -z "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_SAVED_CUDA_PATH="${CUDA_PATH:-}"; fi
if [ -z "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export CUDA_SAVED_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"; fi
if [ -z "${CUDA_SAVED_TORCH_CUDA_ARCH_LIST+x}" ]; then export CUDA_SAVED_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"; fi
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
SH

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cuda.sh" <<'SH'
if [ -n "${CUDA_SAVED_CUDA_HOME+x}" ]; then export CUDA_HOME="${CUDA_SAVED_CUDA_HOME}"; else unset CUDA_HOME; fi
if [ -n "${CUDA_SAVED_CUDA_PATH+x}" ]; then export CUDA_PATH="${CUDA_SAVED_CUDA_PATH}"; else unset CUDA_PATH; fi
if [ -n "${CUDA_SAVED_LD_LIBRARY_PATH+x}" ]; then export LD_LIBRARY_PATH="${CUDA_SAVED_LD_LIBRARY_PATH}"; else unset LD_LIBRARY_PATH; fi
if [ -n "${CUDA_SAVED_TORCH_CUDA_ARCH_LIST+x}" ]; then export TORCH_CUDA_ARCH_LIST="${CUDA_SAVED_TORCH_CUDA_ARCH_LIST}"; else unset TORCH_CUDA_ARCH_LIST; fi
if [ -n "$CUDA_ENV_PREFIX" ]; then
    PATH=":${PATH:-}:"; PATH="${PATH//:$CUDA_ENV_PREFIX\/bin:/:}"; PATH="${PATH#:}"; PATH="${PATH%:}"; export PATH
fi
unset CUDA_ENV_PREFIX CUDA_SAVED_CUDA_HOME CUDA_SAVED_CUDA_PATH CUDA_SAVED_LD_LIBRARY_PATH CUDA_SAVED_TORCH_CUDA_ARCH_LIST
SH

# Verify CUDA platform
if ! command -v nvidia-smi > /dev/null; then
    echo "Error: nvidia-smi not found. CUDA platform required."
    exit 1
fi

pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Force CUDA arch flags for builds without GPU access
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0;12.0}"
export FORCE_CUDA=1

TEMP_DIR="/tmp/extensions"
rm -rf $TEMP_DIR
mkdir -p $TEMP_DIR

# Install flash-attention
echo "Installing flash-attention..."
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
rm flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install Kaolin
echo "Installing Kaolin..."
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

# Install cumesh
echo "Installing cumesh..."
git clone https://github.com/JeffreyXiang/CuMesh.git $TEMP_DIR/CuMesh --recursive
(cd $TEMP_DIR/CuMesh && git checkout 6f403664cc79e5a0f0c993d2d03976ace0e7f829)
pip install $TEMP_DIR/CuMesh --no-build-isolation

# Install flexgemm
echo "Installing flexgemm..."
git clone https://github.com/JeffreyXiang/FlexGEMM.git $TEMP_DIR/FlexGEMM --recursive
(cd $TEMP_DIR/FlexGEMM && git checkout de6411284d20a6d41362db27a59a6923aeceebfe)
pip install $TEMP_DIR/FlexGEMM --no-build-isolation

# Install o-voxel
echo "Installing o-voxel..."
cp -r o-voxel $TEMP_DIR/o-voxel
git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git $TEMP_DIR/o-voxel/third_party/eigen
pip install $TEMP_DIR/o-voxel --no-build-isolation

echo "Environment setup complete."

# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the validation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
    args: '--port 10006'
  }]
};
EOF

rm -rf $TEMP_DIR