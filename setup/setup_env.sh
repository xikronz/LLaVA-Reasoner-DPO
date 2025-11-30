#!/bin/bash
# ----------------------------
# Fix caches to avoid permission issues
export XDG_CACHE_HOME=$HOME/.cache
export PIP_CACHE_DIR=$HOME/.cache/pip
mkdir -p $XDG_CACHE_HOME $PIP_CACHE_DIR

export DATA_DIR=$HOME/data
export CKPT_DIR=$HOME/ckpt
export CACHE_DIR=$HOME/cache
mkdir -p $DATA_DIR $CKPT_DIR $CACHE_DIR

# ----------------------------
# Initialize conda only if not already in the environment
if [[ "$CONDA_DEFAULT_ENV" != "r2dtuning" ]]; then
    if [ -f "/share/apps/software/anaconda3/etc/profile.d/conda.sh" ]; then
        source /share/apps/software/anaconda3/etc/profile.d/conda.sh
        export CONDA_ENVS_PATH=$HOME/.conda/envs
        conda activate r2dtuning
    else
        echo "Error: conda.sh not found!"
        exit 1
    fi
fi

echo "Using conda environment: $CONDA_DEFAULT_ENV"

# ----------------------------
# Install build dependencies first
pip install --upgrade pip setuptools wheel setuptools_scm

# ----------------------------
# Huggingface setup
echo "Huggingface setup"
source setup/install_hf.sh

# ----------------------------
# Install requirements
echo "Installing requirements"
pip install --no-build-isolation -r setup/requirements.txt

# ----------------------------
# Source secret/path scripts if exist
if [ -f setup/set_path_secret.sh ]; then
    echo "Sourcing set_path_secret.sh"
    source setup/set_path_secret.sh
else
    echo "Sourcing set_path.sh"
    source setup/set_path.sh
fi

# ----------------------------
# Install LLaVA-Reasoner-DPO repo
cd /share/cuvl/cc2864/sft_cot/LLaVA-Reasoner-DPO
pip install -e .
