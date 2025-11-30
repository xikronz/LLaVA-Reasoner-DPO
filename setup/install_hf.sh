#!/bin/bash

# ----------------------------
# Setup caches to avoid permission issues
export XDG_CACHE_HOME=$HOME/.cache
export PIP_CACHE_DIR=$HOME/.cache/pip
mkdir -p $XDG_CACHE_HOME $PIP_CACHE_DIR

# ----------------------------
# Ensure conda commands work
if [ -f "/share/apps/software/anaconda3/etc/profile.d/conda.sh" ]; then
    source /share/apps/software/anaconda3/etc/profile.d/conda.sh
else
    echo "Error: conda.sh not found!"
    exit 1
fi


conda activate r2dtuning
conda install -c conda-forge git-lfs
git lfs install
# huggingface-cli lfs-enable-largefiles .

git config --global credential.helper cache
pip install huggingface_hub==0.25.0