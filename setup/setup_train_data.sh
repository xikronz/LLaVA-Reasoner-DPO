#!/bin/bash

# ----------------------------
# Set directories
DATA_DIR=/share/cuvl/cc2864/LLaVA-Reasoner-DPO/data
IMAGE_DATA_DIR=$DATA_DIR/images
mkdir -p $IMAGE_DATA_DIR

# ----------------------------
# Set code directory
CODE_DIR=/share/cuvl/cc2864/LLaVA-Reasoner-DPO

# ----------------------------
# Check git-lfs
if ! [ -x "$(command -v git-lfs)" ]; then
    echo "Error: git-lfs is not installed." >&2
    exit 1
fi

# ----------------------------
# Clone dataset if missing
cd $DATA_DIR
repo_name=sft_data
repo_path=https://huggingface.co/datasets/Share4oReasoning/$repo_name

if [ ! -d "$repo_name" ]; then
    git clone $repo_path
else
    echo "$repo_name already exists, skipping clone."
fi

# ----------------------------
# Extract all tar(.gz) files
cd $repo_name/image_data || { echo "Directory $repo_name/image_data not found"; exit 1; }

for file in *.tar.gz *.tar; do
    [ -e "$file" ] || continue
    echo "Extracting $file ..."
    tar -xf "$file" -C $IMAGE_DATA_DIR || echo "Warning: $file failed to extract"
    echo "$file extraction finished."
done

# ----------------------------
# Extract image_mix if exists
if [ -d image_mix ]; then
    cd image_mix
    for file in *.tar.gz *.tar; do
        [ -e "$file" ] || continue
        echo "Extracting $file ..."
        tar -xf "$file" -C $IMAGE_DATA_DIR || echo "Warning: $file failed to extract"
        echo "$file extraction finished."
    done
fi

# ----------------------------
# Return to code directory
cd $CODE_DIR
echo "Data setup complete. All files extracted to $IMAGE_DATA_DIR."
