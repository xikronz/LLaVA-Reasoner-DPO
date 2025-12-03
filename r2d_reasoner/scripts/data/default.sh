#!/bin/bash
#SBATCH -J r2d-3-4                      # Job name
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH --cpus-per-task=16                    # Total number of cores requested per task
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=128G                           # server memory requested in MB (per node)
#SBATCH -t 72:00:00                           # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition        # Request partition
#SBATCH --gres=gpu:nvidia_h100_nvl:2                # Number of GPUs

# Run two processes in parallel, each on a different GPU
# GPU 0: samples 0-5000
# GPU 1: samples 5000-10000

python scripts/data/evaluate_3VL.py --start 0 --end 5000 --gpu 0 &
python scripts/data/evaluate_3VL.py --start 5000 --end 10000 --gpu 1 &

# Wait for both processes to complete
wait

echo "Both GPU processes completed!"