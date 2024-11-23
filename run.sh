#!/bin/bash
#SBATCH --job-name=ddp_training
#SBATCH --p=v100  # Use the V100 GPU partition
#SBATCH --N=1         # Number of nodes
#SBATCH --ntasks-per-node=1  # Number of tasks per node (one task launches multiple processes)
#SBATCH --gres=gpu:v100:8      # Number of GPUs per node
#SBATCH --cpus-per-task=32  # Number of CPU cores per task (adjust based on your needs)
#SBATCH --time=00:01:00   # Time limit
#SBATCH --output=ddp_job_%j.out  # Standard output and error log


module load cuda/11.0
module load anaconda3
conda activate AI  


export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355  # You can choose any free port number
export WORLD_SIZE=8       # Total number of processes (GPUs)

# Run your training script
python train.py \
    --epochs 100 \
    --batch_size 256 \
    --workers 8 \
    --dist_url env:// \
    --world_size 1  # Set to 1 since we're spawning processes manually
