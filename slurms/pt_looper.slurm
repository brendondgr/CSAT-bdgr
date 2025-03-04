#!/bin/bash
#SBATCH --job-name=ptloop_3
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --partition=gpu2
#SBATCH --account=gpu2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --output=/work/bdgr/CSAT_2/slurm_outputs/pt_looper/%j_fold-3.out
#SBATCH --time=7-0

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

LD_LIBRARY_PATH="/home/bdgr/.conda/envs/bdgr_torch/lib:$LD_LIBRARY_PATH"
export PATH="/work/bdgr/CSAT_2:$PATH"
export PATH="/work/bdgr/CSAT_2/data:$PATH"

python /work/bdgr/CSAT_2/gpu_tests/gpu_test.py
python /work/bdgr/CSAT_2/pretrain_looper.py --cf 3 --save_per_x 10 --epochs 100