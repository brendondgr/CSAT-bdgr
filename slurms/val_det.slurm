#!/bin/bash
#SBATCH --job-name=val_det2
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --partition=gpu2
#SBATCH --account=gpu2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --output=/work/bdgr/CSAT_2/slurm_outputs/val_det/%j.out
#SBATCH --time=2-0

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

LD_LIBRARY_PATH="/home/bdgr/.conda/envs/bdgr_torch/lib:$LD_LIBRARY_PATH"
export PATH="/work/bdgr/CSAT_2:$PATH"
export PATH="/work/bdgr/CSAT_2/data:$PATH"

python /work/bdgr/CSAT_2/gpu_tests/gpu_test.py
python /work/bdgr/CSAT_2/validate_detection.py --pretrained_weights_path /work/bdgr/CSAT_2/runs/4bb_best_pretrainer_4.pth #--validation_iter_stop True --validation_stop_round 20 --debug True