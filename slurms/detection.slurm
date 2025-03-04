#!/bin/bash
#SBATCH --job-name=csat_det2
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --partition=gpu2
#SBATCH --account=gpu2
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --output=/work/bdgr/CSAT_2/slurm_outputs/detect/detect_%j.out
#SBATCH --time=2-0

LD_LIBRARY_PATH="/home/bdgr/.conda/envs/bdgr_torch/lib:$LD_LIBRARY_PATH"
export PATH="/work/bdgr/CSAT_2:$PATH"
export PATH="/work/bdgr/CSAT_2/data:$PATH"

python /work/bdgr/CSAT_2/gpu_tests/gpu_test.py
python /work/bdgr/CSAT_2/detector.py