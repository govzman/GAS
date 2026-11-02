#!/bin/bash
#SBATCH --job-name=synt-coeff-eval
#SBATCH --output=outputs/evaluate_coefficient_checkpoint_%j.out
#SBATCH --error=outputs/evaluate_coefficient_checkpoint_%j.err
#SBATCH --gpus=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --time=0-1:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint="type_a|type_b|type_c"

set -euo pipefail
cd GAS
source gas/bin/activate
export GAS_CHECKPOINT="${1:?Pass checkpoint path relative to GAS}"
HYDRA_FULL_ERROR=1 python evaluate_coefficient_checkpoint.py
