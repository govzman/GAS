#!/bin/bash
#SBATCH --job-name=synt-prompt-permutation
#SBATCH --output=outputs/evaluate_prompt_permutation_%j.out
#SBATCH --error=outputs/evaluate_prompt_permutation_%j.err
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
# Supply the checkpoint at submission: sbatch ... <path-relative-to-GAS>
export GAS_CHECKPOINT="${1:?Pass the conditional checkpoint path as the first argument}"
HYDRA_FULL_ERROR=1 python evaluate_prompt_permutation.py
