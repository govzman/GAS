#!/bin/bash
#SBATCH --job-name=permute-prompt-5to4
#SBATCH --output=outputs/evaluate_prompt_permutation_5to4_%j.out
#SBATCH --error=outputs/evaluate_prompt_permutation_5to4_%j.err
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=1
#SBATCH --time=0-1:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint="type_a|type_b|type_c"

set -euo pipefail
cd GAS
source gas/bin/activate
export GAS_CHECKPOINT="${1:?Pass CONDITIONAL checkpoint path relative to GAS}"
HYDRA_FULL_ERROR=1 python evaluate_prompt_permutation.py \
  --config-name gs_synt_5to4_prompt_dependent_conditional_500
