#!/bin/bash
#SBATCH --job-name=synt-expand-500
#SBATCH --output=outputs/generate_prompt_dependent_500_%j.out
#SBATCH --error=outputs/generate_prompt_dependent_500_%j.err
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint="type_a|type_b|type_c"

set -euo pipefail
cd GAS
source gas/bin/activate
HYDRA_FULL_ERROR=1 python expand_synthetic_dataset.py
