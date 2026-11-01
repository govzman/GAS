#!/bin/bash
#SBATCH --job-name=diagnose-synthetic
#SBATCH --output=outputs/diagnose_prompt_dependent_%j.out
#SBATCH --error=outputs/diagnose_prompt_dependent_%j.err
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
HYDRA_FULL_ERROR=1 python diagnose_synthetic.py
