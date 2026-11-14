#!/bin/bash
#SBATCH --job-name=resume-global-5to4
#SBATCH --output=outputs/resume_global_5to4_%j.out
#SBATCH --error=outputs/resume_global_5to4_%j.err
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint="type_a|type_b|type_c"

set -euo pipefail
cd GAS
source scripts/load_env.sh
source gas/bin/activate
export GAS_CHECKPOINT="${1:?Pass GLOBAL checkpoint path relative to GAS}"
shift
HYDRA_FULL_ERROR=1 python main.py --config-name resume_synt_5to4_global_500 "$@"
