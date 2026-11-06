#!/bin/bash
#SBATCH --job-name=fit-individual-5to4
#SBATCH --output=outputs/fit_individual_5to4_%j.out
#SBATCH --error=outputs/fit_individual_5to4_%j.err
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
export GAS_CHECKPOINT="${1:?Pass GLOBAL checkpoint path relative to GAS}"
shift
HYDRA_FULL_ERROR=1 python fit_individual_solver.py "$@"
