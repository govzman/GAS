#!/usr/bin/env bash
#SBATCH --job-name=synt-constant-noise-cond
#SBATCH --output=outputs/train_constant_varied_noise_conditional_%j.out
#SBATCH --error=outputs/train_constant_varied_noise_conditional_%j.err
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint=type_a|type_b|type_c

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/load_env.sh"
source "${REPO_ROOT}/gas/bin/activate"

HYDRA_FULL_ERROR=1 python main.py --config-name gs_synt \
  datasets=synt_constant_varied_noise \
  student_solver=sd_student_conditional_ac \
  writer.run_name=synthetic-constant-varied-noise-conditional-4to4
