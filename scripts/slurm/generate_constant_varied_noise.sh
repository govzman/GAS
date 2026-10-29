#!/usr/bin/env bash
#SBATCH --job-name=synt-constant-noise
#SBATCH --output=outputs/generate_constant_varied_noise_%j.out
#SBATCH --error=outputs/generate_constant_varied_noise_%j.err
#SBATCH --gpus=1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --time=0-1:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint=type_a|type_b|type_c

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/GAS" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}/GAS"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/load_env.sh"
source "${REPO_ROOT}/gas/bin/activate"

python generate.py --config-name generate_synt_constant_varied_noise
python collate.py \
  --synt_dir data/teachers/sd-v1/synthetic/constant_varied_noise/dataset \
  --out_pkl data/teachers/sd-v1/synthetic/constant_varied_noise/dataset.pkl \
  --num_samples 100
