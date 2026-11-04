#!/bin/bash
#SBATCH --job-name=gen-5to4-prompt_dependent
#SBATCH --output=outputs/gen-5to4-prompt_dependent_%j.out
#SBATCH --error=outputs/gen-5to4-prompt_dependent_%j.err
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
# Separate output from all previous 4-to-4 datasets; refuse accidental reruns.
if [[ -e "data/teachers/sd-v1/synthetic/5to4_prompt_dependent_500" ]]; then
  echo "Output data/teachers/sd-v1/synthetic/5to4_prompt_dependent_500 already exists; choose a new output directory before rerunning." >&2
  exit 1
fi
HYDRA_FULL_ERROR=1 python generate.py --config-name generate_synt_5to4_prompt_dependent_500
python collate.py \
  --synt_dir data/teachers/sd-v1/synthetic/5to4_prompt_dependent_500/dataset \
  --out_pkl data/teachers/sd-v1/synthetic/5to4_prompt_dependent_500/dataset.pkl \
  --num_samples 520

