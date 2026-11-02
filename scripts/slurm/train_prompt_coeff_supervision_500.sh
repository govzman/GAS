#!/bin/bash
#SBATCH --job-name=synt-coeff-500
#SBATCH --output=outputs/train_prompt_coeff_supervision_500_%j.out
#SBATCH --error=outputs/train_prompt_coeff_supervision_500_%j.err
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
HYDRA_FULL_ERROR=1 python main.py --config-name gs_synt \
  datasets=synt_prompt_dependent_500 \
  +loss_config.coefficient_supervision=true \
  writer.run_name=synthetic-prompt-coefficient-supervision-500train-4to4
