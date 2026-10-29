#!/usr/bin/env bash
#SBATCH --job-name=synt-prompt-global
#SBATCH --output=outputs/train_prompt_dependent_global_%j.out
#SBATCH --error=outputs/train_prompt_dependent_global_%j.err
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=1
#SBATCH --time=0-10:00
#SBATCH --mail-user=idgovzman_2@edu.hse.ru
#SBATCH --mail-type=START,END,FAIL
#SBATCH --constraint=type_a|type_b|type_c

set -euo pipefail
cd GAS
source scripts/load_env.sh
source gas/bin/activate

HYDRA_FULL_ERROR=1 python main.py --config-name gs_synt \
  datasets=synt_prompt_dependent \
  student_solver=sd_student_baseline_ac_fixed_time \
  writer.run_name=synthetic-prompt-dependent-global-4to4
