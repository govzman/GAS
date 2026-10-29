#!/usr/bin/env bash
#SBATCH --job-name=synt-fixed-noise-global
#SBATCH --output=outputs/train_constant_fixed_noise_global_%j.out
#SBATCH --error=outputs/train_constant_fixed_noise_global_%j.err
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
  datasets=synt_constant_fixed_noise \
  student_solver=sd_student_baseline_ac_fixed_time \
  writer.run_name=synthetic-constant-fixed-noise-global-4to4
