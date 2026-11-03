#!/bin/bash
#SBATCH --job-name=synt-latent-global-500
#SBATCH --output=outputs/train_prompt_latent_global_500_low_lr_%j.out
#SBATCH --error=outputs/train_prompt_latent_global_500_low_lr_%j.err
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
  student_solver=sd_student_baseline_ac_fixed_time \
  +loss_config.coefficient_supervision=false \
  loss_config.loss_key=loss_l1_latents \
  optimizer.lr=0.0001 \
  writer.checkpoint_freq=250 \
  writer.run_name=synthetic-prompt-latent-global-500train-lr-1e-4-4to4
