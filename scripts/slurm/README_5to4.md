# Synthetic solver distillation: 5 teacher evaluations to 4 student evaluations

Stage 1: start with the prompt-dependent teacher to test whether the known
synthetic context signal helps distillation. From the directory containing GAS/:

```bash
sbatch GAS/scripts/slurm/generate_synt_5to4_prompt_dependent_500.sh
```

Each job generates and collates 520 examples (500 train, 20 validation).
Both controls select the same prompts and noise seeds. Their validation set is
new: it is not the old 4-to-4 validation set. Separate output directories protect
the previous datasets. Existing output directories cause an explicit error.

After the matching generation job finishes successfully:

```bash
sbatch GAS/scripts/slurm/train_synt_5to4_prompt_dependent_conditional_500.sh
sbatch GAS/scripts/slurm/train_synt_5to4_prompt_dependent_global_500.sh
```

Stage 2 (later): the constant-teacher control is prepared separately. Generate
its data, wait for generation to finish, then submit its two training jobs:

```bash
sbatch GAS/scripts/slurm/generate_synt_5to4_constant_500.sh
# After successful generation:
sbatch GAS/scripts/slurm/train_synt_5to4_constant_conditional_500.sh
sbatch GAS/scripts/slurm/train_synt_5to4_constant_global_500.sh
```

These runs optimize latent L1, with lr=1e-4, 5000 steps, batch size 2,
seed 42 and checkpoint frequency 250. Student t/t_couple remain fixed.
Teacher is a five-step GS with order 3 and manual synthetic coefficients,
not the original iPNDM teacher. A constant teacher still has prompt-dependent
SD trajectories; its five-step coefficient table is global.

There is no direct coefficient supervision or teacher/student coefficient MSE:
five-step and four-step tables cannot be compared elementwise. The old
oracle coefficient replay diagnostic also assumes equal teacher/student NFE
and must not be used as a four-step zero-loss benchmark here.
