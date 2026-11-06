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

## Diagnostic: is there room for individual a/c adaptation?

From the directory containing GAS/, supply a global checkpoint path relative
to GAS/ (prefer a late checkpoint, ideally from a converged global run):

```bash
sbatch GAS/scripts/slurm/fit_individual_synt_5to4.sh checkpoints/GLOBAL_RUN/5000.pt
```

By default this chooses 8 validation examples with selection seed 42, loads EMA
weights and holds the four-step schedule and t_couple fixed. It first fits one
shared a/c table on those 8 examples, then independently fits a/c for each one.
Every fit starts fresh from the same global checkpoint with a new Adam optimizer
(no weight decay). Each example is seen 250 times in each condition; shared
fitting averages gradients across all 8 before updating. Learning rate decays
from 1e-4 to 1e-5. Neither fit trains SD or a context predictor. Only the 18
active student coefficients are fitted; teacher coefficient labels are unused.

This intentionally fits validation targets: the reported best losses quantify
empirically found fitting headroom, not generalization or a theoretical optimum.
The shared control helps distinguish individual adaptation from simply further
optimizing an undertrained global checkpoint on the selected subset.

Results appear in the Slurm stdout log and in
GAS/outputs/individual_solver_5to4/TIMESTAMP/:
config.yaml, history.csv, samples.csv, summary.json and best_coefficients.pt.
The history records every step; summaries report starting, best and final losses.
Nonfinite losses/gradients stop the affected fit and are reported explicitly.
The saved coefficients are diagnostic tables, not normal training checkpoints.

Start with a short remote smoke test if desired:

```bash
sbatch GAS/scripts/slurm/fit_individual_synt_5to4.sh checkpoints/GLOBAL_RUN/5000.pt diagnostic.count=1 diagnostic.iterations=2
```

Overrides can also select raw checkpoint weights (`diagnostic.weights=raw`),
train examples (`diagnostic.split=train`) or a larger optimization budget.
Do not interpret a small gain after a finite budget as proof that adaptation
cannot help. Check the history for convergence and consider learning-rate or
restart checks before making that inference.

For the independent predictor-prompt permutation check, pass a conditional
checkpoint instead:

```bash
sbatch GAS/scripts/slurm/evaluate_prompt_permutation_5to4.sh checkpoints/CONDITIONAL_RUN/5000.pt
```

This runs the correct 5-to-4 dataset/config, reports raw and EMA results over all
20 validation examples, and leaves SD conditioning unchanged. No Comet key is
needed for either diagnostic.
