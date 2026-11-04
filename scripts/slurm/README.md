# Controlled synthetic experiments

Run these scripts from the directory that contains `GAS/`, as in the existing
cluster workflow. Each script begins with `cd GAS`, so Slurm keeps
`outputs/*.out` and `outputs/*.err` relative to that parent directory.

One-time remote setup:

```bash
cp GAS/.env.example GAS/.env
# Edit .env locally on the cluster and set COMET_API_KEY.
```

Run a teacher-generation script first. It also collates its 100 shards into
`dataset.pkl` when generation succeeds:

```bash
sbatch GAS/scripts/slurm/generate_prompt_dependent.sh
```

After the matching generation job is complete, run exactly one conditional and
one global-baseline training script for that dataset:

```bash
sbatch GAS/scripts/slurm/train_prompt_dependent_conditional.sh
sbatch GAS/scripts/slurm/train_prompt_dependent_global.sh
```

The remaining two controls use the corresponding `generate_*` and `train_*`
scripts. Slurm logs are written to `outputs/` beside `GAS/`.

For the existing 500/20 dataset, two additional latent-loss ablations use the
same seed, 5000 steps, batch size 2, lr=1e-4 and checkpoints every 250 steps:

```bash
sbatch GAS/scripts/slurm/train_prompt_latent_500_shuffled_prompt.sh
sbatch GAS/scripts/slurm/train_prompt_latent_500_prompt_only.sh
```

The shuffle run samples a fresh unrelated prompt from the entire train pool
for each predictor input during training. SD always receives the original
prompt; validation uses correct prompts. Sampling is with replacement and
excludes the current prompt; validation prompts never enter the pool.
The prompt-only run removes the noise encoder and FiLM layers, retaining the
text branch and output head plus global coefficient tables. Its smaller model
changes capacity as well as inputs, which should be considered when comparing.
To evaluate a checkpoint with the correct architecture, pass its config name:

```bash
GAS_CHECKPOINT=checkpoints/RUN/5000.pt python evaluate_coefficient_checkpoint.py --config-name gs_synt_500_prompt_only
```

That last command runs from inside `GAS/`, with the Python environment active.
