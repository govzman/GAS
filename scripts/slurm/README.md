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
scripts. Slurm output and error files are written to the tracked `outputs/`
directory; its generated contents are ignored by Git.
