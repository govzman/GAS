# Controlled synthetic experiments

One-time remote setup from the repository root:

```bash
cp .env.example .env
# Edit .env locally on the cluster and set COMET_API_KEY.
```

Submit a teacher-generation job first. It also collates its 100 shards into
`dataset.pkl` when generation succeeds:

```bash
sbatch scripts/slurm/generate_prompt_dependent.sh
```

After the matching generation job is complete, submit exactly one conditional
and one global-baseline training job for that dataset:

```bash
sbatch scripts/slurm/train_prompt_dependent_conditional.sh
sbatch scripts/slurm/train_prompt_dependent_global.sh
```

The remaining two controls use the corresponding `generate_*` and `train_*`
scripts. Slurm output and error files are written to the tracked `outputs/`
directory; its generated contents are ignored by Git.
