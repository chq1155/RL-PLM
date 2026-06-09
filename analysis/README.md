# Analysis Utilities

`metrics.py` computes the main capability-level diagnostics used in the paper:
pass@k, Expansion-Shrinkage Ratio (ESR), Dual-Reward ESR, and delta-ESR.

Prepare a scored sample CSV with one row per generated sequence. The default
schema is:

```text
context_id,model,rank,sequence,train_score,independent_score
```

- `context_id`: design problem or prompt identifier.
- `model`: `base` for the pretrained PLM or `rl` for the RL-tuned PLM.
- `rank`: optional sample order within each context.
- `train_score`: training reward score.
- `independent_score`: orthogonal evaluator score.

Single-reward ESR:

```bash
python analysis/metrics.py \
  --samples exports/kinase_scored_samples.csv \
  --k 32 \
  --score-col fitness \
  --threshold 0.0 \
  --direction higher
```

Dual-Reward ESR:

```bash
python analysis/metrics.py \
  --samples exports/amp_scored_samples.csv \
  --k 32 \
  --train-score-col apexmic_score \
  --independent-score-col heldout_mic_score \
  --train-threshold 0.5 \
  --independent-threshold 0.5 \
  --train-direction higher \
  --independent-direction higher
```

The output JSON reports base and RL pass@k, the preserved/expansion/shrinkage/
out-of-support counts, ESR, and delta-ESR for dual-reward runs.
