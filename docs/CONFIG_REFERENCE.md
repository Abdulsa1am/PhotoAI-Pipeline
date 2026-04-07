# Config Reference

## Static vs Dynamic Settings

## Static (requires DB reset)

These affect raw extraction output. Change them only before Step 1 or after resetting the DB:

- `det_size`
- `det_thresh`
- source directory selection (if changing dataset scope)

If changed after extraction, reset the database and rerun Step 1.

## Dynamic (Step 4 re-archive only)

These can be changed and then applied by rerunning Step 4:

- `min_cluster_size`
- `merge_threshold`
- `confidence_threshold`

## Compression Settings

Step 5 compression settings are independent from steps 0-4 and can be rerun without touching extraction data.

## Recommended Baselines

- Extraction: `det_size=640` for maximum compatibility, or your tuned higher setting if your environment is stable.
- Clustering: `min_cluster_size` around 4-7 for noisy libraries.
- Classification: script default is 0.18, GUI default is 0.15; tune from 0.15 upward based on your noise tolerance.

## Safety Rules

1. Do not mix static-setting changes with partial reruns.
2. Keep one archive build per threshold experiment.
3. Use trace history to compare behavior between runs.

