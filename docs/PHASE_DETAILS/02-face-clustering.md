# Step 2: Face Clustering

## Purpose

Group extracted face embeddings into identity clusters.

## Inputs

- `faces` embeddings from Step 1
- clustering settings (`min_cluster_size`, `merge_threshold`)

## Output

- cluster labels and people centroids used by Step 4 people routing

## Optionality

- Optional for archive build.
- If skipped, Step 4 routes faces to fallback people folders.

## Tuning

- Raise `min_cluster_size` for stricter clusters.
- Lower `merge_threshold` for stricter centroid matching.

## Rerun Guidance

You can tune dynamic clustering behavior and rerun Step 4 without rerunning Step 1.
