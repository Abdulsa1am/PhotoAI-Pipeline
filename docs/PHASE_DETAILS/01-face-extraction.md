# Step 1: Face Extraction

## Purpose

Detect faces and compute embeddings for each image, then persist results in SQLite.

## Inputs

- source directory images
- extraction settings (`det_size`, `det_thresh`)

## Output

- `images` table rows
- `faces` table rows with embedding, bbox, and score

## Runtime

- GPU-first via DirectML when available
- CPU fallback path for known DirectML reshape runtime failures

## Important Rules

- This step is required before archive build.
- If you change static extraction settings, reset DB and rerun Step 1.
- Step 1 is resumable only when static settings stay unchanged.

## Diagnostics

Step 1 prints effective config, provider info, and traceback samples for early failures. Use `logs/last_run.log` for exact run context.
