# Step 4: Build Archive

## Purpose

Assemble the final folder structure from DB outputs.

## Inputs

- required: `faces` data from Step 1
- optional: clustering and classification results
- dynamic thresholds at runtime

## Output

- archive folder versioned under output directory

## Optional-Step Compatibility

- Works when Step 2 is skipped (fallback face grouping)
- Works when Step 3 is skipped (non-face to `Uncategorized`)
- Fails only if Step 1 data is missing

## Dynamic Behavior

This step evaluates runtime thresholds each run, so you can re-archive quickly after parameter changes.

## Practical Use

Use Step 4-only reruns to iterate archive quality without paying extraction cost again.
