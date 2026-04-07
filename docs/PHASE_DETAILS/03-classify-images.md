# Step 3: Classify Images

## Purpose

Apply semantic classification to non-face images using CLIP-family models.

## Inputs

- non-face images from DB
- classification settings (`confidence_threshold`, model size)

## Output

- `classifications` table entries with category and confidence

## Optionality

- Optional for archive build.
- If skipped, Step 4 places non-face files in `Uncategorized`.

## Tuning

- Lower confidence threshold for broader category assignment.
- Higher threshold for stricter, lower-noise categories.

## Model Choice

- Base: faster, lower resource usage
- Large/Ultra: better semantic quality, slower runtime
