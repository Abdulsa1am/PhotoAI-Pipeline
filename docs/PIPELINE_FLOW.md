# Pipeline Flow

## Overview
PhotoAI processes images in phases:

1. Step 0: Setup CLIP (one-time model export)
2. Step 1: Face Extraction (required)
3. Step 2: Face Clustering (optional)
4. Step 3: Classify Images (optional)
5. Step 4: Build Archive (required)
6. Step 5: Compress Images (optional and separate)

## Mandatory vs Optional

| Step | Name | Required | If Skipped |
|---|---|---|---|
| 0 | Setup CLIP | Required once before Step 3 | Step 3 cannot run |
| 1 | Face Extraction | Required | No usable archive organization |
| 2 | Face Clustering | Optional | Face images route to fallback people folders |
| 3 | Classify Images | Optional | Non-face images route to Uncategorized |
| 4 | Build Archive | Required | No archive output |
| 5 | Compress Images | Optional | Archive stays uncompressed |

## Run Modes

### Mode A: Full pipeline (recommended first run)
`0 -> 1 -> 2 -> 3 -> 4`

- Best quality organization for both people and non-face content.
- Use this when processing a library for the first time.

### Mode B: Faces only
`0 -> 1 -> 2 -> 4`

- Skips semantic classification.
- Non-face images go to `Uncategorized`.

### Mode C: Fast fallback archive
`0 -> 1 -> 4`

- Skips clustering and classification.
- Face images go to fallback people folders.
- Non-face images go to `Uncategorized`.

### Mode D: Re-archive only
`4` only (after Step 1 exists)

- Rebuilds folders using new dynamic thresholds.
- No need to rerun extraction.

## Step 4 Fallback Behavior

Step 4 handles optional step gaps safely:

- If Step 2 is missing, faces are grouped into fallback people buckets.
- If Step 3 is missing, non-face images are written to `Uncategorized`.
- If Step 1 is missing, Step 4 stops with a fatal message because `faces` data is foundational.

## Rerun Guidance

- Changing static extraction settings (`Det Size`, `Det Thresh`) requires resetting the database before rerunning Step 1.
- Changing dynamic thresholds (`Min Cluster`, `Merge Threshold`, `Confidence`) only needs rerunning Step 4.

## Related Docs

- [Stability Guide](STABILITY_GUIDE.md)
- [Config Reference](CONFIG_REFERENCE.md)
- [Step 0 Details](PHASE_DETAILS/00-setup-clip.md)
- [Step 1 Details](PHASE_DETAILS/01-face-extraction.md)
- [Step 2 Details](PHASE_DETAILS/02-face-clustering.md)
- [Step 3 Details](PHASE_DETAILS/03-classify-images.md)
- [Step 4 Details](PHASE_DETAILS/04-build-archive.md)
- [Step 5 Details](PHASE_DETAILS/05-compress-images.md)
