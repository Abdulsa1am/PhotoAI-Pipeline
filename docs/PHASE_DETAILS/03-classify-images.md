# Step 3: Classify Images

## Purpose

Apply semantic classification to non-face images using CLIP-family models.

## Inputs

- non-face images from DB
- confidence_threshold
- clip_model_size

## Output

- classifications table entries with category and confidence

## Optionality

- Optional for archive build.
- If skipped, Step 4 places non-face files in Uncategorized.

## Runtime Model Selection (actual code)

- clip_model_size=base -> models/clip-vit-base-patch32-onnx
- clip_model_size=large -> models/clip-vit-large-patch14-onnx

## Scoring Path

- Runtime detects SigLIP only if model path contains siglip.
- Current base/large runtime paths are CLIP paths, so CLIP softmax path is used.

## Defaults in code

- Script default confidence_threshold: 0.18
- GUI default confidence_threshold: 0.15
- Effective value comes from pipeline_config.json when present.

## Tuning

- Lower confidence threshold for broader category assignment.
- Higher threshold for stricter, lower-noise categories.
