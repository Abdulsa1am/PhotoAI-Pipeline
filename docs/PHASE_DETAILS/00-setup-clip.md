# Step 0: Setup CLIP

## Purpose

Export an ONNX image-classification model used by Step 3.

## Inputs

- pipeline_config.json key: clip_model_size
- local models directory

## Output (actual code behavior)

- clip_model_size=base:
  - model id: openai/clip-vit-base-patch32
  - output dir: models/clip-vit-base-patch32-onnx
- clip_model_size=large:
  - model id: google/siglip2-so400m-patch14-384
  - output dir: models/siglip2-so400m-patch14-384-onnx

## Important Current Mismatch

- Step 0 exports SigLIP2 for clip_model_size=large.
- Step 3 currently loads models/clip-vit-large-patch14-onnx when clip_model_size=large.
- This mismatch is in code and should be resolved in code in a later pass.

## When to Run

- First-time setup
- After changing clip_model_size
- After deleting model.onnx files

## Notes

- Not required for Step 1 or Step 2.
- Full pipeline runs Step 0 before Step 3.
