# Step 0: Setup CLIP

## Purpose

Exports the selected CLIP model to ONNX so Step 3 can run efficiently.

## Inputs

- `pipeline_config.json` model preference
- local models directory

## Output

- ONNX model files under `models/`

## When to Run

- First-time setup
- After changing to a different model family that needs export

## Notes

- This is not required for Step 1 or Step 2.
- Full pipeline includes this step before classification.
