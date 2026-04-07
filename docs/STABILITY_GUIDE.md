# Stability Guide

## Bigger-Run Status

Current pipeline behavior is stable for larger runs with a GPU-first approach, with one known environment-specific limitation:

- Some DirectML + ONNX Runtime combinations can fail in Step 1 with `Reshape_223` / `80070057` when detection size differs from 640.

Step 1 now attempts to keep GPU usage practical and recover gracefully when that specific runtime failure is detected.

## GPU-First Step 1 Guidance

Recommended:

- `Det Size`: 1024 request is supported via GPU-safe mode in Step 1.
- `Det Thresh`: around 0.5 to 0.7 depending on strictness needed.

What happens:

1. Step 1 initializes DirectML + CPU providers.
2. For high det-size requests, it uses a GPU-safe path designed to avoid known DirectML reshape instability.
3. If the known reshape runtime exception appears, it retries on CPU for completion.

## Large-Run Checklist

1. Confirm source directory exists and is readable.
2. Confirm output drive has enough free space.
3. Keep `logs/last_run.log` open for the current run trace.
4. Use `logs/history/` to compare up to 10 recent runs.
5. If you change static extraction settings, reset DB first.

## Failure Signatures and Actions

### Signature: `Reshape_223` with `80070057`

Action:

1. Allow automatic fallback to CPU for completion.
2. Keep the trace file for diagnosis.
3. If needed, re-run with det-size 640 baseline for maximum runtime compatibility.

### Signature: all images fail, zero faces

Action:

1. Validate source path and sample image readability.
2. Check provider initialization lines in trace.
3. Confirm the run is not blocked by invalid config values.

## Optional-Step Effects

- Skipping Step 2 does not break Step 4.
- Skipping Step 3 does not break Step 4.
- Skipping Step 1 does break Step 4 (required data missing).

## Operational Recommendation

For very large datasets, do one medium-size validation run first, then launch the full run with unchanged static extraction settings to preserve resumability.
