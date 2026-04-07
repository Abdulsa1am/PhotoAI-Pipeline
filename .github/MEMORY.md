# PhotoAI Memory Ledger

Use this file as the persistent continuity contract for future AI sessions.

## Update Rules

1. Keep entries short and factual.
2. Record decisions, not raw logs.
3. Prefer newest-first ordering.
4. When a previous assumption is wrong, mark it as superseded.

## Entry Template

- Date:
- Area:
- Change:
- Why:
- Validation:
- Follow-up:

## Current Baseline

- Step 1 includes diagnostics and DirectML reshape-error fallback logic.
- Step 4 supports missing classifications table (Step 3 optional).
- Trace logging writes latest run to `logs/last_run.log` and archives recent runs in `logs/history/`.

- Date: 2026-04-07
- Area: Step 3 SigLIP2 throughput
- Change: Added `ensemble_mode` toggle in Script 3; default `False` now uses primary-prompt-only classification (1 prompt/category) while `True` keeps full 4-prompt ensemble.
- Why: Reduce per-image text batch from 84 prompts to 21 prompts for large SigLIP2 speedup on DirectML.
- Validation: Static checks clean; runtime sanity check confirms 21 labels in fast mode and 84 in full mode.
- Follow-up: If higher precision is needed for hard classes, run with `ensemble_mode=true` selectively.

- Date: 2026-04-07
- Area: Vision model config naming
- Change: Renamed `clip_model_size` values to `clip` (Base) and `siglip2` (Ultra); added legacy normalization (`base->clip`, `large->siglip2`) in GUI load path and backward-compatible parsing in Script 3.
- Why: Make model selection names explicit and aligned with actual model families while preserving old config compatibility.
- Validation: Static checks clean; Script 3 resolves `siglip2` to the SigLIP2 ONNX directory.
- Follow-up: Optional: migrate older config files automatically on save to remove legacy values from disk.

- Date: 2026-04-07
- Area: Step 3 model loading
- Change: Added `clip_model_dir` config override in Script 3 as highest-priority model path and set default confidence threshold fallback to 0.25.
- Why: Ensure installed SigLIP2 ONNX directory can be selected directly from config instead of being blocked behind clip_model_size mapping.
- Validation: Static checks clean; runtime import confirms `CLIP_MODEL_DIR` resolves to SigLIP2 path and threshold resolves to 0.25.
- Follow-up: Confirm run log prints "Loading SigLIP 2 model from ONNX..." in next Script 3 execution.

- Date: 2026-04-07
- Area: Step 2 HDBSCAN runtime compatibility
- Change: Added a clusterer builder that forces HDBSCAN `algorithm='generic'` when metric is `cosine`.
- Why: hdbscan default BallTree path rejects cosine in current environment and aborts Phase 2 with "Unrecognized metric 'cosine'".
- Validation: Focused tests pass (7/7), full test suite passes (21/21), and static checks are clean.
- Follow-up: If runtime speed is insufficient at scale, benchmark alternatives while preserving cosine geometry.

- Date: 2026-04-07
- Area: Step 2 HDBSCAN dtype compatibility
- Change: Cast Phase 2 clustering input to float64 only for the cosine + generic HDBSCAN path before `fit_predict`.
- Why: hdbscan generic cosine path raised `ValueError: Buffer dtype mismatch, expected 'double_t' but got 'float'` with float32 embeddings.
- Validation: Added regression test for float64 input path; focused tests pass (8/8), full test suite passes (22/22), static checks clean.
- Follow-up: Keep this cast scoped to cosine+generic to avoid unnecessary memory growth for other metrics.

- Date: 2026-04-07
- Area: GUI reset database action
- Change: Updated Reset Database dialog/log messaging to be delete-only and removed the "Re-run Steps 1-4" guidance line.
- Why: Prevent confusion that reset triggers pipeline execution; reset now clearly performs immediate DB deletion only.
- Validation: Static check clean for app UI and reset text verified in `_reset_db`.
- Follow-up: Optional UX enhancement is a separate one-click "Reset + Run" action if desired.

- Date: 2026-04-07
- Area: Repository structure
- Change: Moved executable pipeline scripts into `scripts/` (`setup_clip_model.py`, steps 1-5), updated GUI script dispatch and tests to the new paths, and switched moved scripts to resolve `pipeline_config.json` and `models/` from project root.
- Why: Keep repository root clean and make script responsibilities easier to discover.
- Validation: Static analysis clean (`get_errors` reports no errors across app, scripts, and tests).
- Follow-up: Run GUI step smoke test once to confirm script launch paths in runtime environment.

- Date: 2026-04-07
- Area: Repository hygiene and layout
- Change: Expanded `.gitignore` for runtime/local artifacts, untracked generated files (`pipeline_config.json`, runtime logs, local debug DB), added `pipeline_config.example.json`, and documented folder intent in `docs/REPOSITORY_LAYOUT.md`.
- Why: Keep GitHub history clean, reproducible, and free from machine-specific noise.
- Validation: Verified git index updates and resulting status; no runtime code paths changed.
- Follow-up: If onboarding grows, add a short bootstrap script to generate local config from the example template.

- Date: 2026-04-07
- Area: Step 2 face clustering assignment strategy
- Change: Replaced greedy nearest-centroid matching with confidence-gated assignment (`ASSIGNED`/`REVIEW`/`NEW_IDENTITY`), added protected EMA centroid updates, persisted review queue table, and JSONL assignment audit logging.
- Why: Reduce irreversible false merges and centroid contamination; add deferred resolution path for ambiguous faces.
- Validation: Static validation passed (`get_errors` clean) and new unit tests added for gate outcomes and centroid update guard.
- Follow-up: Run replay evaluation on labeled holdout to calibrate thresholds and monitor review graduation rate.

- Date: 2026-04-07
- Area: Step 4 archive behavior
- Change: Added `_pending` archive routing for faces marked `assignment_status='REVIEW'`.
- Why: Keep unresolved assignments out of person folders while preserving backward compatibility.
- Validation: Code paths updated in label loading and directory mapping.
- Follow-up: Add/extend integration tests for archive routing with review-status rows.

## Open Operational Checks

- Confirm trace-history retention after 11+ runs in GUI execution.
- Keep static-vs-dynamic setting guidance aligned with runtime behavior.

## Post-Fix Action Required

The database photo_catalog.db was built with det_thresh=0.5 and
min_cluster_size=4. All centroids and cluster assignments in it are
unreliable. Before running the pipeline again, the user must:
1. Delete or rename photo_catalog.db
2. Rerun Script 1 (face extraction) with the new det_thresh=0.65
3. Rerun Script 2 (face clustering) with the new HDBSCAN parameters
Do not skip this step. Running Script 2 on top of the old database
will not fix the wrong-person-in-folder problem.

- Date: 2026-04-07
- Area: Step 2 clustering dimensionality
- Change: Removed Phase 2 PCA down-projection branch and now always cluster with full 512D normalized embeddings.
- Why: Preserve identity separation signal in high-dimensional embedding space and avoid projection-induced merges.
- Validation: Static analysis clean for Step 2 and full test suite passes (14/14).
- Follow-up: Monitor Phase 2 runtime on largest batches and tune scheduling if needed.

## Post-Accuracy-Fix Action Required

Bugs 1–5 have been fixed. The existing database photo_catalog.db
was built with the wrong HDBSCAN metric (euclidean), wrong cluster
selection method (eom), and 96D PCA-compressed embeddings. All
existing cluster assignments and centroids are unreliable.

Before running the pipeline again you must:
1. Delete or rename photo_catalog.db
2. Rerun Script 1 (face extraction) — this will now apply the
	min_face_area_px=1600 filter and only store quality faces
3. Rerun Script 2 (face clustering) — this will now cluster on
	full 512D cosine space with leaf selection

Do not skip the database reset. Running Script 2 on top of the
existing database will not fix the wrong-person-in-folder problem
because the contaminated centroids will persist.
