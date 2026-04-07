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
