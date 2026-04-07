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

## Open Operational Checks

- Confirm trace-history retention after 11+ runs in GUI execution.
- Keep static-vs-dynamic setting guidance aligned with runtime behavior.
