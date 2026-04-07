# PhotoAI Agent Context

## Mission

Keep PhotoAI runs stable, explainable, and resumable across long sessions.

## Always Track

1. What changed this session
2. What is currently in progress
3. What is blocked and why
4. What was validated versus assumed

## Runtime Priorities

1. Prefer GPU-first behavior for Step 1, but preserve completion fallback paths.
2. Treat Step 2 and Step 3 as optional for Step 4 compatibility.
3. Preserve user data and avoid destructive operations.

## Repo Facts

- Full pipeline button runs steps 0 through 4.
- Step 5 is separate optional compression.
- Latest trace file is `logs/last_run.log`.
- Trace history is in `logs/history/` and should retain the latest 10 runs.

## Working Agreement

1. Before edits, review current traces and changed files.
2. After edits, run quick validation and summarize outcomes.
3. When behavior changes, update docs and memory notes in the same pass.

## Required End-of-Task Output

Include these sections in final responses:

1. What changed
2. Validation results
3. Remaining risks
4. Next recommended action
