---
description: "Use when working in PhotoAI to keep continuity of what happened, what is happening, and what still needs verification."
applyTo: "**/*"
---

# Memory Tracking Instruction

On every meaningful code change or debugging run:

1. Capture current state before edits.
2. Record outcome after edits.
3. Distinguish validated results from assumptions.

## Mandatory Checklist

1. Check changed files before editing to avoid conflicts.
2. Reference latest run trace when diagnosing runtime issues.
3. Update `.github/MEMORY.md` when behavior, decision, or policy changes.
4. Keep notes concise and remove stale assumptions.

## What to Record

- Decision made and rationale
- Runtime behavior change
- New fallback or error-handling behavior
- Validation evidence and gaps

## What Not to Record

- Sensitive data
- Huge raw logs
- Duplicated notes already superseded
