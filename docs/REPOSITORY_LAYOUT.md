# Repository Layout

This repository keeps executable pipeline code at the root and groups non-runtime assets under dedicated folders.

## Top-level files

- `app.py`: Tkinter GUI launcher and pipeline orchestrator.
- `pipeline_config.example.json`: Example config template for new machines.

## Directories

- `scripts/`: Executable pipeline step scripts (`setup` and steps 1 to 5).
- `docs/`: All project documentation.
- `docs/PHASE_DETAILS/`: Per-step operational details.
- `docs/plans/`: Design and improvement plans.
- `docs/reviews/`: Code review notes and reports.
- `tests/`: Automated test suite.
- `debug/`: Local debugging scripts and disposable local DB files.
- `logs/`: Runtime logs and run traces (kept out of git, except placeholders).
- `models/`: Local model exports/downloads (not tracked).

## Git hygiene rules

- Do not commit machine-local runtime state (`pipeline_config.json`, `photo_catalog.db`, logs).
- Keep large model binaries in `models/` only.
- Use `pipeline_config.example.json` as a baseline template for local setup.
