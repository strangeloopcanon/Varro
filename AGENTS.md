# Repository Guidelines

## Project Structure & Module Organization
- Core code lives at the top level with orchestration scripts: `run_daily_pipeline.py`, `run_gspo_training.py`. Helper utilities are under `scripts/` (e.g., `scripts/manage_models.py`).
- Modules:
  - `data_collection/` (RSS ingest, timestamped IO), `prediction_generation/` (rollouts), `outcome_tracking/` (tracking + evaluation), `training/` (checkpoints, stats).
  - `config/` holds JSON configs (sources, templates, training params).
  - Date-partitioned artifacts: `timestamped_storage*/`, `archive/`, and reports under `reports/`.

## Build, Test, and Development Commands
- Setup environment:
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Run daily pipeline:
  - `python run_daily_pipeline.py --mode full` (morning → evening → night)
  - `python run_daily_pipeline.py --mode morning --date 20250807 --seed 1234`
  - `python run_daily_pipeline.py --mode evening --date 20250807`
- Train GSPO:
  - `python run_gspo_training.py --epochs 1 --learning_rate 5e-7`
  - `python run_gspo_training.py --load_checkpoint training/checkpoints/gspo/final_model`
- Manage models:
  - `python scripts/manage_models.py --list` | `--info <name>` | `--archive 7`

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indent; UTF-8.
- Names: `snake_case` for files/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints and f-strings; use `logging` (no print in libraries).
- Docstrings: module/class/function triple-quoted summaries + args/returns where helpful.

## Testing Guidelines
- Current repo has no test suite. Prefer `pytest` with:
  - Location: `tests/` alongside code; files `test_*.py`.
  - Style: Arrange-Act-Assert; small, deterministic tests.
  - Quick start (once added): `pytest -q`; aim for meaningful coverage of data flows.

## Commit & Pull Request Guidelines
- Use Conventional Commits (seen in history): `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`.
  - Example: `feat: add composite reward to outcome evaluation`
- PRs: focused scope, descriptive summary, reproduction steps/commands, linked issues, and sample output paths (e.g., `timestamped_storage/20250807/...`).
- Keep diffs minimal; update README/config examples when behavior changes.

## Security & Configuration Tips
- Do not commit large generated artifacts; rely on existing `.gitignore` and use `archive/` for aging data.
- Store tunables in `config/*.json`; avoid hardcoding credentials or URLs in code.
