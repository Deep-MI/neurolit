# Repository Guidelines

## Repository Overview
This repository contains `neurolit`, a lesion inpainting tool for brain MRI. It is designed for standalone use and for integration into the FastSurfer neuroimaging toolbox, where inpainted images can be used for downstream whole-brain segmentation and related postprocessing workflows.

## Project Structure & Module Organization
Core package code lives in `neurolit/`. Key areas are `data/` for preprocessing and datasets, `networks/` for model definitions, `postprocessing/` for lesion integration utilities and config JSON files, `scripts/` for CLI entrypoints and container wrappers, and `utils/` for logging, plotting, and checkpoint download helpers. Tests are colocated in `neurolit/tests/`. Documentation sources live in `doc/`, Docker assets in `containerization/`, and generated outputs such as `build/` and `doc-build/` should not be edited manually.

## Build, Test, and Development Commands
Use Python 3.10+.

- `python -m pip install -e .[test,style]` installs the package with test and lint tools.
- `pytest neurolit` runs the test suite used in CI.
- `ruff check .` runs the primary linter and import-order checks.
- `pydocstyle .` validates NumPy-style docstrings for `neurolit/`.
- `codespell` catches spelling issues in code, docs, and filenames.
- `python -m build` creates source and wheel distributions.
- `python -m pip install .[doc] && sphinx-build ./doc ./doc-build/dev -W --keep-going` builds documentation locally.

## Coding Style & Naming Conventions
Follow existing Python conventions: 4-space indentation, snake_case for functions/modules, PascalCase for classes, and descriptive CLI names such as `lit-inpainting` and `lit-postprocessing`. Ruff enforces import sorting and core lint rules; the configured line length is 150. Prefer NumPy-style docstrings for public functions and keep package-facing APIs inside `neurolit/` rather than top-level scripts.

## Testing Guidelines
Add tests under `neurolit/tests/` and name files `test_*.py`. Keep tests focused on CLI behavior, data transforms, and postprocessing logic. Run `pytest neurolit` before opening a PR; CI executes the suite across Python 3.10, 3.11, and 3.12 on Linux, macOS, and Windows. Coverage is configured, but no explicit threshold is enforced here, so add tests for new behavior and regressions.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages such as `removed unused import`, `spelling fix`, and `improve CLI tools, versioning, documentation, docker`. Keep commits small and specific. PRs should include a concise description, linked issue when applicable, test evidence (`pytest`, `ruff`, docs build), and screenshots or command examples for CLI or documentation changes.

## Security & Configuration Tips
Do not commit model weights, local virtual environments, or generated artifacts. Large checkpoints are downloaded via `lit-download-models`; keep paths and credentials out of source control. When editing container or postprocessing flows, verify assumptions against `neurolit/scripts/run_lit_containerized.sh` and the JSON configs in `neurolit/postprocessing/`.
