# Repository preparation notes

## Included and reorganized

- Five dataset preprocessing scripts are in `preprocessing/`.
- The episodic dataset loader, active CrossConv/CrossBlock core, losses, model, validation scripts, training script, and held-out evaluator are in `training/`.
- Five optional source-data inspection/audit scripts are in `dataset_inspection/`.
- The official implementation comparison notebook is in `notebooks/`; its saved outputs have been cleared so no example medical image is embedded in the notebook file.
- Added README and dataset source documentation, dependency lists, `.gitignore`, release checklist, project structure guide, and lightweight tests.

## Removed as unnecessary or unsuitable for the public source repository

- `__pycache__/` directories and `.pyc` bytecode files.
- `universeg_core previous.py`, an older duplicate CrossConv implementation. The active model uses the newer `universeg_core.py` implementation.
- `manifest_wbc before.csv`, an outdated intermediate manifest. The final WBC manifest superseded it.
- `history(2).csv`, an exact duplicate of `outputs/training/cpu_main/history.csv`.
- The Windows Explorer screenshot `dataset list.png`.
- `UniverSeg_MiniMedical.pdf`, which contains personal and university-course information. Keep the original report privately; prepare a redacted version separately if it is needed publicly.
- Raw and processed images/masks, subject-level manifests, generated predictions/probability maps, checkpoints, and run outputs. These are local research artifacts, not source code, and are excluded from this code-only repository.

## Code adjustments

- Removed fixed `H:\1--MiniMedical` project-root defaults. Scripts resolve the repository root from their own location or accept `--project-root`.
- Standardized local input data under `data/raw/` and generated files/manifests under `data/processed/`.
- Added a configurable project-root argument to BUSI preprocessing.
- Updated imports to use the active core module and fixed the unresolved `universeg_core` module reference in the CrossConv validation script.

## What remains useful but optional

- `dataset_inspection/` scripts are diagnostic utilities; they are not needed for normal training after the data layout is confirmed.
- `notebooks/official_implementation_validation.ipynb` is a separate reference comparison notebook, not part of the from-scratch training pipeline.
- The ACDC HDF5 inspection utility is only needed if the downloaded source includes HDF5 files.

## Checks

- Python compilation completed for the reorganized source tree.
- Three data-independent pytest smoke tests passed.
- The CrossConv implementation check passed on CPU.
- Full data-backed preprocessing, training, and held-out evaluation were not rerun because the dataset files and processed manifests are intentionally excluded.
