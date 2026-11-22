# Project structure

This tree is organized for a focused research implementation of UniverSeg, not a general-purpose segmentation package. It separates the from-scratch model implementation, experiment pipeline, dataset preparation, validation, and research documentation.

The repository separates data preparation, model implementation, experiment execution, dataset auditing, and documentation. Dataset files and generated artifacts remain local and are not committed.

```text
universeg-minimedical/
│
├── README.md
├── DATASETS.md
├── LICENSE
├── NOTICE
├── .gitignore
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
│
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
│
├── preprocessing/
│   ├── __init__.py
│   ├── 01_preprocess_acdc.py
│   ├── 02_preprocess_busi.py
│   ├── 03_preprocess_wbc.py
│   ├── 04_preprocess_pandental.py
│   └── 05_merge_manifests.py
│
├── training/
│   ├── __init__.py
│   ├── episodic_dataset.py
│   ├── universeg_core.py
│   ├── universeg_losses.py
│   ├── universeg_model_v1.py
│   ├── 06_validate_episodic_pipeline.py
│   ├── 07_validate_crossconv.py
│   ├── 08_validate_crossblock.py
│   ├── 09_validate_universeg_model.py
│   ├── 10_validate_loss_training_step.py
│   ├── 11_train_universeg_cpu.py
│   └── 12_evaluate_heldout_universeg.py
│
├── dataset_inspection/
│   ├── __init__.py
│   ├── 02b_audit_busi.py
│   ├── 03_inspect_acdc.py
│   ├── 03b_inspect_acdc_h5.py
│   ├── 05_inspect_pandental.py
│   └── 05b_analyze_pandental_annotations.py
│
├── notebooks/
│   └── official_implementation_validation.ipynb
│
├── tests/
│   ├── conftest.py
│   └── test_model_components.py
│
├── outputs/
│   ├── .gitkeep
│   ├── training/
│   │   └── .gitkeep
│   └── evaluation/
│       └── .gitkeep
│
└── docs/
    ├── PROJECT_STRUCTURE.md
    ├── PREPARATION_NOTES.md
    ├── RESEARCH_SCOPE.md
    ├── LICENSING.md
    └── RELEASE_CHECKLIST.md
```

## What each section contains

- **`preprocessing/`** — one adapter per dataset plus the script that validates and merges the dataset manifests.
- **`training/`** — episodic data loading, CrossConv/CrossBlock components, the reduced UniverSeg model, losses, staged validation, CPU training, and held-out evaluation.
- **`dataset_inspection/`** — optional source-data audit tools. They are useful for checking the input datasets but are not required for every training run.
- **`notebooks/`** — an optional comparison against the public reference implementation. It is separate from the from-scratch model implementation.
- **`tests/`** — lightweight, data-independent tests for core model components and metrics.
- **`data/raw/` and `data/processed/`** — local-only dataset inputs and generated images, masks, and manifests.
- **`outputs/`** — local-only checkpoints, training history, predictions, probability maps, and evaluation reports.
- **`docs/`** — repository preparation notes, release checks, and this structure guide.

## Main execution order

1. Obtain the four datasets separately and place them under `data/raw/`.
2. Run the five preprocessing scripts in numerical order.
3. Run the staged validation scripts in `training/`.
4. Run `11_train_universeg_cpu.py`.
5. Run `12_evaluate_heldout_universeg.py`.
6. Run `python -m pytest -q` for the data-independent tests.
