# UniverSeg MiniMedical: A From-Scratch Research Implementation

An independent, from-scratch PyTorch implementation of the core UniverSeg-style architecture, developed as a focused medical-image-segmentation research project. The project explores episodic support-query learning across multiple medical imaging datasets using a reduced, CPU-friendly experimental configuration.

This repository is intended to document the implementation, data preparation, experiments, validation procedures, and held-out evaluation. **It is not presented as a general-purpose segmentation framework, a clinically validated system, or an exact reproduction of every experiment in the original UniverSeg paper.**

> **Research-use notice:** This is an experimental research implementation, not a clinically validated medical device. It must not be used for diagnosis, treatment decisions, or other clinical purposes.

## Original research and inspiration

This research project was **inspired by the original UniverSeg work and its official implementation**, [JJGO/UniverSeg](https://github.com/JJGO/UniverSeg). The original authors introduced UniverSeg's support-query approach and CrossBlock architecture for universal medical image segmentation.

**Original paper:** Victor Ion Butoi, Jose Javier Gonzalez Ortiz, Tianyu Ma, Mert R. Sabuncu, John Guttag, and Adrian V. Dalca (2023). *UniverSeg: Universal Medical Image Segmentation*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 21438–21451. [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Butoi_UniverSeg_Universal_Medical_Image_Segmentation_ICCV_2023_paper.html) · [DOI](https://doi.org/10.1109/ICCV51070.2023.01960) · [Official code](https://github.com/JJGO/UniverSeg).

**Relationship to the original project:** The model components and research pipeline in this repository were developed as an independent, from-scratch educational/research implementation based on the ideas in the published work. This repository is **not the official UniverSeg implementation**, does not claim affiliation with or endorsement by the original authors, and does not distribute their pretrained model weights. A separate notebook under `notebooks/` uses the official upstream repository for reference comparison, not as the main model implementation. The training datasets and experimental protocol also differ from the original publication.

## Research objective

The objective is to implement the central UniverSeg learning approach from scratch and investigate its behavior on a small multi-dataset medical image segmentation setting.

The implementation includes:

- Pairwise query-support feature interaction through CrossConv operations.
- CrossBlock building blocks.
- An encoder-decoder segmentation model.
- Episodic support-query data preparation.
- Segmentation losses and validity-mask handling.
- Staged component validation, CPU-oriented training, and held-out evaluation.

The model code is implemented independently in this repository using PyTorch. The separate notebook under `notebooks/` is used for reference comparison with the public upstream implementation; it is not imported as the model implementation used by the training scripts.

## Experimental scope

The current experiment uses four dataset sources:

| Dataset and source | Experimental role | Main segmentation tasks |
|---|---|---|
| [ACDC](https://www.kaggle.com/datasets/anhoangvo/acdc-dataset) ([official challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/)) | Training | Left ventricle, myocardium, right ventricle |
| [BUSI](https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt) | Training | Lesion segmentation, including benign and malignant cases |
| [WBC Dataset 1](https://github.com/zxaoyou/segmentation_WBC) | Training | Nucleus and whole-cell segmentation |
| [WBC Dataset 2](https://github.com/zxaoyou/segmentation_WBC) | Held-out source subset | Nucleus and whole-cell segmentation |
| [PanDental](https://www.kaggle.com/datasets/daverattan/dental-xrary-tfrecords) | Held-out domain evaluation | Mandible segmentation |

The cited dataset papers for ACDC, BUSI, and WBC, along with all four source links, are documented in [DATASETS.md](DATASETS.md). For PanDental, the Kaggle dataset link is given as the source; a definitive original-paper citation for this specific upload has not been confirmed. **No original or processed dataset files are distributed in this repository.** Users must obtain the datasets independently and follow the terms supplied by their respective sources.

### Interpretation and limitations

- This is a reduced, CPU-friendly research configuration. It should not be described as a direct numerical reproduction of the original UniverSeg paper.
- ACDC is used for training in this project; this differs from the original paper's experimental setup.
- WBC Dataset 2 is held out at the source-subset level, but WBC Dataset 1 exposes the model to the same microscopy modality and segmentation task types. It is therefore not a fully unseen task or modality.
- PanDental is used as a held-out domain evaluation source.
- The project results should be interpreted in light of the dataset composition, preprocessing choices, split strategy, training budget, and hardware constraints.
- The repository does not claim clinical readiness or diagnostic performance.

For the full research scope and reporting guidance, see [docs/RESEARCH_SCOPE.md](docs/RESEARCH_SCOPE.md).

## Repository structure

```text
universeg-minimedical/
├── README.md
├── DATASETS.md
├── LICENSE
├── NOTICE
├── .gitignore
├── requirements.txt
├── requirements-optional.txt
├── requirements-dev.txt
├── preprocessing/          # Dataset preprocessing adapters and manifest merge
├── training/               # From-scratch model, episodic data, validation, training, evaluation
├── dataset_inspection/     # Dataset audit and annotation inspection utilities
├── notebooks/              # Separate comparison with the public reference implementation
├── data/
│   ├── raw/                # Local source datasets (not committed)
│   └── processed/          # Generated images, masks, manifests (not committed)
├── outputs/                # Local checkpoints, metrics, and predictions (not committed)
├── tests/                  # Lightweight model-component tests
└── docs/                   # Research scope, project structure, and release notes
```

See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the complete file tree and the purpose of each directory.

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer (below 3.0)
- NumPy, Pillow, h5py, and SciPy
- Optional: nibabel and pydicom for selected source-inspection utilities; JupyterLab and Matplotlib for notebooks

### 1. Create an environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch and project dependencies

Install the PyTorch build appropriate for your machine (CPU or CUDA) using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then install the remaining project dependencies:

```bash
python -m pip install -r requirements.txt
```

For dataset-inspection utilities and notebooks:

```bash
python -m pip install -r requirements-optional.txt
```

For development and linting:

```bash
python -m pip install -r requirements-dev.txt
```

## Dataset preparation

1. Obtain each dataset separately from its source and review the source's current access and usage terms.
2. Place the source folders under `data/raw/` using the layout in [DATASETS.md](DATASETS.md).
3. Run each preprocessing adapter from the repository root:

```bash
python preprocessing/01_preprocess_acdc.py --project-root .
python preprocessing/02_preprocess_busi.py --project-root .
python preprocessing/03_preprocess_wbc.py --project-root .
python preprocessing/04_preprocess_pandental.py --project-root .
python preprocessing/05_merge_manifests.py --project-root .
```

The scripts write generated images, masks, validity masks, manifests, and summaries to `data/processed/`. Some adapters support `--overwrite`; inspect each script's `--help` output before regenerating existing data.

## Validation, training, and evaluation

Run the independent CrossConv implementation check first:

```bash
python training/07_validate_crossconv.py --project-root .
```

The remaining pipeline and model validations require the processed data and merged manifest:

```bash
python training/06_validate_episodic_pipeline.py --project-root .
python training/08_validate_crossblock.py --project-root .
python training/09_validate_universeg_model.py --project-root .
python training/10_validate_loss_training_step.py --project-root .
```

Run a short end-to-end CPU smoke test before a full training run:

```bash
python training/11_train_universeg_cpu.py --project-root . --smoke-test
```

Then run the configured training job:

```bash
python training/11_train_universeg_cpu.py --project-root .
```

Evaluate the selected checkpoint on held-out tasks/domains:

```bash
python training/12_evaluate_heldout_universeg.py --project-root .
```

Training checkpoints and evaluation outputs are written under `outputs/`. They are excluded from Git by default. Review each script's `--help` for additional settings, resume options, and overwrite behavior.

Run the lightweight, data-independent unit tests with:

```bash
python -m pytest -q
```

## Reference implementation comparison

`notebooks/official_implementation_validation.ipynb` is a separate reference-comparison notebook. It requires internet access to obtain the upstream [JJGO/UniverSeg](https://github.com/JJGO/UniverSeg) repository and optional notebook dependencies. The main training code in `training/` uses this project's own implementation.

## Data, privacy, and model-weight policy

This repository contains source code and documentation only. It does not include raw or processed medical images, masks, subject-level manifests, model checkpoints, or generated predictions. Dataset availability online does not automatically grant redistribution rights. Do not add patient/subject-level information, DICOM metadata, private source paths, or trained weights unless you have reviewed the applicable privacy, sharing, and licensing requirements.


## Acknowledgements and references

Credit for the original UniverSeg architecture and research belongs to Butoi et al. (ICCV 2023) and the [JJGO/UniverSeg](https://github.com/JJGO/UniverSeg) team. Please cite their paper when referencing the approach. This repository is an independent, reduced-scale research implementation, not an official release.

The data sources and publications are documented in [DATASETS.md](DATASETS.md): [ACDC](https://www.kaggle.com/datasets/anhoangvo/acdc-dataset), [BUSI](https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt), [WBC](https://github.com/zxaoyou/segmentation_WBC), and [PanDental](https://www.kaggle.com/datasets/daverattan/dental-xrary-tfrecords). All datasets must be obtained independently.
