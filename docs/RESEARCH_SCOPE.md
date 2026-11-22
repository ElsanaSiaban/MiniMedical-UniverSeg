# Research scope and implementation notes

## Project identity

**UniverSeg MiniMedical** is a research project demonstrating an independent, from-scratch PyTorch implementation of the core UniverSeg-style architecture in a reduced multi-dataset medical image segmentation experiment.

It is not intended to be a general-purpose library or a clinically deployable application.

## Original research attribution

The project's research direction and CrossBlock/support-query concepts were inspired by **Butoi et al. (2023), _UniverSeg: Universal Medical Image Segmentation_**, ICCV 2023, and the [official JJGO/UniverSeg implementation](https://github.com/JJGO/UniverSeg). This repository implements a reduced experimental version independently; it is not the official code or an exact reproduction of the published experimental results. See the `README.md` for the full paper citation.

## Implementation components

The main implementation is contained in `training/`:

- `universeg_core.py`: CrossConv and CrossBlock components that perform feature interaction between query and support examples.
- `universeg_model_v1.py`: the encoder-decoder model assembled from the project's CrossBlocks.
- `episodic_dataset.py`: support-query episode construction and data access.
- `universeg_losses.py`: segmentation losses and validity-mask-aware calculations.
- `11_train_universeg_cpu.py`: the configured CPU-oriented training workflow.
- `12_evaluate_heldout_universeg.py`: evaluation on the selected held-out source subset and domain.

The model implementation is written in this repository using PyTorch. The public upstream UniverSeg code is used only by the separate reference-comparison notebook, not as an imported dependency of the main model implementation.

## Dataset roles in this experiment

- ACDC: training tasks for cardiac structure segmentation.
- BUSI: training tasks for breast ultrasound lesion segmentation.
- WBC Dataset 1: training tasks for nucleus and whole-cell segmentation.
- WBC Dataset 2: held-out source subset for evaluation.
- PanDental: held-out domain evaluation for mandible segmentation.

See `DATASETS.md` for source links, citations, and local folder conventions. Dataset files are not distributed by this repository.

## Claims this repository can make

It is appropriate to describe this work as:

- An independent, from-scratch PyTorch implementation of the core UniverSeg-style architecture.
- A reduced, CPU-friendly research implementation.
- An experimental evaluation across the listed medical imaging datasets.
- A project with a separate comparison notebook for the public reference implementation.

## Claims that require additional evidence

Do not claim any of the following without further experiments and documentation:

- Exact reproduction of the original UniverSeg paper or its reported metrics.
- State-of-the-art performance.
- Generalization to all medical imaging modalities or segmentation tasks.
- Fully unseen WBC task generalization, because WBC Dataset 1 and Dataset 2 share task types and modality.
- Clinical validity, diagnostic utility, or suitability for patient care.

## Recommended reporting checklist

When reporting results, document:

1. Exact dataset versions and source links.
2. Inclusion/exclusion criteria and patient/source-level split strategy.
3. Preprocessing and image/mask resizing details.
4. Model configuration and parameter count.
5. Training episodes, support/query sizes, optimizer, learning rate, epochs, and random seeds.
6. Hardware, runtime, and software versions.
7. Per-dataset and per-task metrics, including the number of evaluated cases.
8. Whether the reported score is per-image, per-task, macro-averaged, or pooled.
9. Limitations, possible data leakage, and the distinction between held-out source subset and unseen-domain evaluation.

## Clinical and privacy limitations

The project is for research and educational use only. It is not a medical device and must not be used for diagnosis, treatment decisions, or clinical workflows. Do not commit identifiable or sensitive patient information, source metadata, or private dataset manifests.
