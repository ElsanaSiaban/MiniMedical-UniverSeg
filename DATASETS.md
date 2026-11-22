# Datasets used in this project

This repository contains **source code and documentation only**. It does not distribute the original datasets, processed images, segmentation masks, subject-level manifests, predictions, or model checkpoints. Each dataset must be obtained separately from its source.

## Dataset sources and roles

| Dataset | Source used | Role in this project | Tasks |
|---|---|---|---|
| ACDC (Automated Cardiac Diagnosis Challenge) | [Kaggle: ACDC Dataset](https://www.kaggle.com/datasets/anhoangvo/acdc-dataset) · [ACDC Challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Training | Left ventricle, myocardium, and right ventricle segmentation |
| BUSI (Breast Ultrasound Images) | [Kaggle: Dataset BUSI with Ground Truth](https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt) | Training | All-lesion, benign-lesion, and malignant-lesion segmentation |
| WBC (White Blood Cell) | [GitHub: zxaoyou/segmentation_WBC](https://github.com/zxaoyou/segmentation_WBC) | Dataset 1: training; Dataset 2: held-out source subset | Nucleus and whole-cell segmentation |
| PanDental (Panoramic Dental X-rays) | [Kaggle: Dental X-ray TFRecords](https://www.kaggle.com/datasets/daverattan/dental-xrary-tfrecords) | Held-out domain evaluation | Mandible segmentation |

## Important experimental notes

- ACDC, BUSI, and WBC Dataset 1 are used to form the training tasks.
- WBC Dataset 2 is held out by source subset. However, the same microscopy modality and nucleus/whole-cell task types are represented by WBC Dataset 1 during training. It is therefore **not** a fully unseen task or modality.
- PanDental is used as the held-out domain evaluation source. Segmentation 1 is the primary reference mask; Segmentation 2 is retained locally for inter-annotator comparison.
- The project is a reduced, CPU-friendly, from-scratch methodological reproduction. Its training collection and evaluation protocol differ from the original UniverSeg paper; results should not be described as a direct numerical reproduction of that paper.
- STARE appears only in a leftover unlabeled index in the supplied working archive. It is not part of the merged training/held-out manifest or the quantitative evaluation pipeline, so it is not listed as a dataset used in the main experiment.

## Expected local dataset layout

Place the separately obtained source datasets under `data/raw/` using these folder names (or adjust the corresponding script arguments where available):

```text
data/raw/
├── ACDC/
│   ├── ACDC_training_volumes/
│   └── ACDC_testing_volumes/
├── Dataset_BUSI_with_GT/
│   ├── benign/
│   ├── malignant/
│   └── normal/
├── segmentation_WBC-master/
│   ├── Dataset 1/
│   └── Dataset 2/
└── Panoramic_Dental_Xrays/
    ├── Images/
    ├── Segmentation1/
    └── Segmentation2/
```

The preprocessing scripts write generated files and manifests under `data/processed/`. These local data directories are excluded from Git by `.gitignore`.

## Dataset attribution and citations

The sources above identify the **specific dataset download locations** used for the experiments. Please cite the associated original publications when publishing scientific results. The dataset citations are separate from the [UniverSeg paper attribution](README.md#original-research-and-inspiration).


Please cite the dataset sources and their associated publications in any paper, report, or public result that uses them. In particular:

- **ACDC:** Bernard O, Lalande A, Zotti C, et al. (2018). *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?* IEEE Transactions on Medical Imaging, 37(11), 2514–2525. https://doi.org/10.1109/TMI.2018.2837502
- **BUSI:** Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. (2020). *Dataset of breast ultrasound images*. Data in Brief, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863
- **WBC:** Zheng X, Wang Y, Wang G, Liu J. (2018). *Fast and Robust Segmentation of White Blood Cell Images by Self-supervised Learning*. Micron, 107, 55–71. https://doi.org/10.1016/j.micron.2018.01.010
- **PanDental:** [Panoramic Dental Xray Dataset (daverattan, Kaggle)](https://www.kaggle.com/datasets/daverattan/dental-xrary-tfrecords). This is the verified download/source link for the experiment. A definitive primary research citation for this exact Kaggle upload was **not confirmed**, so no publication is attributed to it here without evidence. If the dataset uploader provides a recommended paper citation, add it after verification.

The dataset links above identify the sources used; they do not imply that this repository redistributes or relicenses those datasets.
