# Raw datasets (local only)

Place your legally obtained source datasets in this directory. The dataset folders expected by the preprocessing scripts are:

- `ACDC/` — HDF5 ACDC files, including `ACDC_training_volumes/` and `ACDC_testing_volumes/`.
- `Dataset_BUSI_with_GT/` — BUSI data with `benign/`, `malignant/`, and `normal/` subfolders.
- `segmentation_WBC-master/` — WBC source data with `Dataset 1/` and `Dataset 2/`.
- `Panoramic_Dental_Xrays/` — PanDental data with `Images/`, `Segmentation1/`, and `Segmentation2/`.

Do not commit source images, masks, DICOM/HDF5 files, or other dataset contents unless the relevant dataset license explicitly permits public redistribution. This folder is ignored by Git except for this README.
