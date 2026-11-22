# Processed data (local only)

Preprocessing scripts write the processed 128×128 images, masks, validity masks, manifests, and summaries here. The training and evaluation scripts read the merged `manifest.csv` from this directory.

Processed medical images, subject-level manifests, and derived files are ignored by Git by default.
