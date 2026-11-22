# Public release checklist

- [ ] Confirm that all included Python source code is your own or is appropriately attributed and compatible with the license you choose.
- [ ] Review the selected PolyForm Noncommercial 1.0.0 license and replace `[Your Full Legal Name]` in `NOTICE` before publishing.
- [ ] Confirm that no raw/processed dataset images, masks, patient/subject-level manifests, predictions, probability maps, or model checkpoints have been staged.
- [ ] Confirm that the four dataset source links and task roles in `DATASETS.md` are accurate.
- [ ] Check that no local absolute paths, credentials, private keys, or personal information are present.
- [ ] Review the optional reference notebook and its upstream UniverSeg dependency.
- [ ] Run `python -m compileall preprocessing training dataset_inspection tests`.
- [ ] Run `python -m pytest -q`.
- [ ] Inspect `git status --short` and every staged file before the first push.

- [ ] Verify `README.md` credits the original UniverSeg authors, paper, and official JJGO repository.
- [ ] Verify `DATASETS.md` lists all four dataset source links and the confirmed original paper citations; the dental citation remains source-only pending confirmation.
