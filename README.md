# CELM-Cardio — code and results

Code, data splits, trained meta-learners and per-image results for:

> **CELM-Cardio: A Heterogeneous CNN Ensemble for Cardiomegaly Classification on Chest Radiographs with Leakage-Controlled Evaluation and External Validation**
> International Journal of Intelligent Engineering and Systems (revised manuscript, Paper ID 20266157).

All numbers in the manuscript come from the single run whose outputs are stored here (seed 42).

## Datasets (not included)

The chest radiographs are **not redistributed**. Both datasets are public and must be obtained from their providers under their own terms of use:

| Dataset | Role | Source |
|---|---|---|
| NIH ChestX-ray14 (Wang et al., CVPR 2017) | development and internal test | https://nihcc.app.box.com/v/ChestXray-NIHCC |
| CheXpert (Irvin et al., AAAI 2019) | external test only (frozen models) | https://stanfordmlgroup.github.io/competitions/chexpert/ |

The notebook downloads the Kaggle copies actually used: `khanfashee/nih-chest-x-ray-14-224x224-resized` (NIH, 224 × 224) and `ashery/chexpert` (CheXpert-v1.0-small). `results/split_manifest.csv` lists every NIH image ID with its patient ID and partition, so the exact split can be rebuilt from the original files.

## Contents

```
notebook/CELM_Cardio.ipynb      full pipeline (Google Colab): split, training, FPPR, XGBoost,
                                TTA, statistics, calibration, CheXpert evaluation, automated CTR
results/
  experiment_config.json        seed, split ratios and counts, hyperparameters, checkpoint rule,
                                FPPR weights, TTA settings, model/threshold selection, final results
  split_manifest.csv            image → patient → partition (train / val / meta / test)
  val_acc_fppr.json             validation accuracy of each checkpoint and the FPPR weights (Eq. 1)
  predictions_{val,meta,test}.csv        per-image probabilities of all 14 configurations (NIH)
  predictions_chexpert_{A,B}.csv         per-image probabilities on CheXpert cohorts A and B
  final_metrics*.csv            test metrics with patient-level bootstrap 95% CIs
  statistical_tests.csv         DeLong and McNemar tests with Holm correction (NIH test)
  external_*                    CheXpert cohort description, metrics and tests
  ctr_measurements.csv, ctr_analysis.json   automated CTR per test image and its analysis
  literature_comparison.csv     values used in Table 6
  environment.txt               Python and library versions (pip freeze)
logs/                           per-epoch training logs (Phase 1 = *_fase1.csv, Phase 2 = *.csv)
models/xgb_*.pkl                trained XGBoost meta-learners (with/without FPPR feature, with/without TTA)
figures/                        figures of the revised manuscript
```

The three backbone checkpoints (`eff_v7.keras`, `den_v7.keras`, `res_v7.keras`, 0.1–0.3 GB each) are archived with the Zenodo record of this repository: [DOI].

## Protocol in brief

- **Split:** patient-level, stratified, 60 / 10 / 15 / 15 % into train / val / meta / test; no patient appears in two partitions.
- **Backbones:** EfficientNetB4, DenseNet121, ResNet50V2 (ImageNet), two-phase training (frozen head, then full fine-tuning with cosine decay), focal loss, augmentation without horizontal flip. The retained checkpoint is the best validation accuracy over both phases.
- **Level 1:** unweighted mean and FPPR weighting, with weights from validation accuracy.
- **Level 2:** XGBoost trained only on backbone predictions for the meta partition; hyperparameters chosen by patient-grouped 5-fold CV.
- **Model and threshold selection:** highest meta-partition AUC from nested cross-validation (ties broken by Brier score); Youden threshold on the meta partition. The test partition is used once.
- **External test:** all models and thresholds are frozen and applied to CheXpert.
- **CTR:** heart and lung segmentation with the TorchXRayVision PSPNet (ChestX-Det), CTR = MHD / MTD, with automatic quality control.

## Reproducing

1. Open `notebook/CELM_Cardio.ipynb` in Google Colab (GPU runtime).
2. Upload a Kaggle API token (`kaggle.json`) when prompted. The notebook downloads both datasets.
3. Run all cells. Outputs are written to Google Drive in the same structure as `results/`.

Main versions: TensorFlow 2.21.0, Keras 3.13.2, XGBoost 3.4.1, scikit-learn 1.6.1, SHAP 0.52.0, TorchXRayVision 1.5.5 (complete list in `results/environment.txt`). GPU non-determinism can cause small differences when retraining. The stored predictions reproduce every table exactly.

## Citation

Please cite the article above. Contact: Eko Mulyanto Yuniarno (corresponding author), Institut Teknologi Sepuluh Nopember.
