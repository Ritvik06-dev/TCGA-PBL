# Does Multimodal Fusion Beat Clinical Variables? Breast Cancer Progression Risk on TCGA-BRCA

A leakage-controlled benchmark of multimodal survival prediction on **898 TCGA-BRCA patients** with **five paired data types**: clinical variables, RNA expression, somatic mutations, pathology report text and whole-slide image features.

**Main finding:** routine clinical variables alone reach a C-index of **0.680**. The best fusion method (stacking) reaches **0.696**. That gain (+0.017) appears in 19 of 25 splits but is **not statistically significant** (p = 0.30). Naive fusion performs *worse* than clinical data alone.

> Project Based Learning-3 (DSE3170), Department of Data Science and Engineering, Manipal University Jaipur.

---

## Results

C-index over 25 test splits (5-fold cross-validation × 5 repeats). Gain = model minus clinical on the same split; 95% CI and p-value from the Nadeau–Bengio corrected resampled t-test.

| Model | C-index (mean ± SD) | Gain vs clinical | 95% CI of gain | p | Better than clinical in |
|---|---|---|---|---|---|
| Clinical (7 variables) | 0.680 ± 0.039 | — | — | — | — |
| RNA expression | 0.611 ± 0.060 | −0.069 | −0.146 to 0.007 | 0.075 | 3 / 25 |
| Pathology reports | 0.606 ± 0.060 | −0.074 | −0.158 to 0.011 | 0.084 | 5 / 25 |
| Slides (attention MIL) | 0.584 ± 0.056 | −0.095 | −0.173 to −0.018 | 0.018 | 3 / 25 |
| Mutations | 0.464 ± 0.061 | −0.216 | −0.304 to −0.129 | < 0.001 | 0 / 25 |
| Fusion: concatenation | 0.674 ± 0.044 | −0.006 | −0.059 to 0.046 | 0.81 | 9 / 25 |
| Fusion: equal-weight average | 0.661 ± 0.047 | −0.019 | −0.078 to 0.041 | 0.52 | 9 / 25 |
| **Fusion: stacking** | **0.696 ± 0.040** | **+0.017** | **−0.016 to 0.049** | **0.30** | **19 / 25** |

**Stacking weights:** clinical 0.50 · reports 0.25 · RNA 0.20 · slides −0.05 · mutations −0.12

**Kaplan–Meier (median split, 449 vs 449):** log-rank p = 7.8 × 10⁻⁵ (clinical), 1.2 × 10⁻⁷ (stacking)

Figures: `fig_models.png`, `fig_gain.png`, `fig_weights.png`, `fig_km.png`

---

## Method

| Data type | Features | Model |
|---|---|---|
| Clinical | Age, AJCC stage, T, N, lobular, other histology, post-menopausal (7) | Cox proportional hazards |
| RNA | log2 RSEM → top 2,000 variable genes → PCA (10) | Cox |
| Mutations | Top 10 mutated genes (0/1) + log tumour mutation burden | Cox |
| Reports | TF-IDF (1–2-grams, 5,000 terms) → truncated SVD (10) | Cox |
| Slides | UNI2-h patch features (1,536-d), 1,024 patches per patient | Attention MIL (ABMIL) with Cox loss |

**Fusion:** concatenation (early), equal-weight average of standardised risks (late), and stacking with a Cox meta-learner trained on out-of-fold risks.

**Evaluation:** `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)`, identical for every model. All preprocessing is fitted on training patients only.

**Target:** progression-free interval (PFI) from TCGA-CDR, with 122 events among 898 patients.

---

## Data (not included in this repo)

Download the data yourself and place it in the project folder:

| Data | Source | File / folder |
|---|---|---|
| Outcomes + clinical | [TCGA-CDR (GDC PanCanAtlas)](https://gdc.cancer.gov/about-data/publications/pancanatlas) | `cdr.xlsx` |
| RNA, mutations, extra clinical | [cBioPortal: brca_tcga_pan_can_atlas_2018](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018) | `brca_tcga_pan_can_atlas_2018/` |
| Pathology reports | TCGA-Reports (Kefeli & Tatonetti, 2024) | `TCGA_Reports.csv` |
| Slide features | [MahmoodLab/UNI2-h-features](https://huggingface.co/datasets/MahmoodLab/UNI2-h-features) (gated; request access) | `features/` |

The UNI2-h features are licensed CC-BY-NC-ND and must not be redistributed, which is why `features/`, `bags.npy` and other derived arrays are git-ignored.

---

## How to run

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib h5py openpyxl
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build
```

Run in this order:

| Step | Script | What it does |
|---|---|---|
| 1 | `final_cohort.py` | Builds the 908-patient paired cohort → `cohort_final.csv` |
| 2 | `clinical_plus.py` | Clinical baseline with 7 variables |
| 3 | `build_bags.py` | Samples 1,024 slide patches per patient → `bags.npy` |
| 4 | `attn_model.py` | Trains attention MIL on GPU (~20 min on an RTX 3050) → `slide_attn_risks.npz` |
| 5 | `fusion2.py` | All single-modality models + 3 fusion methods on 25 splits → `scores_fusion2.csv` |
| 6 | `figures.py` | Result figures |
| 7 | `get_numbers.py` | Prints the results table, gains, CIs, p-values, weights and log-rank tests |

Earlier exploration scripts: `rna_model.py`, `mutation_model.py`, `text_model.py`, `pool_slides.py`, `slide_model.py`, `fusion.py`.

---

## Limitations and future work

- Only 122 events, which limits statistical power
- Single dataset; external validation on METABRIC is planned
- Simple text, RNA and mutation encoders (BioBERT, pathway features to try)
- Discrimination only; calibration not yet assessed

---

## Team

Shivangi Kotnala · Sumit Gareri · Aadhya Govil · Ritvik · Anushka Dhanuka

Guide: Dr. Ram Kumar Yadav, Department of Data Science and Engineering, Manipal University Jaipur

---

## Key references

- Liu et al. (2018). An integrated TCGA pan-cancer clinical data resource. *Cell*.
- Ilse, Tomczak & Welling (2018). Attention-based deep multiple instance learning. *ICML*.
- Chen et al. (2024). Towards a general-purpose foundation model for computational pathology (UNI). *Nature Medicine*.
- Kefeli & Tatonetti (2024). TCGA-Reports: a machine-readable pathology report resource. *Patterns*.
- Zhou, Zhou & Chen (2024). Cohort-individual cooperative learning for multimodal cancer survival analysis. [arXiv:2404.02394](https://arxiv.org/abs/2404.02394)
- Nadeau & Bengio (2003). Inference for the generalization error. *Machine Learning*.