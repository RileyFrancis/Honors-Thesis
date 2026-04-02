# Data Files Used

## LCGA Input Data

Located in `/shared/healthinfolab/datasets/ABCD/Irritability/Clinical_Data/Irritability/Release_5.0/`

| File | Description |
|------|-------------|
| `abcd_cbcl_irr_index_release5.0_0m_long.csv` | CBCL irritability index — baseline (0 months) |
| `abcd_cbcl_irr_index_release5.0_12m_long.csv` | CBCL irritability index — 12-month follow-up |
| `abcd_cbcl_irr_index_release5.0_24m_long.csv` | CBCL irritability index — 24-month follow-up |
| `abcd_cbcl_irr_index_release5.0_36m_long.csv` | CBCL irritability index — 36-month follow-up |
| `abcd_cbcl_irr_index_release5.0_48m_long.csv` | CBCL irritability index — 48-month follow-up |

**Key variables used:** `src_subject_id`, `eventname`, `cbcl_irr_index_cnst`

---

## ABCD Package Files

Located in `/shared/healthinfolab/datasets/ABCD/Package_1215452/`

### Demographics

| File | Description | Key Variables |
|------|-------------|---------------|
| `pdem02.txt` | Parent demographics | `interview_age`, `demo_sex_v2`, `demo_race_a_p___*`, `demo_ethn_v2`, `demo_comb_income_v2`, `demo_prnt_ed_v2`, `demo_prnt_marital_v2` |

### CBCL (Child Behavior Checklist)

| File | Description | Key Variables |
|------|-------------|---------------|
| `abcd_cbcls01.txt` | CBCL syndrome scales and DSM-oriented T-scores | `cbcl_scr_syn_*_t`, `cbcl_scr_dsm5_*_t` |
| `abcd_cbcl01.txt` | CBCL raw item responses | Raw item columns (T-scores are **not** in this file) |

### K-SADS (Kiddie Schedule for Affective Disorders and Schizophrenia)

| File | Description | Key Variables |
|------|-------------|---------------|
| `abcd_ksad01.txt` | K-SADS parent diagnostic interview | `ksads_1_3_p`, `ksads_3_229_p`, `ksads_15_432_p`, `ksads_15_91_p` (irritability); `ksads_14_*` (ADHD); `ksads_2_*` (ODD) |

### Emotion Dysregulation

| File | Description | Key Variables |
|------|-------------|---------------|
| `diff_emotion_reg_p01.txt` | DERS — Difficulties in Emotion Regulation Scale, parent-report | `ders_upset_*` items (summed to `ders_total`); `ders_upset_irritation_p` |

> **Note:** This file contains **no baseline rows**. Data is loaded across all timepoints and each subject's earliest available observation is used.

### ODD (Oppositional Defiant Disorder)

| File | Description | Key Variables |
|------|-------------|---------------|
| `opp_defiant_disorder_p01.txt` | K-SADS ODD symptom items | `ksads_odd_raw_*` (binary 0/1 items summed to symptom count) |

---
According to the paper [Characterizing the Neural Correlates of Response Inhibition and Error Processing in Children With Symptoms of Irritability and/or Attention-Deficit/Hyperactivity Disorder in the ABCD Study®](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.803891/full), The most important  odules to look at are:

- From the Major Depressive Disorder module: "Irritability Present"
- From the DMDD module: "Temper outbursts occur 3 or more times per week"
- From the Oppositional Defiant Disorder module: "Often touchy or easily annoyed Present", "Often loses temper Present"