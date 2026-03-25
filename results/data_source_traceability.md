# Data Source Traceability for Paper Figures and Tables

**Purpose:** MDPI Buildings 투고용 — 논문의 모든 수치의 데이터 출처와 계산 근거 명시

---

## Primary Data Source

| 파일 | 설명 | 행 수 |
|------|------|-------|
| `results/cbecs2018_c14_median_evaluation.csv` | 611 BDG-2 건물의 평가 결과 (box-cox CVRMSE, EUI Score, Pattern Score, Level 2/3 분류) | 611 |
| `results/predictions_TransformerWithGaussian-L_bdg2_raw.csv` | 건물별 시간별 예측값/실측값 (9,247,992 timesteps) | 9,247,992 |
| `metadata/BDG2_metadata_with_sqft_eui.csv` | 건물 메타데이터 (면적, 유형, subtype) | 611 |

---

## Subset Definitions

| 서브셋 | 필터 조건 | 건물 수 | 용도 |
|--------|----------|---------|------|
| ALL | 없음 | 611 | CVRMSE by type (Figure 7/Table) |
| CBECS-mapped | `building_type NOT IN ('Other','Technology','Parking','Utility')` | 583 | Quadrant, Level 2/3, Findings |
| ATYPICAL | CBECS-mapped + `l2_cause == 'ATYPICAL'` | 58 | Level 3 NMBE 분석 |
| Best Practice | ATYPICAL + `nmbe < -0.02` + `mean_load_kwh >= 5` | 9 | Table 12 |

---

## Threshold Convention

| 분류 | 기준 | 비고 |
|------|------|------|
| Quadrant A (Excellent) | EUI Score ≥ 50 AND Pattern Score ≥ 50 | `>=50` inclusive |
| Quadrant B (Eff. but Irreg.) | EUI Score ≥ 50 AND Pattern Score < 50 | |
| Quadrant C (Cons. but Ineff.) | EUI Score < 50 AND Pattern Score ≥ 50 | |
| Quadrant D (Needs Impr.) | EUI Score < 50 AND Pattern Score < 50 | |
| NORMAL | Pattern Score ≥ 50 (= Quadrant A + C) | |
| CV_DRIVEN | Pattern Score < 50 AND Excess CVRMSE ≤ 5pp | |
| ATYPICAL | Pattern Score < 50 AND Excess CVRMSE > 5pp | |

---

## Tables — Data Sources and Computation

### Table 1. Comparison with Existing Frameworks
- **Source:** Literature review (no CSV data)

### Table 2. CBECS 2018 Table C14 Reference Values
- **Source:** CBECS 2018 Table C14 (external), mapped via `metadata/BDG2_metadata_with_sqft_eui.csv`
- **Columns used:** `c14_type`, `cbecs_median_kbtu_sqft`, `cbecs_std_kbtu_sqft`

### Table 3–5. Methodology descriptions
- **Source:** Framework definition (no CSV data)

### Table 6. Level 1 Quadrant Results (n=583)
- **Source:** `cbecs2018_c14_median_evaluation.csv`, CBECS-mapped subset
- **Columns:** `quadrant_c14`, `eui_score_c14`, `pattern_score`
- **Computation:** `value_counts('quadrant_c14')`, `median('eui_score_c14')`, `median('pattern_score')` per quadrant
- **Values:** A=122 (20.9%), B=107 (18.4%), C=250 (42.9%), D=104 (17.8%)

### Table 7. Quadrant by Building Type
- **Source:** `cbecs2018_c14_median_evaluation.csv`, CBECS-mapped subset
- **Columns:** `building_type`, `quadrant_c14`
- **Computation:** `crosstab(building_type, quadrant_c14)` → row percentages
- **Filter:** building types with n ≥ 15

### Table 8. Level 2 Cross-tab (Quadrant × L2)
- **Source:** `cbecs2018_c14_median_evaluation.csv`, CBECS-mapped subset
- **Columns:** `quadrant_c14`, `l2_cause`
- **Computation:** `crosstab(quadrant_c14, l2_cause)`
- **Values:** A=[122,0,0], B=[0,81,26], C=[250,0,0], D=[0,72,32]
- **Note:** B/D have zero NORMAL buildings (verified: all pattern_score=50 buildings fall in A/C with >=50 threshold)

### Table 9. NMBE by Level 2
- **Source:** `cbecs2018_c14_median_evaluation.csv`, CBECS-mapped subset
- **Columns:** `l2_cause`, `nmbe`
- **Computation:** `groupby('l2_cause').agg(mean=abs(nmbe).mean, pct_over5=(abs(nmbe)>0.05).mean, median=abs(nmbe).median)`
- **Values:** NORMAL: 0.76%/0.3%/0.55%, CV_DRIVEN: 1.41%/1.3%/1.11%, ATYPICAL: 5.29%/31.0%/2.68%

### Table 10. Level 3 NMBE Direction
- **Source:** `cbecs2018_c14_median_evaluation.csv`, ATYPICAL subset (n=58)
- **Columns:** `nmbe` with thresholds ±2%

### Table 11. Four Reversal Cases
- **Source:** `cbecs2018_c14_median_evaluation.csv`, CBECS-mapped subset
- **Computation:** Quadrant-based classification with CV/CVRMSE comparisons

### Table 12. Best Practice Buildings
- **Source:** `cbecs2018_c14_median_evaluation.csv` + `predictions_TransformerWithGaussian-L_bdg2_raw.csv`
- **Filter:** ATYPICAL + NMBE < -2% + mean_load ≥ 5 kWh → 9 buildings
- **Hourly patterns verified from:** prediction residuals grouped by hour

---

## Figures — Type and Data Sources

### Figures to keep as images (conceptual diagrams):
| Figure | Type | Description | Source |
|--------|------|-------------|--------|
| Fig 1 | Conceptual SVG | Research positioning matrix | Manual design |
| Fig 2 | Flowchart SVG | Framework overview | Manual design |
| Fig 3 | Flowchart SVG | Hierarchical algorithm with data flow | Manual design, numbers from CSV |
| Fig 4 | Illustration SVG | CVRMSE decomposition example | Computed examples (not from CSV) |

### Figures converted to tables for MDPI:
| Former Figure | Now Table | Reason |
|---------------|-----------|--------|
| Fig 6 (Quadrant bar) | **Table 6** | Data already in Table 6 |
| Fig 7 (CVRMSE by type) | **New Table** in Section 5.3 | Horizontal bar → table with mean/median/n |
| Fig 8 (NMBE comparison) | **Table 9** | Data already in Table 9 |
| Fig 9 (Level2×Quadrant) | **Table 8** | Data already in Table 8 |

### Figures to keep as images (data-driven):
| Figure | Type | Description | Data Source | Script |
|--------|------|-------------|------------|--------|
| Fig 5 | Scatter plot | EUI Score vs Pattern Score | `cbecs2018_c14_median_evaluation.csv` (583 rows, `eui_score_c14`, `pattern_score`) | `scripts/generate_paper_figures.py` |

---

## Key Statistics — Computation Details

| Statistic | Value | Source | Computation |
|-----------|-------|--------|-------------|
| Raw EUI↔CVRMSE (n=583) | r = −0.082, p = 0.047 | `cbecs2018_c14_median_evaluation.csv` CBECS subset | `pearsonr(eui_kbtu_sqft, cvrmse)` |
| Raw EUI↔CVRMSE (n=611) | r = −0.029, p = 0.47 | `cbecs2018_c14_median_evaluation.csv` all | `pearsonr(eui_kbtu_sqft, cvrmse)` |
| EUI Score↔Pattern Score (n=583) | r = −0.291, p < 0.001 | CBECS subset | `pearsonr(eui_score_c14, pattern_score)` |
| CVRMSE~CV regression | CVRMSE = 0.541×CV − 0.030, R²=0.700 | All 611 | `OLS(cvrmse ~ cv)` |
| ENERGY STAR reversal | 55/85 = 64.7% | CBECS subset | `eui_score_c14 >= 75` AND `pattern_score < 50` |
| UNDER-consuming best practice | 9 buildings | CBECS ATYPICAL subset | `nmbe < -0.02` AND `mean_load_kwh >= 5` |

---

## Reproducibility

All figures and tables can be regenerated using:
```bash
python scripts/generate_paper_figures.py
```

Input: `results/cbecs2018_c14_median_evaluation.csv`
Output: `figures/fig*.png`, `figures/fig*.pdf`
