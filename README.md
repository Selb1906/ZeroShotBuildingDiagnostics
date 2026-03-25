# Hierarchical Building Energy Performance Framework

Analysis code and results for the paper:

**"Beyond Annual EUI: A Hierarchical Building Energy Performance Framework Using Zero-Shot Temporal Pattern Analysis from a Pre-Trained Foundation Model"**

Submitted to *Buildings* (MDPI)

## Overview

This repository contains the analysis pipeline that reproduces all results in the paper. The framework evaluates 611 buildings from the [Building Data Genome Project 2 (BDG-2)](https://github.com/buds-lab/building-data-genome-project-2) using zero-shot predictions from the [BuildingsBench](https://github.com/NREL/BuildingsBench) TransformerWithGaussian-L model.

### Key Results

| Metric | Value |
|--------|-------|
| Buildings analyzed | 611 (583 CBECS-mapped) |
| Observation-prediction pairs | 9,247,992 |
| EUI-CVRMSE correlation | r = -0.082 (R^2 = 0.007) |
| ATYPICAL buildings | 58 (9.9% of CBECS-mapped) |
| ENERGY STAR blind spot | 64.7% of certifiable buildings |

## Quick Start

### From pre-computed results (recommended for reviewers)

```bash
pip install -r requirements.txt

# Verify all 50 paper claims against CSV
python scripts/04_verify_all_claims.py

# Generate paper figures
python scripts/03_generate_figures.py

# View threshold justification
python scripts/06_threshold_justification.py
```

### Full reproduction (from raw predictions)

```bash
# Step 1: Run zero-shot inference (requires BuildingsBench + GPU)
# See scripts/01_run_inference.sh

# Step 2: Reproduce all results from predictions
python scripts/02_reproduce_paper.py

# Step 3: Verify
python scripts/04_verify_all_claims.py
```

## Repository Structure

```
scripts/
  01_run_inference.sh            Zero-shot inference (BuildingsBench)
  02_reproduce_paper.py          Full pipeline: predictions -> evaluation CSV
  03_generate_figures.py         Generate paper figures (PNG/PDF, 300 DPI)
  04_verify_all_claims.py        Automated verification of 50 paper claims
  05_analyze_hourly_patterns.py  Table 12: best-practice building patterns
  06_threshold_justification.py  5pp threshold statistical evidence

results/
  cbecs2018_c14_median_evaluation.csv   Per-building evaluation (611 rows)
  data_source_traceability.md           Data source audit trail

metadata/
  BDG2_metadata_with_sqft_eui.csv       Building metadata (type, sqft, EUI)
```

## Pipeline

```
BDG-2 hourly data + TransformerWithGaussian-L checkpoint
    |
    v
[01] Zero-shot inference (BuildingsBench)
    |
    v
predictions CSV (9.2M rows, ~1.4 GB)
    |
    v
[02] Per-building metrics (CVRMSE, NMBE, CV)
  -> EUI Score (CBECS 2018 Table C14 z-score -> normal CDF)
  -> Pattern Score (within-type CVRMSE z-score -> normal CDF)
  -> Level 1: Quadrant classification (EUI Score >= 50, Pattern Score >= 50)
  -> Level 2: CVRMSE decomposition (Excess CVRMSE > 5pp -> ATYPICAL)
  -> Level 3: NMBE direction (>+2% OVER, <-2% UNDER)
    |
    v
evaluation CSV (611 rows) -> [03] figures, [04] verification
```

## External Data Sources

| Source | URL | Used For |
|--------|-----|----------|
| BDG-2 | [github.com/buds-lab/building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) | Building hourly load data |
| BuildingsBench | [github.com/NREL/BuildingsBench](https://github.com/NREL/BuildingsBench) | Pre-trained model + benchmark framework |
| CBECS 2018 | [eia.gov/consumption/commercial](https://www.eia.gov/consumption/commercial/data/2018/) | Table C14 EUI reference values |

## Methodology Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Pattern Score threshold | >= 50 (inclusive) | Standard percentile convention |
| Excess CVRMSE threshold | 5 pp | IQR fence (11.6 pp), Cohen's d = 1.22, min n = 10 |
| NMBE direction threshold | +/- 2% | Conservative ASHRAE Guideline 14 criterion |
| Low-load exclusion | mean < 5 kWh/hr | Denominator inflation prevention |
| EUI Score formula | norm.cdf(-z) * 100 | z = (EUI - CBECS_median) / CBECS_std |

## Verification

Running `04_verify_all_claims.py` checks 50 quantitative claims from the paper:

- Quadrant distribution (4 counts)
- Level 2 classification (3 counts)
- Level 2 x Quadrant cross-tab (4 cells)
- Correlation coefficients (2 values)
- CV-CVRMSE regression (slope, intercept, R^2)
- NMBE statistics by group (6 values)
- Mann-Whitney U test (U statistic, p-value)
- NMBE direction counts (3 values)
- ENERGY STAR reversal (3 values)
- Reversal case counts (4 values)
- CVRMSE by building type (6 means)
- Causal decomposition (4 counts)
- Best-practice candidates (2 counts)

**All 50 checks: PASSED**

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{building_energy_framework_2026,
  title={Beyond Annual {EUI}: A Hierarchical Building Energy Performance Framework
         Using Zero-Shot Temporal Pattern Analysis from a Pre-Trained Foundation Model},
  journal={Buildings},
  year={2026}
}
```
