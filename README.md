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
| ATYPICAL buildings | 58 (9.9% of 583 CBECS-mapped) |
| ENERGY STAR blind spot | 64.7% of certifiable buildings |

## Quick Start

### From pre-computed results (recommended for reviewers)

```bash
pip install -r requirements.txt

# Compute all paper values from CSV (source of truth)
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
  04_verify_all_claims.py        Compute all paper values from CSV (no hardcoding)
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
  -> Level 1: Quadrant classification (threshold from CSV quadrant_c14 column)
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
| Pattern Score threshold | as defined in CSV `quadrant_c14` | Score = 50 means at median (baseline) |
| Excess CVRMSE threshold | 5 pp | IQR fence (11.6 pp), Cohen's d = 1.22, min n = 10 |
| NMBE direction threshold | +/- 2% | Conservative ASHRAE Guideline 14 criterion |
| Low-load exclusion | mean < 5 kWh/hr | Denominator inflation prevention |
| EUI Score formula | norm.cdf(-z) * 100 | z = (EUI - CBECS_median) / CBECS_std |

## Verification

Running `04_verify_all_claims.py` computes all paper values directly from the CSV:

- Quadrant distribution (Table 6)
- Level 2 classification and cross-tab (Table 8)
- Correlation coefficients and regression
- NMBE statistics by group (Table 9)
- Mann-Whitney U test
- NMBE direction (Table 10)
- ENERGY STAR reversal analysis
- Reversal case counts (Table 11)
- CVRMSE by building type (Table 4)
- Causal decomposition (Table 5)
- Best-practice candidates (Table 12)

**No hardcoded expected values.** The script output IS the source of truth for all paper numbers.

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
