# Box-Cox λ Sensitivity Analysis

**Purpose**: Address Reviewer 1, Comment 10 — "The framework's effectiveness is highly dependent on the Box-Cox transformation."

## Setup

- **Baseline λ**: −0.06722 (single global Box-Cox parameter from `metadata/transforms/boxcox.pkl`)
- **Perturbations**: λ × {0.8, 0.9, 1.0, 1.1, 1.2} → λ ∈ {−0.0538, −0.0605, −0.0672, −0.0739, −0.0807}
- **Population**: All 611 BDG-2 buildings

## Method

For each λ' variant:
1. Apply Box-Cox transform with λ' to (actual, predicted) hourly time series
2. Recompute CVRMSE in transformed space
3. Recompute Pattern Score (within-type z-score → CDF) using new CVRMSE
4. Compare building rankings vs baseline λ

## Results — Rank Stability

| λ perturbation | λ value | Spearman ρ (CVRMSE) | Spearman ρ (Pattern Score) |
|:---:|:---:|:---:|:---:|
| −20% | −0.0538 | **0.9999** | 0.9974 |
| −10% | −0.0605 | **0.9999** | 0.9990 |
| baseline | −0.0672 | 1.0000 | 1.0000 |
| +10% | −0.0739 | **0.9999** | 0.9979 |
| +20% | −0.0807 | **0.9999** | 0.9855 |

## Key Finding

**Spearman ρ > 0.985 for all ±20% λ perturbations**, confirming that the framework's building rankings are essentially unchanged by Box-Cox parameter selection. The framework is robust to λ choice.

## Interpretation for Paper

> A sensitivity analysis confirmed that the Box-Cox λ parameter selection does not materially affect building rankings. Perturbing λ by ±20% (from −0.0672 to a range of −0.054 to −0.081) preserved CVRMSE-based rankings at Spearman ρ > 0.9999 and Pattern Score rankings at ρ > 0.985 across all 611 buildings. This robustness arises because Box-Cox is a monotonic transformation: changing λ stretches the scale non-linearly but preserves ordinal relationships.

## Files

- `boxcox_sensitivity.csv` — Numerical data
- `figure_boxcox_sensitivity.png` — Bar chart (300 DPI)
