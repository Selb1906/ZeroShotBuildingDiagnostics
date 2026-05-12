# Appendix D & E Analysis Results

## Appendix D: Threshold Sensitivity Analysis

### D.1 Excess CVRMSE Threshold Sweep (Table D1)

Population: 583 CBECS-mapped buildings, pattern_score < 50 (quadrants B/D)
NMBE direction threshold fixed at +-2%

| threshold_pp | n_ATYPICAL | n_CV_DRIVEN | mean_abs_nmbe_atypical_pct | mean_abs_nmbe_cv_pct | cohens_d | pct_over | pct_under | pct_neutral |
| ------------ | ---------- | ----------- | -------------------------- | -------------------- | -------- | -------- | --------- | ----------- |
| 3.0          | 69.0       | 142.0       | 4.69                       | 1.3984               | 0.7368   | 40.58    | 14.49     | 44.93       |
| 4.0          | 64.0       | 147.0       | 4.9354                     | 1.4036               | 0.7957   | 42.19    | 15.62     | 42.19       |
| 5.0          | 58.0       | 153.0       | 5.2908                     | 1.4073               | 0.8833   | 43.1     | 17.24     | 39.66       |
| 6.0          | 52.0       | 159.0       | 5.5436                     | 1.4712               | 0.9281   | 44.23    | 17.31     | 38.46       |
| 7.0          | 47.0       | 164.0       | 5.9997                     | 1.4646               | 1.0468   | 44.68    | 19.15     | 36.17       |
| 8.0          | 43.0       | 168.0       | 6.4569                     | 1.4556               | 1.1702   | 48.84    | 20.93     | 30.23       |
| 10.0         | 36.0       | 175.0       | 7.0108                     | 1.5417               | 1.2858   | 47.22    | 22.22     | 30.56       |

### D.2 NMBE Direction Threshold Sweep (Table D2)

Population: ATYPICAL buildings (5pp excess threshold, pattern_score < 50), n=58

| nmbe_threshold_pct | n_OVER | n_UNDER | n_NEUTRAL | mean_nmbe_pct |
| ------------------ | ------ | ------- | --------- | ------------- |
| 1.0                | 28.0   | 16.0    | 14.0      | 3.6553        |
| 2.0                | 25.0   | 10.0    | 23.0      | 3.6553        |
| 3.0                | 22.0   | 5.0     | 31.0      | 3.6553        |
| 5.0                | 16.0   | 2.0     | 40.0      | 3.6553        |

### D.3 Cross-Sensitivity Heatmap (Table D3: n_OVER)

Cell values = number of buildings classified as OVER-consuming (NMBE > threshold)
Population: CBECS-mapped, pattern_score < 50

| excess_threshold_pp | n_over_nmbe_1pct | n_over_nmbe_2pct | n_over_nmbe_3pct | n_over_nmbe_5pct |
| ------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 3                   | 34               | 28               | 24               | 16               |
| 4                   | 31               | 27               | 23               | 16               |
| 5                   | 28               | 25               | 22               | 16               |
| 6                   | 26               | 23               | 21               | 15               |
| 7                   | 24               | 21               | 20               | 15               |
| 8                   | 22               | 21               | 20               | 15               |
| 10                  | 17               | 17               | 16               | 13               |

## Appendix E: Leave-One-Site-Out Cross-Validation

### Table E1: Regression Parameters per Fold

| fold             | n_train | n_test | alpha  | beta    | R2_train | R2_test |
| ---------------- | ------- | ------ | ------ | ------- | -------- | ------- |
| Hold-out Bear    | 520     | 91     | 0.5409 | -0.0342 | 0.7067   | 0.6338  |
| Hold-out Fox     | 476     | 135    | 0.4745 | -0.0054 | 0.6984   | 0.6787  |
| Hold-out Rat     | 331     | 280    | 0.6132 | -0.0454 | 0.6977   | 0.6417  |
| Hold-out Panther | 506     | 105    | 0.5592 | -0.0391 | 0.7125   | 0.4849  |

### Table E2: Classification Agreement per Fold

| fold             | n_test_cbecs | ATYPICAL_fold | ATYPICAL_baseline | agreement_pct | cohen_kappa |
| ---------------- | ------------ | ------------- | ----------------- | ------------- | ----------- |
| Hold-out Bear    | 85           | 14            | 14                | 100.0         | 1.0         |
| Hold-out Fox     | 131          | 7             | 8                 | 95.4          | 0.895       |
| Hold-out Rat     | 275          | 21            | 23                | 85.5          | 0.743       |
| Hold-out Panther | 92           | 11            | 13                | 95.7          | 0.893       |

### E.3 Overall Summary (Pooled across all folds)

- Total test buildings (CBECS-mapped, all folds): 583
- Overall agreement: 91.4%
- Overall Cohen's kappa: 0.837

### E.4 Pooled Confusion Matrix

| Baseline  | NORMAL | CV_DRIVEN | ATYPICAL |
| --------- | ------ | --------- | -------- |
| NORMAL    | 338    | 32        | 2        |
| CV_DRIVEN | 7      | 145       | 1        |
| ATYPICAL  | 4      | 4         | 50       |
