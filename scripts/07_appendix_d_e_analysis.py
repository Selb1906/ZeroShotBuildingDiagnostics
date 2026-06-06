#!/usr/bin/env python3
"""
Appendix D & E Analysis for MDPI Buildings journal Major Revision.

Appendix D: Threshold Sensitivity Analysis
  - Part A: Excess CVRMSE threshold sweep [3,4,5,6,7,8,10] pp
  - Part B: NMBE direction threshold sweep for ATYPICAL buildings
  - Part C: Cross-sensitivity heatmap data

Appendix E: Leave-One-Site-Out Cross-Validation
  - 4 sites: Bear, Fox, Rat, Panther
  - Per-fold regression fit, pattern score recomputation, classification comparison

Data: results/cbecs2018_c14_median_evaluation.csv (611 buildings)
CBECS-mapped subset: exclude building_type in ['Other','Technology','Parking','Utility'] -> 583 buildings

Usage: python scripts/appendix_d_e_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(BASE_DIR, 'results', 'cbecs2018_c14_median_evaluation.csv')
OUT_DIR = os.path.join(BASE_DIR, 'working')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXCLUDED_TYPES = ['Other', 'Technology', 'Parking', 'Utility']
SITES = ['Bear', 'Fox', 'Rat', 'Panther']
EXCESS_THRESHOLDS = [3, 4, 5, 6, 7, 8, 10]   # pp
NMBE_THRESHOLDS   = [1, 2, 3, 5]              # pp (±)
BASELINE_THRESHOLD_PP = 5
BASELINE_NMBE_PP      = 2


# ===========================================================================
# Helper functions
# ===========================================================================

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled Cohen's d (a vs b)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var) if pooled_var > 0 else np.nan


def fit_regression(df: pd.DataFrame):
    """Fit OLS CV -> CVRMSE and return (slope, intercept, r2)."""
    slope, intercept, r, _, _ = stats.linregress(df['cv'], df['cvrmse'])
    return slope, intercept, r**2


def compute_pattern_scores(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.Series:
    """
    Recompute pattern_score for test buildings using training set within-type statistics.
    Formula: pattern_score = round((1 - norm.cdf((cvrmse - mu_train) / sd_train)) * 100)
    where mu_train, sd_train are computed per building_type_c14 from train_df.
    """
    scores = pd.Series(index=test_df.index, dtype=float)
    for bt in test_df['building_type_c14'].unique():
        tr_mask = train_df['building_type_c14'] == bt
        ts_mask = test_df['building_type_c14'] == bt
        tr_sub  = train_df.loc[tr_mask, 'cvrmse']
        ts_idx  = test_df.index[ts_mask]
        if len(tr_sub) < 2:
            scores.loc[ts_idx] = 50.0
        else:
            mu = tr_sub.mean()
            sd = tr_sub.std(ddof=1)
            if sd == 0 or np.isnan(sd):
                scores.loc[ts_idx] = 50.0
            else:
                z = (test_df.loc[ts_mask, 'cvrmse'] - mu) / sd
                scores.loc[ts_idx] = (1.0 - norm.cdf(z)) * 100.0
    return scores.round()


def classify_l2(pattern_score: float, excess_cvrmse_pp: float) -> str:
    """Apply the 3-way classification rule."""
    if pattern_score >= 50:
        return 'NORMAL'
    elif excess_cvrmse_pp > BASELINE_THRESHOLD_PP:
        return 'ATYPICAL'
    else:
        return 'CV_DRIVEN'


def cohen_kappa(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Cohen's kappa for two categorical series."""
    labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
    n = len(y_true)
    if n == 0:
        return np.nan
    # Observed agreement
    po = (y_true == y_pred).mean()
    # Expected agreement
    pe = 0.0
    for lab in labels:
        pe += (y_true == lab).mean() * (y_pred == lab).mean()
    return (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 1.0


# ===========================================================================
# Load data
# ===========================================================================
def load_data():
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 611, f"Expected 611 rows, got {len(df)}"
    cbecs = df[~df['building_type'].isin(EXCLUDED_TYPES)].copy()
    assert len(cbecs) == 583, f"Expected 583 CBECS rows, got {len(cbecs)}"
    return df, cbecs


# ===========================================================================
# Analysis 1 – Threshold Sensitivity
# ===========================================================================

def analysis_1a(cbecs: pd.DataFrame):
    """
    Excess CVRMSE threshold sweep.
    Applied to: CBECS-mapped buildings with pattern_score < 50 (B/D quadrants).
    """
    print("\n" + "="*70)
    print("ANALYSIS 1A: Excess CVRMSE Threshold Sweep")
    print("  Population: 583 CBECS-mapped, pattern_score < 50 (quadrants B/D)")
    print("="*70)

    bd = cbecs[cbecs['pattern_score'] < 50].copy()
    print(f"  n (pattern_score < 50) = {len(bd)}")
    print(f"  Baseline NMBE threshold for direction split = ±{BASELINE_NMBE_PP}%\n")

    header = (
        f"{'Thr':>4}  {'n_ATYP':>7}  {'n_CV':>6}  "
        f"{'|NMBE|_ATYP':>11}  {'|NMBE|_CV':>9}  "
        f"{'Cohen_d':>8}  {'OVER%':>6}  {'UNDER%':>7}  {'NEUTRAL%':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for thr in EXCESS_THRESHOLDS:
        atypical  = bd[bd['excess_cvrmse'] * 100 > thr]
        cv_driven = bd[bd['excess_cvrmse'] * 100 <= thr]

        n_atyp = len(atypical)
        n_cv   = len(cv_driven)

        abs_nmbe_a = atypical['nmbe'].abs()
        abs_nmbe_c = cv_driven['nmbe'].abs()

        d = cohens_d(abs_nmbe_a, abs_nmbe_c)

        nmbe_thr = BASELINE_NMBE_PP / 100.0
        pct_over    = (atypical['nmbe'] > nmbe_thr).mean() * 100 if n_atyp > 0 else np.nan
        pct_under   = (atypical['nmbe'] < -nmbe_thr).mean() * 100 if n_atyp > 0 else np.nan
        pct_neutral = (atypical['nmbe'].abs() <= nmbe_thr).mean() * 100 if n_atyp > 0 else np.nan

        d_label = "(large)" if d >= 0.8 else "(medium)" if d >= 0.5 else "(small)"
        print(
            f"{thr:>3}pp  {n_atyp:>7}  {n_cv:>6}  "
            f"{abs_nmbe_a.mean()*100:>9.2f}%  {abs_nmbe_c.mean()*100:>7.2f}%  "
            f"{d:>7.3f}{d_label:>9}  {pct_over:>5.1f}%  {pct_under:>6.1f}%  {pct_neutral:>8.1f}%"
        )
        rows.append(dict(
            threshold_pp=thr,
            n_ATYPICAL=n_atyp, n_CV_DRIVEN=n_cv,
            mean_abs_nmbe_atypical_pct=round(abs_nmbe_a.mean() * 100, 4),
            mean_abs_nmbe_cv_pct=round(abs_nmbe_c.mean() * 100, 4),
            cohens_d=round(d, 4),
            pct_over=round(pct_over, 2),
            pct_under=round(pct_under, 2),
            pct_neutral=round(pct_neutral, 2),
        ))

    return pd.DataFrame(rows)


def analysis_1b(cbecs: pd.DataFrame):
    """
    NMBE direction threshold sweep for ATYPICAL buildings (5pp baseline).
    """
    print("\n" + "="*70)
    print("ANALYSIS 1B: NMBE Direction Threshold Sweep")
    print("  Population: ATYPICAL buildings (5pp excess threshold, pattern_score < 50)")
    print("="*70)

    bd = cbecs[cbecs['pattern_score'] < 50].copy()
    atypical = bd[bd['excess_cvrmse'] * 100 > BASELINE_THRESHOLD_PP]
    print(f"  n_ATYPICAL (5pp) = {len(atypical)}")
    print(f"  Mean NMBE = {atypical['nmbe'].mean()*100:.2f}%\n")

    header = f"{'NMBE_thr':>9}  {'n_OVER':>7}  {'n_UNDER':>8}  {'n_NEUTRAL':>10}  {'Mean_NMBE%':>11}"
    print(header)
    print("-" * len(header))

    rows = []
    for nt in NMBE_THRESHOLDS:
        nt_frac = nt / 100.0
        n_over    = (atypical['nmbe'] > nt_frac).sum()
        n_under   = (atypical['nmbe'] < -nt_frac).sum()
        n_neutral = (atypical['nmbe'].abs() <= nt_frac).sum()
        mean_nmbe = atypical['nmbe'].mean() * 100
        print(f"  +/-{nt:>2}%    {n_over:>7}  {n_under:>8}  {n_neutral:>10}  {mean_nmbe:>10.2f}%")
        rows.append(dict(
            nmbe_threshold_pct=nt,
            n_OVER=int(n_over), n_UNDER=int(n_under), n_NEUTRAL=int(n_neutral),
            mean_nmbe_pct=round(mean_nmbe, 4),
        ))
    return pd.DataFrame(rows)


def analysis_1c(cbecs: pd.DataFrame):
    """
    Cross-sensitivity heatmap: n_OVER for each (excess_thr, nmbe_thr) combination.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1C: Cross-Sensitivity Heatmap (n_OVER)")
    print("  Population: CBECS-mapped, pattern_score < 50")
    print("="*70)

    bd = cbecs[cbecs['pattern_score'] < 50].copy()

    # Print heatmap
    header_cols = ["  ".join([f"NMBE>{nt}%" for nt in NMBE_THRESHOLDS])]
    print(f"\n{'Excess_thr':>11}  " + "  ".join([f"NMBE>±{nt}%" for nt in NMBE_THRESHOLDS]))
    print("-" * 55)

    rows = []
    for et in EXCESS_THRESHOLDS:
        atypical_et = bd[bd['excess_cvrmse'] * 100 > et]
        row = {'excess_threshold_pp': et}
        parts = []
        for nt in NMBE_THRESHOLDS:
            n_over = int((atypical_et['nmbe'] > nt/100.0).sum())
            row[f'n_over_nmbe_{nt}pct'] = n_over
            parts.append(f"{n_over:>9}")
        print(f"{et:>8}pp   " + "  ".join(parts))
        rows.append(row)

    return pd.DataFrame(rows)


# ===========================================================================
# Analysis 2 – Leave-One-Site-Out Cross-Validation
# ===========================================================================

def analysis_2(df: pd.DataFrame, cbecs: pd.DataFrame):
    """
    Leave-one-site-out cross-validation across 4 sites.
    """
    print("\n" + "="*70)
    print("ANALYSIS 2: Leave-One-Site-Out Cross-Validation (Appendix E)")
    print("="*70)

    rows_e1 = []   # regression metrics per fold
    rows_e2 = []   # classification agreement per fold

    all_fold_details = []

    for site_out in SITES:
        train_all = df[df['site'] != site_out].copy()
        test_all  = df[df['site'] == site_out].copy()

        # Subset test to CBECS-mapped for classification comparison
        test_cbecs = test_all[~test_all['building_type'].isin(EXCLUDED_TYPES)].copy()

        # 1. Fit regression on ALL training buildings
        alpha, beta, r2_train = fit_regression(train_all)

        # 2. Compute R² on test set
        test_pred = alpha * test_all['cv'] + beta
        ss_res = ((test_all['cvrmse'] - test_pred) ** 2).sum()
        ss_tot = ((test_all['cvrmse'] - test_all['cvrmse'].mean()) ** 2).sum()
        r2_test = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # 3. Compute excess CVRMSE for test using fold regression
        test_cbecs = test_cbecs.copy()
        test_cbecs['excess_loso_pp'] = (
            test_cbecs['cvrmse'] - (alpha * test_cbecs['cv'] + beta)
        ) * 100.0

        # 4. Compute pattern score for test using training within-type stats
        test_cbecs['ps_loso'] = compute_pattern_scores(test_cbecs, train_all)

        # 5. Classify
        test_cbecs['l2_loso'] = test_cbecs.apply(
            lambda r: classify_l2(r['ps_loso'], r['excess_loso_pp']), axis=1
        )

        # 6. Metrics
        n_test_cbecs = len(test_cbecs)
        n_atyp_fold  = int((test_cbecs['l2_loso'] == 'ATYPICAL').sum())
        n_atyp_base  = int((test_cbecs['l2_cause'] == 'ATYPICAL').sum())
        agreement_pct = (test_cbecs['l2_loso'] == test_cbecs['l2_cause']).mean() * 100
        kappa = cohen_kappa(test_cbecs['l2_cause'], test_cbecs['l2_loso'])

        # Store E1 row
        rows_e1.append(dict(
            fold=f"Hold-out {site_out}",
            n_train=len(train_all),
            n_test=len(test_all),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            R2_train=round(r2_train, 4),
            R2_test=round(r2_test, 4),
        ))

        # Store E2 row
        rows_e2.append(dict(
            fold=f"Hold-out {site_out}",
            n_test_cbecs=n_test_cbecs,
            ATYPICAL_fold=n_atyp_fold,
            ATYPICAL_baseline=n_atyp_base,
            agreement_pct=round(agreement_pct, 1),
            cohen_kappa=round(kappa, 3),
        ))

        all_fold_details.append(test_cbecs)

    df_e1 = pd.DataFrame(rows_e1)
    df_e2 = pd.DataFrame(rows_e2)

    # Print Table E1
    print("\nTable E1: Regression Parameters per Fold")
    print("-" * 75)
    print(f"{'Fold':>20}  {'n_train':>7}  {'n_test':>7}  {'alpha':>7}  {'beta':>7}  {'R2_train':>9}  {'R2_test':>8}")
    print("-" * 75)
    for _, row in df_e1.iterrows():
        print(f"{row['fold']:>20}  {row['n_train']:>7}  {row['n_test']:>7}  "
              f"{row['alpha']:>7.4f}  {row['beta']:>7.4f}  "
              f"{row['R2_train']:>9.4f}  {row['R2_test']:>8.4f}")

    # Baseline reference
    baseline_alpha, baseline_beta, baseline_r2 = fit_regression(df)
    print(f"\n  Baseline (all 611): alpha={baseline_alpha:.4f}, beta={baseline_beta:.4f}, R2={baseline_r2:.4f}")

    # Print Table E2
    print("\nTable E2: Classification Agreement per Fold")
    print("-" * 75)
    print(f"{'Fold':>20}  {'n_test':>7}  {'ATYP_fold':>10}  {'ATYP_base':>10}  {'Agree%':>7}  {'Kappa':>7}")
    print("-" * 75)
    for _, row in df_e2.iterrows():
        print(f"{row['fold']:>20}  {row['n_test_cbecs']:>7}  "
              f"{row['ATYPICAL_fold']:>10}  {row['ATYPICAL_baseline']:>10}  "
              f"{row['agreement_pct']:>6.1f}%  {row['cohen_kappa']:>7.3f}")

    # Overall summary
    combined = pd.concat(all_fold_details)
    overall_agree = (combined['l2_loso'] == combined['l2_cause']).mean() * 100
    overall_kappa = cohen_kappa(combined['l2_cause'], combined['l2_loso'])
    print(f"\n  Overall (all folds pooled): n={len(combined)}, "
          f"Agreement={overall_agree:.1f}%, Kappa={overall_kappa:.3f}")

    # Confusion matrix
    print("\nPooled Confusion Matrix (rows=baseline, cols=LOSO fold):")
    labels = ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']
    conf = pd.crosstab(combined['l2_cause'], combined['l2_loso'],
                       rownames=['Baseline'], colnames=['LOSO'])
    for lab in labels:
        if lab not in conf.columns:
            conf[lab] = 0
        if lab not in conf.index:
            conf.loc[lab] = 0
    conf = conf.reindex(index=labels, columns=labels, fill_value=0)
    print(conf.to_string())

    return df_e1, df_e2, combined


# ===========================================================================
# Figures
# ===========================================================================

def plot_cohens_d(df_1a: pd.DataFrame, out_dir: str):
    """Figure D1: Cohen's d vs excess CVRMSE threshold."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    thresholds = df_1a['threshold_pp'].values
    d_values   = df_1a['cohens_d'].values

    ax.plot(thresholds, d_values, 'o-', color='#1f77b4', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgewidth=2)

    # Horizontal reference lines
    ax.axhline(0.8, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8, label='Large (d=0.8)')
    ax.axhline(0.5, color='#ff7f0e', linestyle='--', linewidth=1.2, alpha=0.8, label='Medium (d=0.5)')

    # Vertical marker at baseline threshold
    ax.axvline(5, color='#2ca02c', linestyle=':', linewidth=1.5, alpha=0.9, label='Baseline (5pp)')

    # Annotate each point
    for thr, d in zip(thresholds, d_values):
        ax.annotate(f"{d:.2f}", xy=(thr, d), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)

    ax.set_xlabel("Excess CVRMSE Threshold (percentage points)", fontsize=11)
    ax.set_ylabel("Cohen's d  (|NMBE|: ATYPICAL vs CV-Driven)", fontsize=11)
    # MDPI policy: no figure number/caption embedded in image
    ax.legend(fontsize=9)
    ax.set_xticks(thresholds)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'figure_d1_cohens_d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


# ===========================================================================
# Save results
# ===========================================================================

def save_csv(df: pd.DataFrame, filename: str, out_dir: str):
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Simple markdown table formatter (no tabulate dependency)."""
    cols = df.columns.tolist()
    rows = [cols]
    for _, row in df.iterrows():
        rows.append([str(v) for v in row.values])
    col_widths = [max(len(str(rows[r][c])) for r in range(len(rows))) for c in range(len(cols))]
    lines = []
    def fmt_row(r):
        return "| " + " | ".join(str(rows[r][c]).ljust(col_widths[c]) for c in range(len(cols))) + " |"
    lines.append(fmt_row(0))
    lines.append("| " + " | ".join("-" * col_widths[c] for c in range(len(cols))) + " |")
    for r in range(1, len(rows)):
        lines.append(fmt_row(r))
    return "\n".join(lines)


def save_markdown(df_1a, df_1b, df_1c, df_e1, df_e2, combined, out_dir: str):
    lines = []
    lines.append("# Appendix D & E Analysis Results")
    lines.append("")
    lines.append("## Appendix D: Threshold Sensitivity Analysis")
    lines.append("")
    lines.append("### D.1 Excess CVRMSE Threshold Sweep (Table D1)")
    lines.append("")
    lines.append("Population: 583 CBECS-mapped buildings, pattern_score < 50 (quadrants B/D)")
    lines.append(f"NMBE direction threshold fixed at +-{BASELINE_NMBE_PP}%")
    lines.append("")
    lines.append(_df_to_md_table(df_1a))
    lines.append("")
    lines.append("### D.2 NMBE Direction Threshold Sweep (Table D2)")
    lines.append("")
    lines.append("Population: ATYPICAL buildings (5pp excess threshold, pattern_score < 50), n=58")
    lines.append("")
    lines.append(_df_to_md_table(df_1b))
    lines.append("")
    lines.append("### D.3 Cross-Sensitivity Heatmap (Table D3: n_OVER)")
    lines.append("")
    lines.append("Cell values = number of buildings classified as OVER-consuming (NMBE > threshold)")
    lines.append("Population: CBECS-mapped, pattern_score < 50")
    lines.append("")
    lines.append(_df_to_md_table(df_1c))
    lines.append("")
    lines.append("## Appendix E: Leave-One-Site-Out Cross-Validation")
    lines.append("")
    lines.append("### Table E1: Regression Parameters per Fold")
    lines.append("")
    lines.append(_df_to_md_table(df_e1))
    lines.append("")
    lines.append("### Table E2: Classification Agreement per Fold")
    lines.append("")
    lines.append(_df_to_md_table(df_e2))
    lines.append("")

    # Overall pooled
    overall_agree = (combined['l2_loso'] == combined['l2_cause']).mean() * 100
    overall_kappa = cohen_kappa(combined['l2_cause'], combined['l2_loso'])
    lines.append("### E.3 Overall Summary (Pooled across all folds)")
    lines.append("")
    lines.append(f"- Total test buildings (CBECS-mapped, all folds): {len(combined)}")
    lines.append(f"- Overall agreement: {overall_agree:.1f}%")
    lines.append(f"- Overall Cohen's kappa: {overall_kappa:.3f}")
    lines.append("")

    # Confusion matrix
    labels = ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']
    conf = pd.crosstab(combined['l2_cause'], combined['l2_loso'],
                       rownames=['Baseline'], colnames=['LOSO'])
    for lab in labels:
        if lab not in conf.columns:
            conf[lab] = 0
        if lab not in conf.index:
            conf.loc[lab] = 0
    conf = conf.reindex(index=labels, columns=labels, fill_value=0)
    lines.append("### E.4 Pooled Confusion Matrix")
    lines.append("")
    lines.append(_df_to_md_table(conf.reset_index()))
    lines.append("")

    path = os.path.join(out_dir, 'appendix_d_e_results.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")
    return path


def save_sensitivity_csv(df_1a, df_1b, df_1c, out_dir):
    """Save all sensitivity parts to a single CSV (multi-section)."""
    path = os.path.join(out_dir, 'appendix_d_sensitivity.csv')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Part A: Excess CVRMSE Threshold Sweep\n")
        df_1a.to_csv(f, index=False)
        f.write("\n# Part B: NMBE Direction Threshold Sweep (ATYPICAL n=58 at 5pp)\n")
        df_1b.to_csv(f, index=False)
        f.write("\n# Part C: Cross-Sensitivity Heatmap (n_OVER)\n")
        df_1c.to_csv(f, index=False)
    print(f"  Saved: {path}")
    return path


def save_crossval_csv(df_e1, df_e2, out_dir):
    path = os.path.join(out_dir, 'appendix_e_crossval.csv')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# Table E1: Regression Parameters per Fold\n")
        df_e1.to_csv(f, index=False)
        f.write("\n# Table E2: Classification Agreement per Fold\n")
        df_e2.to_csv(f, index=False)
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("APPENDIX D & E ANALYSIS")
    print("MDPI Buildings Journal - Major Revision")
    print("=" * 70)
    print(f"\nData: {CSV_PATH}")
    print(f"Output: {OUT_DIR}")

    # Load data
    df, cbecs = load_data()
    print(f"\nLoaded: {len(df)} buildings total, {len(cbecs)} CBECS-mapped")

    # Verify baseline regression
    alpha0, beta0, r2_0 = fit_regression(df)
    print(f"\nBaseline regression (all 611): alpha={alpha0:.4f}, beta={beta0:.4f}, R2={r2_0:.4f}")
    print(f"  (Expected: slope=0.541, intercept=-0.030, R2=0.700)")

    # -----------------------------------------------------------------------
    # Analysis 1
    # -----------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# ANALYSIS 1: THRESHOLD SENSITIVITY (Appendix D)")
    print("#"*70)

    df_1a = analysis_1a(cbecs)
    df_1b = analysis_1b(cbecs)
    df_1c = analysis_1c(cbecs)

    # -----------------------------------------------------------------------
    # Analysis 2
    # -----------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# ANALYSIS 2: LEAVE-ONE-SITE-OUT CV (Appendix E)")
    print("#"*70)

    df_e1, df_e2, combined = analysis_2(df, cbecs)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    plot_cohens_d(df_1a, OUT_DIR)
    save_sensitivity_csv(df_1a, df_1b, df_1c, OUT_DIR)
    save_crossval_csv(df_e1, df_e2, OUT_DIR)
    save_markdown(df_1a, df_1b, df_1c, df_e1, df_e2, combined, OUT_DIR)

    print("\nDone. All outputs written to:", OUT_DIR)


if __name__ == '__main__':
    main()
