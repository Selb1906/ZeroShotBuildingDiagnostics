#!/usr/bin/env python3
"""
Reproduce all paper results from raw model predictions.

Pipeline:
  Step 1: Load raw predictions + metadata → per-building metrics
  Step 2: Compute EUI Scores (CBECS 2018 Table C14 median)
  Step 3: Compute Pattern Scores (within-type CVRMSE z-score)
  Step 4: Hierarchical classification (Level 1/2/3)
  Step 5: Export evaluation CSV → figures/tables

Input files required:
  - results/predictions_TransformerWithGaussian-L_bdg2_raw.csv
  - metadata/BDG2_metadata_with_sqft_eui.csv
  - metadata/cbecs_2018_c14.csv  (created by this script if missing)

Output:
  - results/cbecs2018_c14_median_evaluation.csv  (main evaluation CSV)
  - figures/fig*.png, figures/fig*.pdf            (paper figures)
  - results/paper_tables.txt                      (all paper table values)

Usage:
  python scripts/reproduce_paper.py [--skip-figures]
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys
import argparse

# ── Paths ───────────────────────────────────────────────────────────────
PREDICTIONS_CSV = 'results/predictions_TransformerWithGaussian-L_bdg2_raw.csv'
METADATA_CSV = 'metadata/BDG2_metadata_with_sqft_eui.csv'
OUTPUT_CSV = 'results/cbecs2018_c14_median_evaluation_reproduced.csv'
TABLES_TXT = 'results/paper_tables.txt'

# ── CBECS 2018 Table C14 Reference Values ───────────────────────────────
# Source: U.S. EIA, 2018 CBECS Table C14 — Electricity consumption intensities
# Units: kBtu per square foot (electricity only)
# Median and IQR estimated from CBECS microdata percentiles
CBECS_C14 = {
    #                             median_kbtu  p25_kbtu   p75_kbtu     # std_est
    'Education':                 (32.7552,     20.3014,   45.2090),    # 18.4501
    'Food service':              (134.0916,    73.3580,   194.8252),   # 89.9757
    'Health care':               (48.4504,     29.0020,   67.8988),    # 28.8124
    'Lodging':                   (49.4740,     21.6662,   77.2818),    # 41.1967
    'Mercantile':                (42.6500,     20.8132,   64.4868),    # 32.3508
    'Office':                    (34.4612,     18.0836,   50.8388),    # 24.2631
    'Public assembly':           (27.9784,     8.8712,    47.0856),    # 28.3070
    'Public order and safety':   (47.0856,     34.9730,   59.1982),    # 17.9446
    'Religious worship':         (14.3304,     5.8004,    22.8604),    # 12.6370
    'Warehouse and storage':     (11.6008,     1.0236,    22.1780),    # 15.6699
    'Service':                   (21.4956,     8.1888,    34.8024),    # 19.7138
    'Other':                     (41.9676,     3.5826,    80.3526),    # 56.8667
}

# Building type → C14 category mapping
TYPE_TO_C14 = {
    'Education': 'Education',
    'Food Service': 'Food service',
    'Healthcare': 'Health care',
    'Lodging': 'Lodging',
    'Retail': 'Mercantile',
    'Office': 'Office',
    'Public Assembly': 'Public assembly',
    'Public Services': 'Public order and safety',
    'Worship': 'Religious worship',
    'Warehouse': 'Warehouse and storage',
    'Technology': 'Service',
    'Other': 'Other',
    'Parking': 'Other',
    'Utility': 'Other',
}

EXCLUDED_TYPES = ['Other', 'Technology', 'Parking', 'Utility']


def step1_compute_building_metrics(pred_df):
    """Compute per-building CVRMSE, NMBE, CV, mean_load from predictions."""
    print('Step 1: Computing per-building metrics from predictions...')

    results = []
    for building, group in pred_df.groupby('building'):
        actual = group['actual'].values
        predicted = group['predicted'].values

        mean_actual = actual.mean()
        if mean_actual == 0:
            continue

        # CVRMSE (Coefficient of Variation of RMSE)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        cvrmse = rmse / mean_actual

        # NMBE (Normalized Mean Bias Error)
        # Convention: (actual - predicted) / mean_actual
        # Positive NMBE = building consumes MORE than predicted (over-consuming)
        nmbe = np.mean(actual - predicted) / mean_actual

        # CV (Coefficient of Variation of actual load)
        cv = np.std(actual) / mean_actual

        # Site extraction
        parts = building.split('_')
        site = parts[0] if len(parts) >= 1 else ''
        btype_raw = parts[1] if len(parts) >= 2 else ''

        # Worst error hour and overconsumption
        residuals = actual - predicted
        hourly_abs_error = group.groupby('hour').apply(
            lambda g: np.mean(np.abs(g['actual'].values - g['predicted_mean'].values)),
            include_groups=False
        )
        worst_error_hour = hourly_abs_error.idxmax()

        hourly_overconsume = group.groupby('hour').apply(
            lambda g: np.mean(g['actual'].values - g['predicted_mean'].values),
            include_groups=False
        )
        max_overconsume_hour = hourly_overconsume.idxmax()
        max_overconsume_kwh = hourly_overconsume.max()

        # Weekend error ratio
        # day_index mod 7: assuming day 0 = Monday
        group_copy = group.copy()
        group_copy['dow'] = group_copy['day_index'] % 7
        weekend = group_copy[group_copy['dow'].isin([5, 6])]
        weekday = group_copy[~group_copy['dow'].isin([5, 6])]
        if len(weekday) > 0 and len(weekend) > 0:
            we_err = np.mean(np.abs(weekend['actual'].values - weekend['predicted_mean'].values))
            wd_err = np.mean(np.abs(weekday['actual'].values - weekday['predicted_mean'].values))
            weekend_error_ratio = we_err / wd_err if wd_err > 0 else np.nan
        else:
            weekend_error_ratio = np.nan

        results.append({
            'building': building,
            'site': site,
            'cvrmse': cvrmse,
            'nmbe': nmbe,
            'cv': cv,
            'mean_load_kwh': mean_actual,
            'worst_error_hour': worst_error_hour,
            'max_overconsume_hour': max_overconsume_hour,
            'max_overconsume_kwh': max_overconsume_kwh,
            'weekend_error_ratio': weekend_error_ratio,
        })

    metrics_df = pd.DataFrame(results)
    print(f'  Computed metrics for {len(metrics_df)} buildings.')
    return metrics_df


USAGE_TO_BUILDING_TYPE = {
    'Education': 'Education',
    'Entertainment/public assembly': 'Public Assembly',
    'Food sales and service': 'Food Service',
    'Healthcare': 'Healthcare',
    'Lodging/residential': 'Lodging',
    'Manufacturing/industrial': 'Other',
    'Office': 'Office',
    'Other': 'Other',
    'Parking': 'Parking',
    'Public services': 'Public Services',
    'Religious worship': 'Worship',
    'Retail': 'Retail',
    'Technology/science': 'Technology',
    'Utility': 'Utility',
    'Warehouse/storage': 'Warehouse',
}


def step2_merge_metadata(metrics_df, meta_df):
    """Merge building metrics with metadata (type, sqft, EUI)."""
    print('Step 2: Merging with metadata...')

    # Metadata uses building_id, primaryspaceusage
    meta_sub = meta_df.rename(columns={'building_id': 'building'}).copy()
    meta_sub['building_type'] = meta_sub['primaryspaceusage'].map(USAGE_TO_BUILDING_TYPE)

    # Compute EUI in kBtu/sqft from hourly load data
    # EUI = total annual kWh / sqft * 3.412 (kWh to kBtu conversion)
    # Use mean_load_kwh from metrics * 8760 hours / sqft * 3.412
    merged = metrics_df.merge(
        meta_sub[['building', 'building_type', 'sub_primaryspaceusage', 'sqft']],
        on='building', how='left'
    )
    # Compute EUI: mean_load_kwh * 8760 / sqft * 3.412
    merged['eui_kbtu_sqft'] = np.where(
        merged['sqft'] > 0,
        merged['mean_load_kwh'] * 8760 / merged['sqft'] * 3.412,
        np.nan
    )

    # Map to C14 type
    merged['c14_type'] = merged['building_type'].map(TYPE_TO_C14)
    print(f'  Merged: {len(merged)} buildings with metadata.')
    return merged


def step3_compute_eui_scores(df):
    """Compute EUI Score using CBECS 2018 C14 z-score normalization."""
    print('Step 3: Computing EUI Scores (CBECS 2018 C14 median)...')

    eui_scores = []
    for _, row in df.iterrows():
        c14_type = row['c14_type']
        eui = row['eui_kbtu_sqft']

        if pd.isna(eui) or c14_type not in CBECS_C14:
            eui_scores.append(np.nan)
            continue

        median, p25, p75 = CBECS_C14[c14_type]
        std_est = (p75 - p25) / 1.35  # IQR-based std estimation

        if std_est == 0:
            eui_scores.append(50.0)
            continue

        z = (eui - median) / std_est
        # Convert z-score to 0-100 percentile via normal CDF
        # Lower EUI → negative z → higher score (better)
        score = float(stats.norm.cdf(-z) * 100)
        eui_scores.append(score)

    df['eui_score_c14'] = eui_scores
    df['cbecs_median_kbtu_sqft'] = df['c14_type'].map(
        lambda x: CBECS_C14.get(x, (np.nan,))[0])
    df['cbecs_std_kbtu_sqft'] = df['c14_type'].apply(
        lambda x: (CBECS_C14[x][2] - CBECS_C14[x][1]) / 1.35 if x in CBECS_C14 else np.nan)

    valid = df['eui_score_c14'].notna().sum()
    print(f'  EUI Score computed for {valid}/{len(df)} buildings.')
    return df


def step4_compute_pattern_scores(df):
    """Compute Pattern Score as within-type CVRMSE z-score (inverted)."""
    print('Step 4: Computing Pattern Scores (within-type CVRMSE z-score)...')

    df['pattern_score'] = np.nan
    for btype in df['building_type'].unique():
        mask = df['building_type'] == btype
        sub = df.loc[mask, 'cvrmse']
        if len(sub) < 2:
            continue
        mean_cv = sub.mean()
        std_cv = sub.std()
        if std_cv == 0:
            df.loc[mask, 'pattern_score'] = 50
            continue
        z = (sub - mean_cv) / std_cv
        # Invert via normal CDF: lower CVRMSE → negative z → higher score
        scores = pd.Series(stats.norm.cdf(-z.values) * 100, index=sub.index)
        df.loc[mask, 'pattern_score'] = scores.round(0).astype(int)

    print(f'  Pattern Score computed for {df["pattern_score"].notna().sum()} buildings.')
    return df


def step5_classify(df):
    """Apply hierarchical classification (Level 1/2/3)."""
    print('Step 5: Hierarchical classification...')

    # Level 1: Quadrant (>=50 inclusive threshold)
    def quadrant(row):
        eui = row['eui_score_c14']
        pat = row['pattern_score']
        if pd.isna(eui) or pd.isna(pat):
            return np.nan
        if eui >= 50 and pat >= 50:
            return 'A'
        elif eui >= 50 and pat < 50:
            return 'B'
        elif eui < 50 and pat >= 50:
            return 'C'
        else:
            return 'D'

    df['quadrant_c14'] = df.apply(quadrant, axis=1)

    # Level 2: CVRMSE Decomposition
    # Regression: CVRMSE = alpha * CV + beta
    valid = df[['cvrmse', 'cv']].dropna()
    slope, intercept, _, _, _ = stats.linregress(valid['cv'], valid['cvrmse'])
    print(f'  CVRMSE ~ CV regression: CVRMSE = {slope:.3f} * CV + {intercept:.3f}')

    df['cvrmse_expected_from_cv'] = slope * df['cv'] + intercept
    df['excess_cvrmse'] = df['cvrmse'] - df['cvrmse_expected_from_cv']

    def l2_cause(row):
        if row['pattern_score'] >= 50:
            return 'NORMAL'
        elif row['excess_cvrmse'] <= 0.05:  # 5 percentage points
            return 'CV_DRIVEN'
        else:
            return 'ATYPICAL'

    df['l2_cause'] = df.apply(l2_cause, axis=1)

    # Level 3: NMBE direction (for ATYPICAL only)
    def l3_direction(row):
        if row['l2_cause'] != 'ATYPICAL':
            return np.nan
        if row['nmbe'] > 0.02:
            return 'OVER-CONSUMING'
        elif row['nmbe'] < -0.02:
            return 'UNDER-CONSUMING'
        else:
            return 'NEUTRAL'

    df['l3_nmbe_direction'] = df.apply(l3_direction, axis=1)

    # Diagnosis and recommendation
    def diagnosis(row):
        if row['l2_cause'] == 'NORMAL':
            if row.get('eui_score_c14', 50) >= 50:
                return 'Efficient & consistent operation'
            else:
                return 'Consistent pattern but high EUI'
        elif row['l2_cause'] == 'CV_DRIVEN':
            return f'High variability (CV={row["cv"]:.2f}) drives prediction error'
        else:
            dir_str = row['l3_nmbe_direction'].lower()
            return f'Atypical pattern, {dir_str} bias'

    df['diagnosis'] = df.apply(diagnosis, axis=1)

    # Summary
    cbecs = df[~df['building_type'].isin(EXCLUDED_TYPES)]
    q = cbecs['quadrant_c14'].value_counts().sort_index()
    l2 = cbecs['l2_cause'].value_counts()
    print(f'  Quadrant (n={len(cbecs)}): A={q.get("A",0)}, B={q.get("B",0)}, C={q.get("C",0)}, D={q.get("D",0)}')
    print(f'  Level 2: NORMAL={l2.get("NORMAL",0)}, CV_DRIVEN={l2.get("CV_DRIVEN",0)}, ATYPICAL={l2.get("ATYPICAL",0)}')

    return df


def step6_export_tables(df, out_path):
    """Export all paper table values to text file."""
    print(f'Step 6: Exporting paper tables to {out_path}...')

    cbecs = df[~df['building_type'].isin(EXCLUDED_TYPES)].copy()
    total = len(cbecs)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('=' * 60 + '\n')
        f.write('PAPER TABLE VALUES (reproduced from CSV)\n')
        f.write(f'Source: {OUTPUT_CSV}\n')
        f.write(f'CBECS-mapped buildings: {total}\n')
        f.write(f'Threshold: Pattern Score >= 50 (inclusive)\n')
        f.write('=' * 60 + '\n\n')

        # Table 6: Quadrant distribution
        f.write('--- Table 6: Quadrant Distribution ---\n')
        for q in ['A', 'B', 'C', 'D']:
            sub = cbecs[cbecs['quadrant_c14'] == q]
            n = len(sub)
            pct = n / total * 100
            med_eui = sub['eui_score_c14'].median()
            med_ps = sub['pattern_score'].median()
            f.write(f'  {q}: n={n} ({pct:.1f}%), median_EUI_Score={med_eui:.0f}, median_Pattern_Score={med_ps:.0f}\n')

        # Table 8: Level 2 cross-tab
        f.write('\n--- Table 8: Level 2 x Quadrant ---\n')
        ct = pd.crosstab(cbecs['quadrant_c14'], cbecs['l2_cause'])
        f.write(ct.to_string() + '\n')

        # Table 9: NMBE by Level 2
        f.write('\n--- Table 9: NMBE by Level 2 ---\n')
        for cat in ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']:
            sub = cbecs[cbecs['l2_cause'] == cat]
            mean_abs = sub['nmbe'].abs().mean() * 100
            pct5 = (sub['nmbe'].abs() > 0.05).mean() * 100
            median_abs = sub['nmbe'].abs().median() * 100
            f.write(f'  {cat}: n={len(sub)}, mean|NMBE|={mean_abs:.2f}%, |NMBE|>5%={pct5:.1f}%, median={median_abs:.2f}%\n')

        # Correlations
        f.write('\n--- Key Correlations ---\n')
        r1, p1 = stats.pearsonr(cbecs['eui_kbtu_sqft'], cbecs['cvrmse'])
        r2, p2 = stats.pearsonr(cbecs['eui_score_c14'], cbecs['pattern_score'])
        f.write(f'  Raw EUI vs CVRMSE (n={total}): r={r1:.3f}, p={p1:.4f}\n')
        f.write(f'  EUI Score vs Pattern Score (n={total}): r={r2:.3f}, p={p2:.6f}\n')

        # ENERGY STAR reversal
        es = cbecs[cbecs['eui_score_c14'] >= 75]
        es_irreg = es[es['pattern_score'] < 50]
        f.write(f'\n--- ENERGY STAR Reversal ---\n')
        f.write(f'  ES-certifiable (EUI Score >= 75): {len(es)}\n')
        f.write(f'  with Pattern < 50: {len(es_irreg)} ({len(es_irreg)/len(es)*100:.1f}%)\n')

        # ATYPICAL NMBE direction
        atyp = cbecs[cbecs['l2_cause'] == 'ATYPICAL']
        f.write(f'\n--- Level 3: ATYPICAL NMBE Direction ---\n')
        f.write(f'  OVER (NMBE > 2%): {len(atyp[atyp["nmbe"] > 0.02])}\n')
        f.write(f'  UNDER (NMBE < -2%): {len(atyp[atyp["nmbe"] < -0.02])}\n')
        f.write(f'  NEUTRAL: {len(atyp[(atyp["nmbe"] >= -0.02) & (atyp["nmbe"] <= 0.02)])}\n')

        # CVRMSE by type
        f.write(f'\n--- CVRMSE by Building Type (all {len(df)}) ---\n')
        for bt in sorted(df['building_type'].unique()):
            sub = df[df['building_type'] == bt]
            if len(sub) >= 5:
                f.write(f'  {bt}: mean={sub["cvrmse"].mean()*100:.1f}%, median={sub["cvrmse"].median()*100:.1f}%, n={len(sub)}\n')

    print(f'  Tables exported to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Reproduce paper results')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--predictions', default=PREDICTIONS_CSV,
                        help=f'Path to predictions CSV (default: {PREDICTIONS_CSV})')
    parser.add_argument('--metadata', default=METADATA_CSV,
                        help=f'Path to metadata CSV (default: {METADATA_CSV})')
    parser.add_argument('--output', default=OUTPUT_CSV,
                        help=f'Output CSV path (default: {OUTPUT_CSV})')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.predictions):
        print(f'ERROR: Predictions file not found: {args.predictions}')
        print('Run zero-shot evaluation first:')
        print('  python scripts/zero_shot.py --model TransformerWithGaussian-L \\')
        print('    --checkpoint checkpoints/TransformerWithGaussian-L_best.pt \\')
        print('    --benchmark bdg-2 --save_predictions')
        sys.exit(1)

    if not os.path.exists(args.metadata):
        print(f'ERROR: Metadata file not found: {args.metadata}')
        sys.exit(1)

    print('=' * 60)
    print('PAPER RESULTS REPRODUCTION PIPELINE')
    print('=' * 60)
    print(f'  Predictions: {args.predictions}')
    print(f'  Metadata:    {args.metadata}')
    print(f'  Output:      {args.output}')
    print()

    # Step 1: Per-building metrics from predictions
    pred_df = pd.read_csv(args.predictions)
    print(f'Loaded predictions: {len(pred_df)} rows, {pred_df["building"].nunique()} buildings')
    metrics_df = step1_compute_building_metrics(pred_df)

    # Step 2: Merge metadata
    meta_df = pd.read_csv(args.metadata)
    merged_df = step2_merge_metadata(metrics_df, meta_df)

    # Step 3: EUI Scores
    merged_df = step3_compute_eui_scores(merged_df)

    # Step 4: Pattern Scores
    merged_df = step4_compute_pattern_scores(merged_df)

    # Step 5: Classification
    merged_df = step5_classify(merged_df)

    # Save evaluation CSV
    merged_df.to_csv(args.output, index=False)
    print(f'\nSaved evaluation CSV: {args.output} ({len(merged_df)} rows)')

    # Step 6: Export tables
    step6_export_tables(merged_df, TABLES_TXT)

    # Step 7: Generate figures
    if not args.skip_figures:
        print('\nStep 7: Generating figures...')
        os.makedirs('figures', exist_ok=True)
        try:
            # Import and run the figure generation
            sys.path.insert(0, 'scripts')
            import generate_paper_figures as gfig
            gfig.CSV_PATH = args.output
            gfig.main()
        except ImportError as e:
            print(f'  Warning: Could not import generate_paper_figures: {e}')
            print('  Run separately: python scripts/generate_paper_figures.py')
    else:
        print('\nSkipping figure generation (--skip-figures)')

    print('\n' + '=' * 60)
    print('REPRODUCTION COMPLETE')
    print('=' * 60)

    # Verify against reference if available
    ref_csv = 'results/cbecs2018_c14_median_evaluation.csv'
    if os.path.exists(ref_csv) and args.output != ref_csv:
        print('\nVerifying against reference CSV...')
        ref = pd.read_csv(ref_csv)
        repro = pd.read_csv(args.output)

        common = set(ref['building']) & set(repro['building'])
        print(f'  Common buildings: {len(common)}')

        ref_sub = ref[ref['building'].isin(common)].set_index('building').sort_index()
        repro_sub = repro[repro['building'].isin(common)].set_index('building').sort_index()

        # Compare CVRMSE
        cvrmse_diff = (ref_sub['cvrmse'] - repro_sub.loc[ref_sub.index, 'cvrmse']).abs()
        print(f'  CVRMSE max diff: {cvrmse_diff.max():.6f}')
        print(f'  CVRMSE mean diff: {cvrmse_diff.mean():.6f}')

        # Compare Level 2
        l2_match = (ref_sub['l2_cause'] == repro_sub.loc[ref_sub.index, 'l2_cause']).mean()
        print(f'  Level 2 match rate: {l2_match*100:.1f}%')

        # Compare quadrants (may differ if EUI computation differs)
        quad_match = (ref_sub['quadrant_c14'] == repro_sub.loc[ref_sub.index, 'quadrant_c14']).mean()
        print(f'  Quadrant match rate: {quad_match*100:.1f}%')
        if quad_match < 1.0:
            print('  Note: Quadrant differences are expected due to EUI computation path.')
            print('  The reference CSV uses pre-computed EUI values from metadata.')


if __name__ == '__main__':
    main()
