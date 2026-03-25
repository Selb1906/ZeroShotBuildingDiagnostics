#!/usr/bin/env python3
"""
Extract and verify hourly residual patterns for Table 12 best-practice buildings.
Generates per-building hourly profiles and pattern classification.

Usage: python scripts/analyze_hourly_patterns.py
"""
import pandas as pd
import numpy as np
import os

PREDICTIONS_CSV = 'results/predictions_TransformerWithGaussian-L_bdg2_raw.csv'
EVAL_CSV = 'results/cbecs2018_c14_median_evaluation.csv'
OUT_DIR = 'results/hourly_patterns'

# Table 12 buildings
BEST_PRACTICE = [
    'Panther_office_Danica',
    'Panther_education_Scarlett',
    'Bear_assembly_Beatrice',
    'Panther_education_Janis',
    'Panther_office_Brent',
    'Fox_assembly_Boyce',
    'Fox_assembly_Bradley',
    'Rat_office_Avis',
    'Bear_education_Derek',
]


def analyze_building(pred_df, building_id):
    """Compute hourly residual profile for a single building."""
    g = pred_df[pred_df['building'] == building_id]
    if len(g) == 0:
        return None

    actual = g['actual'].values
    predicted = g['predicted'].values

    # Per-building overall metrics
    mean_actual = actual.mean()
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    cvrmse = rmse / mean_actual
    nmbe = np.mean(actual - predicted) / mean_actual

    # Hourly profile
    hourly = g.groupby('hour').agg(
        mean_actual=('actual', 'mean'),
        mean_predicted=('predicted', 'mean'),
        mean_residual=pd.NamedAgg(column='actual', aggfunc=lambda x: np.mean(
            g.loc[x.index, 'actual'].values - g.loc[x.index, 'predicted'].values
        )),
    ).reset_index()

    # Simpler residual calculation
    g_copy = g.copy()
    g_copy['residual'] = g_copy['actual'] - g_copy['predicted']
    hourly_residual = g_copy.groupby('hour')['residual'].mean()

    # Pattern classification
    # Peak suppression: negative residuals during business hours (09-16)
    biz_hours = hourly_residual.loc[9:16]
    off_hours = pd.concat([hourly_residual.loc[0:6], hourly_residual.loc[20:23]])

    # Delayed morning: large negative residuals in 07-10
    morning = hourly_residual.loc[7:10]

    # Evening shutdown: large negative residuals in 19-06
    evening_night = pd.concat([hourly_residual.loc[19:23], hourly_residual.loc[0:5]])

    # Determine dominant pattern
    peak_score = -biz_hours.mean() if biz_hours.mean() < 0 else 0
    morning_score = -morning.mean() if morning.mean() < 0 else 0
    evening_score = -evening_night.mean() if evening_night.mean() < 0 else 0

    if peak_score > morning_score and peak_score > evening_score:
        pattern = 'Peak Suppression'
        peak_hour_range = f'{biz_hours.idxmin():02d}:00-{biz_hours.idxmin()+2:02d}:00'
        peak_saving = biz_hours.min()
    elif morning_score > evening_score:
        pattern = 'Delayed Morning Start'
        peak_hour_range = f'{morning.idxmin():02d}:00-{morning.idxmin()+2:02d}:00'
        peak_saving = morning.min()
    else:
        pattern = 'Evening/Overnight Shutdown'
        peak_hour_range = f'{evening_night.idxmin():02d}:00-{evening_night.idxmin()+2:02d}:00'
        peak_saving = evening_night.min()

    return {
        'building': building_id,
        'cvrmse': cvrmse,
        'nmbe': nmbe,
        'mean_load_kwh': mean_actual,
        'pattern': pattern,
        'peak_hour_range': peak_hour_range,
        'peak_saving_kwh': peak_saving,
        'hourly_residual': hourly_residual,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading predictions...')
    pred_df = pd.read_csv(PREDICTIONS_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    print(f'Analyzing {len(BEST_PRACTICE)} best-practice buildings...\n')

    results = []
    for bid in BEST_PRACTICE:
        result = analyze_building(pred_df, bid)
        if result is None:
            print(f'  WARNING: {bid} not found in predictions')
            continue

        # Get metadata from eval CSV
        eval_row = eval_df[eval_df['building'] == bid]
        if len(eval_row) > 0:
            eval_row = eval_row.iloc[0]
            btype = eval_row['building_type']
            sqft = eval_row['sqft']
            sub_type = eval_row['sub_primaryspaceusage']
        else:
            btype = sqft = sub_type = 'N/A'

        print(f'  {bid}')
        print(f'    Type: {btype} ({sub_type}), Sqft: {sqft}')
        print(f'    CVRMSE: {result["cvrmse"]*100:.1f}%, NMBE: {result["nmbe"]*100:+.1f}%')
        print(f'    Pattern: {result["pattern"]}')
        print(f'    Peak saving: {result["peak_saving_kwh"]:.1f} kWh/hr at {result["peak_hour_range"]}')
        print()

        # Save hourly profile
        hr = result['hourly_residual']
        hr.to_csv(os.path.join(OUT_DIR, f'{bid}_hourly_residual.csv'))

        results.append({
            'building': bid,
            'building_type': btype,
            'sub_type': sub_type,
            'sqft': sqft,
            'cvrmse_pct': round(result['cvrmse'] * 100, 1),
            'nmbe_pct': round(result['nmbe'] * 100, 1),
            'mean_load_kwh': round(result['mean_load_kwh'], 1),
            'pattern': result['pattern'],
            'peak_saving_kwh': round(result['peak_saving_kwh'], 1),
            'peak_hour_range': result['peak_hour_range'],
        })

    # Summary table
    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(OUT_DIR, 'best_practice_summary.csv'), index=False)
    print(f'Summary saved to {OUT_DIR}/best_practice_summary.csv')

    # Pattern counts
    patterns = summary['pattern'].value_counts()
    print(f'\nPattern distribution:')
    for p, n in patterns.items():
        buildings = summary[summary['pattern'] == p]['building'].tolist()
        print(f'  {p}: {n} buildings ({", ".join(b.split("_")[-1] for b in buildings)})')


if __name__ == '__main__':
    main()
