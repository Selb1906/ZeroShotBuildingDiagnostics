#!/usr/bin/env python3
"""
Box-Cox lambda sensitivity analysis.
Addresses Reviewer 1 Comment 10: framework dependency on Box-Cox transformation.

Tests building ranking stability under +/-10% and +/-20% lambda perturbations.

Usage: python scripts/08_boxcox_sensitivity.py
Requires: results/predictions_TransformerWithGaussian-L_bdg2_raw.csv (1.4GB)
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm
from scipy.special import boxcox
import os

PREDICTIONS_CSV = 'results/predictions_TransformerWithGaussian-L_bdg2_raw.csv'
EVAL_CSV = 'results/cbecs2018_c14_median_evaluation.csv'
LAMBDA_BASELINE = -0.06722072  # from metadata/transforms/boxcox.pkl
EXCLUDED = ['Other', 'Technology', 'Parking', 'Utility']
OUT_DIR = 'working'

os.makedirs(OUT_DIR, exist_ok=True)


def cvrmse_in_bc_space(actual, predicted, lam):
    """CVRMSE computed in Box-Cox transformed space."""
    mask = (actual > 0) & (predicted > 0)
    if mask.sum() < 100:
        return np.nan
    a_bc = boxcox(actual[mask], lam)
    p_bc = boxcox(predicted[mask], lam)
    rmse = np.sqrt(np.mean((a_bc - p_bc) ** 2))
    mean_a = np.mean(a_bc)
    return rmse / abs(mean_a) if abs(mean_a) > 1e-10 else np.nan


def main():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f'ERROR: {PREDICTIONS_CSV} not found')
        return

    print(f'Loading predictions...')
    pred = pd.read_csv(PREDICTIONS_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    variants = {
        '-20%': LAMBDA_BASELINE * 0.8,
        '-10%': LAMBDA_BASELINE * 0.9,
        'baseline': LAMBDA_BASELINE,
        '+10%': LAMBDA_BASELINE * 1.1,
        '+20%': LAMBDA_BASELINE * 1.2,
    }

    print('Computing CVRMSE for each lambda variant...')
    results = {}
    for variant, lam in variants.items():
        print(f'  lambda = {lam:.5f} ({variant})')
        results[variant] = {
            bid: cvrmse_in_bc_space(g['actual'].values, g['predicted'].values, lam)
            for bid, g in pred.groupby('building')
        }

    df_cv = pd.DataFrame(results).dropna()

    # Pattern Score recomputation per variant
    pattern_results = {}
    for variant in variants:
        df_eval = eval_df.merge(
            df_cv[[variant]].reset_index().rename(
                columns={'index': 'building', variant: 'cvrmse_new'}),
            on='building', how='left'
        )
        cbecs = df_eval[~df_eval['building_type'].isin(EXCLUDED)].copy()
        cbecs['ps_new'] = np.nan
        for bt in cbecs['building_type'].unique():
            mask = cbecs['building_type'] == bt
            sub = cbecs.loc[mask, 'cvrmse_new']
            if len(sub) < 2 or sub.isna().all():
                continue
            z = (sub - sub.mean()) / sub.std()
            cbecs.loc[mask, 'ps_new'] = norm.cdf(-z.values) * 100
        pattern_results[variant] = cbecs.set_index('building')['ps_new']

    # Rank stability table
    out_rows = []
    baseline_ps = pattern_results['baseline'].dropna()
    for variant, lam in variants.items():
        rho_cv, _ = spearmanr(df_cv['baseline'], df_cv[variant])
        ps_v = pattern_results[variant].reindex(baseline_ps.index)
        rho_ps, _ = spearmanr(baseline_ps, ps_v)
        out_rows.append({
            'variant': variant,
            'lambda': lam,
            'spearman_rho_cvrmse': rho_cv,
            'spearman_rho_pattern_score': rho_ps,
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(f'{OUT_DIR}/boxcox_sensitivity.csv', index=False)
    print(f'\nResults:\n{out_df.to_string(index=False)}')
    print(f'\nSaved: {OUT_DIR}/boxcox_sensitivity.csv')


if __name__ == '__main__':
    main()
