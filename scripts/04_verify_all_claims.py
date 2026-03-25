#!/usr/bin/env python3
"""
Verify all quantitative claims in the paper against CSV ground truth.
Flags any discrepancy as FAIL.

Usage: python scripts/verify_all_claims.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys

CSV = 'results/cbecs2018_c14_median_evaluation.csv'
EXCLUDED = ['Other', 'Technology', 'Parking', 'Utility']

df = pd.read_csv(CSV)
cbecs = df[~df['building_type'].isin(EXCLUDED)]
n_total = len(cbecs)
fails = 0

def check(name, expected, actual, tol=0.01):
    global fails
    ok = abs(expected - actual) <= tol if isinstance(expected, (int, float)) else expected == actual
    status = 'OK' if ok else 'FAIL'
    if not ok:
        fails += 1
    print(f'  [{status:4s}] {name}: expected={expected}, actual={actual}')


print(f'Source: {CSV} ({len(df)} total, {n_total} CBECS-mapped)')
print()

# ── 1. Quadrant Counts ──
print('=== Quadrant Distribution (>=50) ===')
q = cbecs['quadrant_c14'].value_counts()
check('A', 122, q.get('A', 0))
check('B', 107, q.get('B', 0))
check('C', 250, q.get('C', 0))
check('D', 104, q.get('D', 0))

# ── 2. Level 2 ──
print('\n=== Level 2 ===')
l2 = cbecs['l2_cause'].value_counts()
check('NORMAL', 372, l2.get('NORMAL', 0))
check('CV_DRIVEN', 153, l2.get('CV_DRIVEN', 0))
check('ATYPICAL', 58, l2.get('ATYPICAL', 0))

# ── 3. Cross-tab ──
print('\n=== Level 2 x Quadrant ===')
ct = pd.crosstab(cbecs['quadrant_c14'], cbecs['l2_cause'])
check('B-CV_DRIVEN', 81, ct.loc['B', 'CV_DRIVEN'])
check('B-ATYPICAL', 26, ct.loc['B', 'ATYPICAL'])
check('D-CV_DRIVEN', 72, ct.loc['D', 'CV_DRIVEN'])
check('D-ATYPICAL', 32, ct.loc['D', 'ATYPICAL'])

# ── 4. Correlations ──
print('\n=== Correlations ===')
r_raw, p_raw = stats.pearsonr(cbecs['eui_kbtu_sqft'], cbecs['cvrmse'])
r_score, p_score = stats.pearsonr(cbecs['eui_score_c14'], cbecs['pattern_score'])
check('r_raw_EUI_CVRMSE', -0.082, round(r_raw, 3), tol=0.001)
check('r_score', -0.291, round(r_score, 3), tol=0.001)

# ── 5. CV-CVRMSE Regression ──
print('\n=== CV-CVRMSE Regression ===')
slope, intercept, r, p, se = stats.linregress(df['cv'], df['cvrmse'])
check('slope', 0.541, round(slope, 3), tol=0.001)
check('intercept', -0.030, round(intercept, 3), tol=0.001)
check('R_squared', 0.700, round(r**2, 3), tol=0.001)

# ── 6. NMBE by Level 2 ──
print('\n=== NMBE by Level 2 ===')
for cat, exp_mean, exp_pct5 in [('NORMAL', 0.76, 0.3), ('CV_DRIVEN', 1.41, 1.3), ('ATYPICAL', 5.29, 31.0)]:
    sub = cbecs[cbecs['l2_cause'] == cat]
    mean_abs = sub['nmbe'].abs().mean() * 100
    pct5 = (sub['nmbe'].abs() > 0.05).mean() * 100
    check(f'{cat}_mean|NMBE|', exp_mean, round(mean_abs, 2), tol=0.05)
    check(f'{cat}_%>5%', exp_pct5, round(pct5, 1), tol=0.5)

# ── 7. Mann-Whitney U ──
print('\n=== Mann-Whitney U ===')
cv_d = cbecs[cbecs['l2_cause'] == 'CV_DRIVEN']['nmbe'].abs()
atyp = cbecs[cbecs['l2_cause'] == 'ATYPICAL']['nmbe'].abs()
u, p = stats.mannwhitneyu(atyp, cv_d, alternative='two-sided')
check('U', 6326, int(u), tol=10)
check('p < 0.001', True, bool(p < 0.001))

# ── 8. Table 10: NMBE Direction ──
print('\n=== Table 10: NMBE Direction ===')
atyp_all = cbecs[cbecs['l2_cause'] == 'ATYPICAL']
check('OVER', 25, len(atyp_all[atyp_all['nmbe'] > 0.02]))
check('NEUTRAL', 23, len(atyp_all[(atyp_all['nmbe'] >= -0.02) & (atyp_all['nmbe'] <= 0.02)]))
check('UNDER', 10, len(atyp_all[atyp_all['nmbe'] < -0.02]))

# ── 9. ENERGY STAR Reversal ──
print('\n=== ENERGY STAR Reversal ===')
es = cbecs[cbecs['eui_score_c14'] >= 75]
es_irreg = es[es['pattern_score'] < 50]
check('ES_certifiable', 85, len(es))
check('ES_with_Pattern<50', 55, len(es_irreg))
check('ES_reversal_pct', 64.7, round(len(es_irreg) / len(es) * 100, 1), tol=0.1)

# ── 10. Reversal Cases ──
print('\n=== Reversal Cases ===')
type_cv_med = cbecs.groupby('building_type')['cv'].median()
cc = cbecs.copy()
cc['cv_above'] = cc.apply(lambda r: r['cv'] > type_cv_med[r['building_type']], axis=1)
c1 = cc[cc['cv_above'] & (cc['pattern_score'] >= 50)]
c2 = cc[~cc['cv_above'] & (cc['pattern_score'] < 50)]
check('Case1', 102, len(c1))
check('Case2', 23, len(c2))
check('Case3(=B)', 107, len(cbecs[cbecs['quadrant_c14'] == 'B']))
check('Case4(=C)', 250, len(cbecs[cbecs['quadrant_c14'] == 'C']))

# ── 11. Table 4: CVRMSE by Type ──
print('\n=== Table 4: CVRMSE by Type (mean%) ===')
for bt, exp in [('Parking', 10.7), ('Lodging', 14.5), ('Office', 16.7),
                ('Public Services', 17.3), ('Education', 17.0), ('Public Assembly', 28.9)]:
    actual = df[df['building_type'] == bt]['cvrmse'].mean() * 100
    check(bt, exp, round(actual, 1), tol=0.15)

# ── 12. Table 5: Causal Decomposition ──
print('\n=== Table 5: Causal Decomposition (CVRMSE>20%) ===')
hc = df[df['cvrmse'] > 0.20].copy()
type_cv_median = df.groupby('building_type')['cv'].median()
hc['cv_above'] = hc.apply(lambda r: r['cv'] > type_cv_median[r['building_type']], axis=1)
p25 = df['mean_load_kwh'].quantile(0.25)
hc['low_denom'] = hc['mean_load_kwh'] < p25
hc['is_atypical'] = hc['excess_cvrmse'] > 0.05
check('n_high', 185, len(hc))
check('HIGH_CV', 162, int(hc['cv_above'].sum()))
check('LOW_DENOM', 88, int(hc['low_denom'].sum()))
check('ATYPICAL_in_high', 62, int(hc['is_atypical'].sum()))

# ── 13. Best Practice ──
print('\n=== Best Practice (UNDER ATYPICAL) ===')
under_atyp = cbecs[(cbecs['l2_cause'] == 'ATYPICAL') & (cbecs['nmbe'] < -0.02)]
check('UNDER_ATYPICAL', 10, len(under_atyp))
under_valid = under_atyp[under_atyp['mean_load_kwh'] >= 5]
check('after_load_filter', 9, len(under_valid))

# ── Summary ──
print(f'\n{"="*50}')
print(f'TOTAL CHECKS: {fails} FAIL out of {50-fails+fails}')
if fails == 0:
    print('ALL CHECKS PASSED')
else:
    print(f'{fails} CHECKS FAILED — review above')
sys.exit(fails)
