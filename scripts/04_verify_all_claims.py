#!/usr/bin/env python3
"""
Compute and report all paper-reportable values from CSV.
No hardcoded expected values — this script IS the source of truth.
The paper must match these outputs.

Usage: python scripts/04_verify_all_claims.py
"""
import pandas as pd
import numpy as np
from scipy import stats

CSV = 'results/cbecs2018_c14_median_evaluation.csv'
EXCLUDED = ['Other', 'Technology', 'Parking', 'Utility']

df = pd.read_csv(CSV)
cbecs = df[~df['building_type'].isin(EXCLUDED)].copy()
n_all = len(df)
n_cbecs = len(cbecs)

print(f'Source: {CSV}')
print(f'Total buildings: {n_all}')
print(f'CBECS-mapped: {n_cbecs} (excluded: {", ".join(EXCLUDED)})')
print(f'Threshold: quadrant_c14 column as-is from CSV')
print()

# ═══════════════════════════════════════════════════════════
# 1. QUADRANT DISTRIBUTION
# ═══════════════════════════════════════════════════════════
print('=== Table 6: Quadrant Distribution ===')
q = cbecs['quadrant_c14'].value_counts().sort_index()
for quad in ['A', 'B', 'C', 'D']:
    n = q.get(quad, 0)
    print(f'  {quad}: {n} ({n/n_cbecs*100:.1f}%)')
print(f'  Total: {q.sum()}')
print()

# Median scores per quadrant
print('  Quadrant | Median EUI Score | Median Pattern Score')
for quad in ['A', 'B', 'C', 'D']:
    sub = cbecs[cbecs['quadrant_c14'] == quad]
    print(f'  {quad}        | {sub["eui_score_c14"].median():.0f}               | {sub["pattern_score"].median():.0f}')
print()

# ═══════════════════════════════════════════════════════════
# 2. LEVEL 2 CLASSIFICATION
# ═══════════════════════════════════════════════════════════
print('=== Table 8: Level 2 ===')
l2 = cbecs['l2_cause'].value_counts()
for cat in ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']:
    n = l2.get(cat, 0)
    print(f'  {cat}: {n} ({n/n_cbecs*100:.1f}%)')
print()

# Cross-tab
print('  Level 2 x Quadrant:')
ct = pd.crosstab(cbecs['quadrant_c14'], cbecs['l2_cause'])
for col in ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[['NORMAL', 'CV_DRIVEN', 'ATYPICAL']]
print(ct.to_string())
bd = cbecs[cbecs['quadrant_c14'].isin(['B', 'D'])]
n_bd = len(bd)
n_cv = l2.get('CV_DRIVEN', 0)
n_at = l2.get('ATYPICAL', 0)
print(f'\n  B+D total: {n_bd}')
print(f'  CV_DRIVEN: {n_cv} ({n_cv/n_bd*100:.1f}% of B+D)')
print(f'  ATYPICAL:  {n_at} ({n_at/n_bd*100:.1f}% of B+D; {n_at/n_cbecs*100:.1f}% of {n_cbecs})')
print()

# ═══════════════════════════════════════════════════════════
# 3. CORRELATIONS
# ═══════════════════════════════════════════════════════════
print('=== Correlations ===')
r_raw, p_raw = stats.pearsonr(cbecs['eui_kbtu_sqft'], cbecs['cvrmse'])
r_score, p_score = stats.pearsonr(cbecs['eui_score_c14'], cbecs['pattern_score'])
print(f'  Raw EUI vs CVRMSE (n={n_cbecs}): r={r_raw:.3f}, p={p_raw:.4f}, R2={r_raw**2:.3f}')
print(f'  EUI Score vs Pattern Score (n={n_cbecs}): r={r_score:.3f}, p={p_score:.2e}, R2={r_score**2:.3f}')

r_all, p_all = stats.pearsonr(df['eui_kbtu_sqft'], df['cvrmse'])
print(f'  Raw EUI vs CVRMSE (n={n_all}, all): r={r_all:.3f}, p={p_all:.4f}')
print()

# ═══════════════════════════════════════════════════════════
# 4. CV-CVRMSE REGRESSION
# ═══════════════════════════════════════════════════════════
print('=== CV-CVRMSE Regression (n={}) ==='.format(n_all))
slope, intercept, r, p, se = stats.linregress(df['cv'], df['cvrmse'])
print(f'  CVRMSE = {slope:.3f} * CV + {intercept:.3f}')
print(f'  R2 = {r**2:.3f}, p = {p:.2e}')
print()

# ═══════════════════════════════════════════════════════════
# 5. TABLE 4: CVRMSE BY TYPE
# ═══════════════════════════════════════════════════════════
print('=== Table 4: CVRMSE by Building Type (all {}) ==='.format(n_all))
print(f'  {"Type":<25} {"n":>4} {"Mean":>7} {"Median":>7} {"Std":>7} {"IQR":>7}')
for bt in ['Parking', 'Lodging', 'Office', 'Public Services', 'Education',
           'Public Assembly']:
    sub = df[df['building_type'] == bt]['cvrmse'] * 100
    if len(sub) >= 5:
        print(f'  {bt:<25} {len(sub):>4} {sub.mean():>6.1f}% {sub.median():>6.1f}% {sub.std():>6.1f}% {sub.quantile(0.75)-sub.quantile(0.25):>6.1f}%')

# Other = all remaining
other_types = [t for t in df['building_type'].unique() if t not in
               ['Parking', 'Lodging', 'Office', 'Public Services', 'Education', 'Public Assembly']]
other = df[df['building_type'].isin(other_types)]['cvrmse'] * 100
print(f'  {"Other (combined)":<25} {len(other):>4} {other.mean():>6.1f}% {other.median():>6.1f}% {other.std():>6.1f}% {other.quantile(0.75)-other.quantile(0.25):>6.1f}%')

all_cv = df['cvrmse'] * 100
print(f'  {"All buildings":<25} {len(all_cv):>4} {all_cv.mean():>6.1f}% {all_cv.median():>6.1f}% {all_cv.std():>6.1f}% {all_cv.quantile(0.75)-all_cv.quantile(0.25):>6.1f}%')
print()

# ═══════════════════════════════════════════════════════════
# 6. TABLE 5: CAUSAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════
print('=== Table 5: Causal Decomposition (CVRMSE > 20%) ===')
hc = df[df['cvrmse'] > 0.20].copy()
n_hc = len(hc)
print(f'  Buildings with CVRMSE > 20%: {n_hc}')

type_cv_median = df.groupby('building_type')['cv'].median()
hc['cv_above'] = hc.apply(lambda r: r['cv'] > type_cv_median[r['building_type']], axis=1)
p25 = df['mean_load_kwh'].quantile(0.25)
hc['low_denom'] = hc['mean_load_kwh'] < p25
hc['is_atypical'] = hc['excess_cvrmse'] > 0.05
cv_only = hc['cv_above'] & ~hc['low_denom'] & ~hc['is_atypical']

print(f'  HIGH_CV (CV > type median): {int(hc["cv_above"].sum())} ({hc["cv_above"].mean()*100:.1f}%)')
print(f'  LOW_DENOM (mean < p25={p25:.1f} kWh): {int(hc["low_denom"].sum())} ({hc["low_denom"].mean()*100:.1f}%)')
print(f'  ATYPICAL (Excess > 5pp): {int(hc["is_atypical"].sum())} ({hc["is_atypical"].mean()*100:.1f}%)')
print(f'  CV_DRIVEN only (single cause): {int(cv_only.sum())} ({cv_only.mean()*100:.1f}%)')
print()

# ═══════════════════════════════════════════════════════════
# 7. NMBE BY LEVEL 2 (Table 9)
# ═══════════════════════════════════════════════════════════
print('=== Table 9: NMBE by Level 2 ===')
print(f'  {"Group":<12} {"n":>4} {"Mean|NMBE|":>11} {"%>5%":>6} {"Median":>8}')
for cat in ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']:
    sub = cbecs[cbecs['l2_cause'] == cat]
    mean_abs = sub['nmbe'].abs().mean() * 100
    pct5 = (sub['nmbe'].abs() > 0.05).mean() * 100
    median_abs = sub['nmbe'].abs().median() * 100
    print(f'  {cat:<12} {len(sub):>4} {mean_abs:>10.2f}% {pct5:>5.1f}% {median_abs:>7.2f}%')
print()

# ═══════════════════════════════════════════════════════════
# 8. MANN-WHITNEY U
# ═══════════════════════════════════════════════════════════
print('=== Mann-Whitney U (CV_DRIVEN vs ATYPICAL |NMBE|) ===')
cv_d = cbecs[cbecs['l2_cause'] == 'CV_DRIVEN']['nmbe'].abs()
atyp = cbecs[cbecs['l2_cause'] == 'ATYPICAL']['nmbe'].abs()
u, p = stats.mannwhitneyu(atyp, cv_d, alternative='two-sided')
print(f'  U = {int(u)}, p = {p:.2e}')
print(f'  n_ATYPICAL={len(atyp)}, n_CV_DRIVEN={len(cv_d)}')
print()

# ═══════════════════════════════════════════════════════════
# 9. TABLE 10: NMBE DIRECTION (ATYPICAL)
# ═══════════════════════════════════════════════════════════
print('=== Table 10: NMBE Direction (ATYPICAL only) ===')
atyp_all = cbecs[cbecs['l2_cause'] == 'ATYPICAL']
n_atyp = len(atyp_all)
over = atyp_all[atyp_all['nmbe'] > 0.02]
under = atyp_all[atyp_all['nmbe'] < -0.02]
neutral = atyp_all[(atyp_all['nmbe'] >= -0.02) & (atyp_all['nmbe'] <= 0.02)]
print(f'  OVER-CONSUMING:  {len(over)} ({len(over)/n_atyp*100:.1f}%), mean NMBE={over["nmbe"].mean()*100:+.1f}%, mean Excess={over["excess_cvrmse"].mean()*100:.1f}pp')
print(f'  NEUTRAL:         {len(neutral)} ({len(neutral)/n_atyp*100:.1f}%), mean NMBE={neutral["nmbe"].mean()*100:+.1f}%, mean Excess={neutral["excess_cvrmse"].mean()*100:.1f}pp')
print(f'  UNDER-CONSUMING: {len(under)} ({len(under)/n_atyp*100:.1f}%), mean NMBE={under["nmbe"].mean()*100:+.1f}%, mean Excess={under["excess_cvrmse"].mean()*100:.1f}pp')
print(f'  All ATYPICAL:    {n_atyp}, mean NMBE={atyp_all["nmbe"].mean()*100:+.1f}%, mean Excess={atyp_all["excess_cvrmse"].mean()*100:.1f}pp')
print()

# ═══════════════════════════════════════════════════════════
# 10. ENERGY STAR REVERSAL
# ═══════════════════════════════════════════════════════════
print('=== ENERGY STAR Reversal ===')
es = cbecs[cbecs['eui_score_c14'] >= 75]
es_irreg = es[es['pattern_score'] < 50]
print(f'  ES-certifiable (EUI Score >= 75): {len(es)}')
print(f'  with Pattern Score < 50: {len(es_irreg)} ({len(es_irreg)/len(es)*100:.1f}%)')
atyp_in_es = es_irreg[es_irreg['l2_cause'] == 'ATYPICAL']
print(f'  ATYPICAL within irregular ES: {len(atyp_in_es)}')
print(f'  CV_DRIVEN within irregular ES: {len(es_irreg) - len(atyp_in_es)}')
print()

# ═══════════════════════════════════════════════════════════
# 11. REVERSAL CASES (Table 11)
# ═══════════════════════════════════════════════════════════
print('=== Table 11: Reversal Cases ===')
type_cv_med = cbecs.groupby('building_type')['cv'].median()
cc = cbecs.copy()
cc['cv_above'] = cc.apply(lambda r: r['cv'] > type_cv_med[r['building_type']], axis=1)

# Case 1/2 use the SAME threshold as quadrant (from quadrant_c14 column)
# A/C = pattern is good, B/D = pattern is bad
c1 = cc[cc['cv_above'] & cc['quadrant_c14'].isin(['A', 'C'])]  # CV high but good pattern
c2 = cc[~cc['cv_above'] & cc['quadrant_c14'].isin(['B', 'D'])]  # CV low but bad pattern
c3 = cbecs[cbecs['quadrant_c14'] == 'B']
c4 = cbecs[cbecs['quadrant_c14'] == 'C']
union = len(set(c1.index) | set(c2.index) | set(c3.index) | set(c4.index))

print(f'  Case 1 (CV high, good pattern): {len(c1)} ({len(c1)/n_cbecs*100:.1f}%)')
print(f'  Case 2 (CV low, bad pattern):   {len(c2)} ({len(c2)/n_cbecs*100:.1f}%)')
c2_atyp = c2[c2['l2_cause'] == 'ATYPICAL']
print(f'    Case 2 ATYPICAL: {len(c2_atyp)}/{len(c2)} ({len(c2_atyp)/len(c2)*100:.1f}%)')
print(f'  Case 3 (Quadrant B):            {len(c3)} ({len(c3)/n_cbecs*100:.1f}%)')
c3_atyp = c3[c3['l2_cause'] == 'ATYPICAL']
print(f'    Case 3 ATYPICAL: {len(c3_atyp)}/{len(c3)} ({len(c3_atyp)/len(c3)*100:.1f}%)')
print(f'  Case 4 (Quadrant C):            {len(c4)} ({len(c4)/n_cbecs*100:.1f}%)')
print(f'  Union (non-overlapping):         {union} ({union/n_cbecs*100:.1f}%)')
print()

# ═══════════════════════════════════════════════════════════
# 12. BEST PRACTICE
# ═══════════════════════════════════════════════════════════
print('=== Best Practice (UNDER-CONSUMING ATYPICAL) ===')
under_atyp = cbecs[(cbecs['l2_cause'] == 'ATYPICAL') & (cbecs['nmbe'] < -0.02)]
print(f'  UNDER-CONSUMING ATYPICAL: {len(under_atyp)}')
under_valid = under_atyp[under_atyp['mean_load_kwh'] >= 5]
print(f'  After mean_load >= 5 kWh filter: {len(under_valid)}')
if len(under_valid) > 0:
    print(f'  Buildings:')
    for _, r in under_valid.sort_values('nmbe').iterrows():
        print(f'    {r["building"]}: NMBE={r["nmbe"]*100:+.1f}%, CVRMSE={r["cvrmse"]*100:.1f}%')
print()

# ═══════════════════════════════════════════════════════════
# 13. TABLE 7: QUADRANT BY BUILDING TYPE
# ═══════════════════════════════════════════════════════════
print('=== Table 7: Quadrant by Building Type (n >= 15) ===')
print(f'  {"Type":<20} {"n":>4}  {"A%":>5} {"B%":>5} {"C%":>5} {"D%":>5}')
for bt in ['Lodging', 'Public Services', 'Office', 'Education', 'Public Assembly']:
    sub = cbecs[cbecs['building_type'] == bt]
    if len(sub) >= 15:
        pcts = sub['quadrant_c14'].value_counts(normalize=True).reindex(['A','B','C','D'], fill_value=0) * 100
        print(f'  {bt:<20} {len(sub):>4}  {pcts["A"]:>5.1f} {pcts["B"]:>5.1f} {pcts["C"]:>5.1f} {pcts["D"]:>5.1f}')
print()

print('=' * 60)
print('This output is the source of truth for all paper values.')
print('No hardcoded expected values -- everything computed from CSV.')
print('=' * 60)
