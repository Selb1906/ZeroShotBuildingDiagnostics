"""
Generate paper figures from CSV data for MDPI Buildings submission.
Outputs: PNG (300 dpi) + PDF for each figure.

Usage:
    python scripts/generate_paper_figures.py

Source: results/cbecs2018_c14_median_evaluation.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
import os

# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH = 'results/cbecs2018_c14_median_evaluation.csv'
OUT_DIR = 'figures'
DPI = 300
EXCLUDED_TYPES = ['Other', 'Technology', 'Parking', 'Utility']

# MDPI recommended: single column = 85 mm, double column = 170 mm
# At 300 DPI: 85mm ≈ 3.35in, 170mm ≈ 6.69in
FIG_WIDTH_SINGLE = 3.35
FIG_WIDTH_DOUBLE = 6.69

# Color palette
COLORS = {
    'A': '#10B981',  # green
    'B': '#F59E0B',  # amber
    'C': '#3B82F6',  # blue
    'D': '#EF4444',  # red
    'NORMAL': '#10B981',
    'CV_DRIVEN': '#F59E0B',
    'ATYPICAL': '#EF4444',
}

plt.rcParams.update({
    'font.size': 8,
    'font.family': 'sans-serif',
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': DPI,
})


def load_data():
    df = pd.read_csv(CSV_PATH)
    cbecs = df[~df['building_type'].isin(EXCLUDED_TYPES)].copy()
    return df, cbecs


def save_fig(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    print(f'  Saved: {name}.png, {name}.pdf')


# ── Figure 5: EUI Score vs Pattern Score Scatter ────────────────────────
def fig5_scatter(cbecs):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 4.5))

    for q, label, color, marker in [
        ('A', 'A: Excellent', COLORS['A'], 'o'),
        ('B', 'B: Efficient but Irregular', COLORS['B'], 's'),
        ('C', 'C: Consistent but Inefficient', COLORS['C'], '^'),
        ('D', 'D: Needs Improvement', COLORS['D'], 'D'),
    ]:
        sub = cbecs[cbecs['quadrant_c14'] == q]
        ax.scatter(sub['eui_score_c14'], sub['pattern_score'],
                   c=color, alpha=0.5, s=12, marker=marker, label=f'{label} (n={len(sub)})',
                   edgecolors='none')

    # Threshold lines
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Correlation annotation
    r, p = stats.pearsonr(cbecs['eui_score_c14'], cbecs['pattern_score'])
    ax.text(0.02, 0.98, f'r = {r:.3f}, p < 0.001\nn = {len(cbecs)}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    ax.set_xlabel('EUI Score (CBECS 2018 C14 z-score)')
    ax.set_ylabel('Pattern Score (within-type CVRMSE z-score)')
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower right', framealpha=0.9, markerscale=1.5)
    ax.set_title('Figure 5. EUI Score vs. Pattern Score (n=583 CBECS-mapped buildings)')

    save_fig(fig, 'fig5_scatter')
    plt.close(fig)


# ── Figure 6: Quadrant Distribution ─────────────────────────────────────
def fig6_quadrant(cbecs):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 1.5, 3.5))

    counts = cbecs['quadrant_c14'].value_counts().sort_index()
    labels = ['A: Excellent', 'B: Efficient\nbut Irregular', 'C: Consistent\nbut Inefficient', 'D: Needs\nImprovement']
    colors = [COLORS['A'], COLORS['B'], COLORS['C'], COLORS['D']]

    bars = ax.bar(labels, [counts['A'], counts['B'], counts['C'], counts['D']], color=colors, edgecolor='white', width=0.7)

    # Add count + percentage labels
    total = len(cbecs)
    for bar, val in zip(bars, [counts['A'], counts['B'], counts['C'], counts['D']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{val}\n({val/total*100:.1f}%)', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Number of Buildings')
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_title(f'Figure 6. Quadrant Distribution (n={total})')

    save_fig(fig, 'fig6_quadrant')
    plt.close(fig)


# ── Figure 7: CVRMSE by Building Type ───────────────────────────────────
def fig7_cvrmse_type(df):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 4))

    # Use all 611 buildings, group by type, sort by mean CVRMSE
    # Only include types with n >= 5
    type_stats = df.groupby('building_type')['cvrmse'].agg(['mean', 'median', 'count', 'std'])
    type_stats = type_stats[type_stats['count'] >= 5].sort_values('mean')

    y_pos = range(len(type_stats))
    colors_list = ['#3B82F6'] * len(type_stats)

    bars = ax.barh(y_pos, type_stats['mean'] * 100, color=colors_list, edgecolor='white', height=0.6, alpha=0.8)

    # Add median markers
    ax.scatter(type_stats['median'] * 100, y_pos, color='red', s=20, zorder=5, marker='|', linewidths=1.5, label='Median')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{t} (n={int(type_stats.loc[t, 'count'])})" for t in type_stats.index])
    ax.set_xlabel('CVRMSE (%)')

    # Value labels
    for i, (idx, row) in enumerate(type_stats.iterrows()):
        ax.text(row['mean'] * 100 + 0.5, i, f'{row["mean"]*100:.1f}%', va='center', fontsize=6.5)

    ax.legend(loc='lower right', fontsize=7)
    ax.set_title('Figure 7. Mean CVRMSE by Building Type')

    save_fig(fig, 'fig7_cvrmse_type')
    plt.close(fig)


# ── Figure 8: NMBE Comparison ───────────────────────────────────────────
def fig8_nmbe(cbecs):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 1.5, 3.5))

    categories = ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']
    labels = ['NORMAL\n(Pattern >= 50)', 'CV_DRIVEN\n(Excess <= 5pp)', 'ATYPICAL\n(Excess > 5pp)']
    colors = [COLORS['NORMAL'], COLORS['CV_DRIVEN'], COLORS['ATYPICAL']]

    means = []
    sems = []
    ns = []
    for cat in categories:
        sub = cbecs[cbecs['l2_cause'] == cat]
        means.append(sub['nmbe'].abs().mean() * 100)
        sems.append(sub['nmbe'].abs().std() * 100 / np.sqrt(len(sub)))  # SEM
        ns.append(len(sub))

    bars = ax.bar(labels, means, color=colors, edgecolor='white', width=0.6, alpha=0.85)
    ax.errorbar(range(len(categories)), means, yerr=sems, fmt='none', color='black', capsize=4, linewidth=0.8)

    for i, (m, n, s) in enumerate(zip(means, ns, sems)):
        ax.text(i, m + s + 0.15, f'{m:.2f}%\n(n={n})', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('Mean |NMBE| (%)')
    ax.set_ylim(0, max(means) * 1.5)
    ax.set_title('Figure 8. Mean |NMBE| by Level 2 Classification')

    save_fig(fig, 'fig8_nmbe')
    plt.close(fig)


# ── Figure 9: Level 2 Breakdown by Quadrant ─────────────────────────────
def fig9_level2_quadrant(cbecs):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 4))

    ct = pd.crosstab(cbecs['quadrant_c14'], cbecs['l2_cause'])
    # Ensure column order
    for col in ['NORMAL', 'CV_DRIVEN', 'ATYPICAL']:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[['NORMAL', 'CV_DRIVEN', 'ATYPICAL']]

    quadrants = ['A', 'B', 'C', 'D']
    labels = ['A: Excellent', 'B: Eff. but Irreg.', 'C: Cons. but Ineff.', 'D: Needs Impr.']
    x = np.arange(len(quadrants))
    width = 0.55

    normal_vals = [ct.loc[q, 'NORMAL'] for q in quadrants]
    cv_vals = [ct.loc[q, 'CV_DRIVEN'] for q in quadrants]
    atyp_vals = [ct.loc[q, 'ATYPICAL'] for q in quadrants]

    p1 = ax.bar(x, normal_vals, width, label=f'NORMAL (n={sum(normal_vals)})', color=COLORS['NORMAL'], alpha=0.85)
    p2 = ax.bar(x, cv_vals, width, bottom=normal_vals, label=f'CV_DRIVEN (n={sum(cv_vals)})', color=COLORS['CV_DRIVEN'], alpha=0.85)
    p3 = ax.bar(x, atyp_vals, width, bottom=[n+c for n,c in zip(normal_vals, cv_vals)],
                label=f'ATYPICAL (n={sum(atyp_vals)})', color=COLORS['ATYPICAL'], alpha=0.85)

    # Labels on bars
    for i, q in enumerate(quadrants):
        total = normal_vals[i] + cv_vals[i] + atyp_vals[i]
        ax.text(i, total + 2, str(total), ha='center', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of Buildings')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('Figure 9. Level 2 Classification by Quadrant (n=583)')

    save_fig(fig, 'fig9_level2_quadrant')
    plt.close(fig)


# ── Figure 10 (bonus): Box plot of CVRMSE by quadrant ───────────────────
def fig_bonus_cvrmse_box(cbecs):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 1.5, 3.5))

    quadrants = ['A', 'B', 'C', 'D']
    data = [cbecs[cbecs['quadrant_c14'] == q]['cvrmse'] * 100 for q in quadrants]

    bp = ax.boxplot(data, labels=['A', 'B', 'C', 'D'], patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))

    for patch, color in zip(bp['boxes'], [COLORS[q] for q in quadrants]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel('Quadrant')
    ax.set_ylabel('CVRMSE (%)')
    ax.set_title('CVRMSE Distribution by Quadrant')

    save_fig(fig, 'fig_bonus_cvrmse_box')
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, cbecs = load_data()

    print(f'Loaded: {len(df)} total, {len(cbecs)} CBECS-mapped')
    print(f'Output dir: {OUT_DIR}/')
    print()

    # Print verification summary
    q = cbecs['quadrant_c14'].value_counts().sort_index()
    l2 = cbecs['l2_cause'].value_counts()
    print(f'Quadrant: A={q["A"]}, B={q["B"]}, C={q["C"]}, D={q["D"]}')
    print(f'Level 2: NORMAL={l2["NORMAL"]}, CV_DRIVEN={l2["CV_DRIVEN"]}, ATYPICAL={l2["ATYPICAL"]}')
    r, p = stats.pearsonr(cbecs['eui_score_c14'], cbecs['pattern_score'])
    print(f'EUI Score vs Pattern Score: r={r:.3f}')
    print()

    print('Generating figures...')
    fig5_scatter(cbecs)
    fig6_quadrant(cbecs)
    fig7_cvrmse_type(df)
    fig8_nmbe(cbecs)
    fig9_level2_quadrant(cbecs)
    fig_bonus_cvrmse_box(cbecs)

    print()
    print('Done! Figures saved to figures/')


if __name__ == '__main__':
    main()
