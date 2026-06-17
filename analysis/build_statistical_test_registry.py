"""
Build a single registry of all inferential statistical tests reported in the
paper, then apply Benjamini-Hochberg FDR adjustment across the full family.

Family (Option A: single global family, k = 13):
  Tests 1-6:   Decade Offensive vs Defensive Mann-Whitney U tests (from CSV)
  Tests 7-8:   Welch's t-test 1970s vs 2020s within each background (recomputed)
  Test 9:      Overall Year N -> Year N+1 WAR Pearson correlation (recomputed)
  Tests 10-13: Fisher's Z persistence-horizon tests, O vs D (recomputed from
               the per-background multiyear persistence CSV; the *_statistical_
               comparison.csv variant in this directory is stale and does not
               reproduce the p-values reported in the paper)

Output: analysis/outputs/csv/statistical_test_registry.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


CSV_DIR = Path('analysis/outputs/csv')
DECADE_CSV = CSV_DIR / 'coach_background_decade_trend_analysis.csv'
PERSIST_BY_BG_CSV = CSV_DIR / 'coaching_war_multiyear_persistence_by_background.csv'
MATCHED_BG_CSV = CSV_DIR / 'coach_matched_war_background_data.csv'
PERSIST_DATA_CSV = CSV_DIR / 'coaching_war_persistence_data.csv'
OUT_CSV = CSV_DIR / 'statistical_test_registry.csv'

DECADES = [1970, 1980, 1990, 2000, 2010, 2020]


def classify_decade_start(year):
    return (int(year) // 10) * 10


def bh_qvalues(pvals):
    """Benjamini-Hochberg adjusted q-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    raw_q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity scanning largest-to-smallest
    q_mono = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_mono = np.minimum(q_mono, 1.0)
    out = np.zeros(n)
    out[order] = q_mono
    return out


def decade_mannwhitney_rows():
    df = pd.read_csv(DECADE_CSV)
    rows = []
    for _, r in df.iterrows():
        decade_label = f"{int(r['Year'])}s"
        rows.append({
            'test_id': f"decade_{int(r['Year'])}_OvD_MW",
            'description': f"{decade_label} Offensive vs Defensive WAR difference (Mann-Whitney U, two-sided)",
            'paper_section': 'Section 4.2 / Table tab:background_decade_stats',
            'family': 'global',
            'n': f"{int(r['n_offensive'])}/{int(r['n_defensive'])}",
            'effect_size': float(r['Difference']),
            'effect_label': 'Mean(O) - Mean(D), wins/season',
            'ci_lower': float(r['Diff_CI_Lower']),
            'ci_upper': float(r['Diff_CI_Upper']),
            'p_raw': float(r['p_value_mw']),
            'source_csv': DECADE_CSV.name,
        })
    return rows


def background_endpoint_rows():
    """Tests 7 and 8: Welch's t-test comparing 1970s vs 2020s within each
    background group. Reproduces the paper's '+0.83 wins per season from the
    1970s to the 2020s, 95% CI [+0.28, +1.38], p=0.004' style numbers."""
    df = pd.read_csv(MATCHED_BG_CSV)
    df['Decade_Start'] = df['Year'].apply(classify_decade_start)
    df['WAR_Games'] = df['Coaching_WAR'] * 16  # match paper's units
    rows = []
    for bg in ('Offensive', 'Defensive'):
        seventies = df[(df['Background'] == bg) & (df['Decade_Start'] == 1970)]['WAR_Games'].values
        twenties = df[(df['Background'] == bg) & (df['Decade_Start'] == 2020)]['WAR_Games'].values
        # Welch's t-test (unequal variances)
        result = stats.ttest_ind(twenties, seventies, equal_var=False)
        diff = float(twenties.mean() - seventies.mean())
        # Welch CI on the difference
        se = float(np.sqrt(twenties.var(ddof=1) / len(twenties)
                           + seventies.var(ddof=1) / len(seventies)))
        # Welch-Satterthwaite degrees of freedom
        v1 = twenties.var(ddof=1) / len(twenties)
        v2 = seventies.var(ddof=1) / len(seventies)
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(twenties) - 1) + v2 ** 2 / (len(seventies) - 1))
        t_crit = stats.t.ppf(0.975, df=dof)
        ci_lower = diff - t_crit * se
        ci_upper = diff + t_crit * se
        rows.append({
            'test_id': f"endpoint_{bg.lower()}_2020s_vs_1970s",
            'description': (f"{bg} 2020s vs 1970s mean WAR difference "
                            f"(Welch's t-test on wins/season)"),
            'paper_section': 'Section 4.2',
            'family': 'global',
            'n': f"{len(twenties)}/{len(seventies)} seasons (2020s/1970s)",
            'effect_size': diff,
            'effect_label': 'Mean(2020s) - Mean(1970s), wins/season',
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_raw': float(result.pvalue),
            'source_csv': f"{MATCHED_BG_CSV.name} (recomputed)",
        })
    return rows


def overall_persistence_row():
    """Test 9: Pearson correlation between Year N and Year N+1 WAR."""
    df = pd.read_csv(PERSIST_DATA_CSV)
    x = df['Year_N_WAR'].values
    y = df['Year_N_Plus_1_WAR'].values
    r_stat, p_val = stats.pearsonr(x, y)
    n = len(df)
    # Fisher z CI for the correlation
    z = np.arctanh(r_stat)
    z_se = 1.0 / np.sqrt(n - 3)
    r_lower = float(np.tanh(z - 1.96 * z_se))
    r_upper = float(np.tanh(z + 1.96 * z_se))
    return [{
        'test_id': 'persistence_overall_pearson',
        'description': 'Year N vs Year N+1 WAR Pearson correlation different from zero (overall, not segmented)',
        'paper_section': 'Section 4.4 / Abstract',
        'family': 'global',
        'n': str(n),
        'effect_size': float(r_stat),
        'effect_label': 'Pearson r (R^2 = r^2)',
        'ci_lower': r_lower,
        'ci_upper': r_upper,
        'p_raw': float(p_val),
        'source_csv': f"{PERSIST_DATA_CSV.name} (recomputed)",
    }]


def fisher_z_horizon_rows():
    """Tests 10-13: Persistence-horizon Fisher Z tests on the difference
    between offensive and defensive correlations. Recomputed from the
    per-background CSV that matches the paper's published R^2 values."""
    df = pd.read_csv(PERSIST_BY_BG_CSV)
    rows = []
    for horizon in ('1-year', '2-year avg', '3-year avg', '4-year avg'):
        sub = df[df['Time_Horizon'] == horizon]
        off = sub[sub['Background'] == 'Offensive'].iloc[0]
        defn = sub[sub['Background'] == 'Defensive'].iloc[0]
        r1, n1 = float(off['Correlation']), int(off['N'])
        r2, n2 = float(defn['Correlation']), int(defn['N'])
        # Fisher's Z for difference between two independent correlations
        z1, z2 = np.arctanh(r1), np.arctanh(r2)
        se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
        z_diff = (z1 - z2) / se
        p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_diff)))
        horizon_id = horizon.replace(' ', '_').replace('-', '_')
        rows.append({
            'test_id': f"persistence_{horizon_id}_FisherZ",
            'description': (f"{horizon} persistence: Offensive vs Defensive correlation "
                            f"difference (Fisher's Z-test, two-sided)"),
            'paper_section': 'Section 4.4 / Appendix E',
            'family': 'global',
            'n': f"{n1}/{n2}",
            'effect_size': float(r2 ** 2 - r1 ** 2),
            'effect_label': 'R^2(D) - R^2(O)',
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_raw': float(p_val),
            'source_csv': f"{PERSIST_BY_BG_CSV.name} (Fisher's Z recomputed)",
        })
    return rows


def main():
    rows = []
    rows.extend(decade_mannwhitney_rows())
    rows.extend(background_endpoint_rows())
    rows.extend(overall_persistence_row())
    rows.extend(fisher_z_horizon_rows())

    df = pd.DataFrame(rows)

    # BH-FDR across the full family
    df['q_bh'] = bh_qvalues(df['p_raw'].values)
    df['rank'] = df['p_raw'].rank(method='min').astype(int)

    df = df.sort_values('rank').reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nWrote {OUT_CSV} with {len(df)} tests\n")
    cols_to_show = ['rank', 'test_id', 'effect_size', 'p_raw', 'q_bh']
    with pd.option_context('display.max_colwidth', 60, 'display.width', 200):
        print(df[cols_to_show].to_string(index=False))


if __name__ == '__main__':
    main()
