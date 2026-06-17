"""
Compare leakage-test results to the primary analysis.

For each test (imputation, temporal), reports:
  - Train/Test R² and Test RMSE (from the result CSVs)
  - Top 10 / Bottom 10 coach overlap with primary
  - Spearman rank correlation of career WAR (intersection of coaches)
  - 1-year WAR persistence Pearson r (computed from the impact CSV)

Outputs a single summary table to stdout and to
data/final/leakage_tests/leakage_robustness_comparison.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_IMPACT = Path('data/final/coaching_impact_analysis.csv')
PRIMARY_STATS = Path('data/final/coach_career_impact_stats.csv')

IMP_IMPACT = Path('data/final/leakage_tests/imputation/coaching_impact_analysis_imputation_test.csv')
IMP_STATS = Path('data/final/leakage_tests/imputation/coach_career_impact_stats_imputation_test.csv')

TEMP_IMPACT = Path('data/final/leakage_tests/temporal/coaching_impact_analysis_temporal_test.csv')
TEMP_STATS = Path('data/final/leakage_tests/temporal/coach_career_impact_stats_temporal_test.csv')

OUT_CSV = Path('data/final/leakage_tests/leakage_robustness_comparison.csv')
OUT_TOP_BOTTOM = Path('data/final/leakage_tests/top_bottom_comparison.csv')


def war_persistence_pearson(impact_df):
    """Pearson r between team's Year N WAR and Year N+1 WAR (overall)."""
    df = impact_df[['Team', 'Year', 'Coaching_WAR']].dropna().copy()
    df = df.sort_values(['Team', 'Year']).reset_index(drop=True)
    df['Coaching_WAR_Next'] = df.groupby('Team')['Coaching_WAR'].shift(-1)
    df['Year_Next'] = df.groupby('Team')['Year'].shift(-1)
    # only consecutive seasons
    df = df[df['Year_Next'] == df['Year'] + 1].dropna()
    r, p = stats.pearsonr(df['Coaching_WAR'], df['Coaching_WAR_Next'])
    return r, p, len(df)


def load_career_stats(path):
    """Load and standardize career-stats CSV (coach name as index)."""
    df = pd.read_csv(path)
    # First column is Primary_Coach (index when saved)
    if 'Primary_Coach' in df.columns:
        df = df.set_index('Primary_Coach')
    elif df.columns[0] == 'Unnamed: 0':
        df = df.rename(columns={'Unnamed: 0': 'Primary_Coach'}).set_index('Primary_Coach')
    return df


def top_bottom_overlap(primary_stats, test_stats, n=10, metric='Avg_WAR'):
    """Compare top-N and bottom-N coaches between two career-stats tables."""
    # Filter both to coaches present in both (intersection)
    common = primary_stats.index.intersection(test_stats.index)
    p = primary_stats.loc[common].sort_values(metric, ascending=False)
    t = test_stats.loc[common].sort_values(metric, ascending=False)

    top_p = set(p.head(n).index)
    top_t = set(t.head(n).index)
    bot_p = set(p.tail(n).index)
    bot_t = set(t.tail(n).index)

    # Spearman rank correlation across the full common set
    p_rank = p[metric].rank()
    t_rank = t[metric].rank()
    rho, rho_p = stats.spearmanr(
        p_rank.loc[common], t_rank.loc[common], nan_policy='omit'
    )

    return {
        'common_coaches': len(common),
        f'top{n}_overlap': len(top_p & top_t),
        f'top{n}_primary': sorted(top_p),
        f'top{n}_test': sorted(top_t),
        f'bottom{n}_overlap': len(bot_p & bot_t),
        f'bottom{n}_primary': sorted(bot_p),
        f'bottom{n}_test': sorted(bot_t),
        'spearman_rho': rho,
        'spearman_p': rho_p,
    }


def metrics_from_impact(impact_df, label):
    """Pull global model metrics from the impact CSV (which has actual & predicted)."""
    actual = impact_df['Actual_Win_Pct']
    pred_coach = impact_df['Predicted_With_Coach']
    overall_r2 = 1 - ((actual - pred_coach) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()
    rmse = float(np.sqrt(((actual - pred_coach) ** 2).mean()))
    se_wins = impact_df['WAR_SE'].iloc[0] if 'WAR_SE' in impact_df.columns else np.nan
    war_mean = impact_df['Coaching_WAR'].mean()
    war_std = impact_df['Coaching_WAR'].std()
    war_max = impact_df['Coaching_WAR'].max()
    war_min = impact_df['Coaching_WAR'].min()
    return {
        'label': label,
        'overall_r2_full': overall_r2,
        'overall_rmse_full': rmse,
        'reported_se_wins': se_wins,
        'mean_war': war_mean,
        'std_war': war_std,
        'max_war': war_max,
        'min_war': war_min,
    }


def main():
    print('=' * 80)
    print('LEAKAGE ROBUSTNESS COMPARISON')
    print('=' * 80)

    primary_impact = pd.read_csv(PRIMARY_IMPACT)
    imp_impact = pd.read_csv(IMP_IMPACT)
    temp_impact = pd.read_csv(TEMP_IMPACT)

    primary_stats = load_career_stats(PRIMARY_STATS)
    imp_stats = load_career_stats(IMP_STATS)
    temp_stats = load_career_stats(TEMP_STATS)

    # --- 1. Global metrics from impact CSVs ---
    print('\n--- Global metrics (full-data) ---')
    rows = [
        metrics_from_impact(primary_impact, 'Primary'),
        metrics_from_impact(imp_impact, 'Imputation test'),
        metrics_from_impact(temp_impact, 'Temporal test'),
    ]
    metrics_df = pd.DataFrame(rows)
    with pd.option_context('display.float_format', '{:.4f}'.format,
                           'display.width', 200):
        print(metrics_df.to_string(index=False))

    # --- 2. WAR persistence (1-year, overall) ---
    print('\n--- 1-year WAR persistence (overall, all team-years) ---')
    persistence_rows = []
    for label, df in [('Primary', primary_impact),
                      ('Imputation test', imp_impact),
                      ('Temporal test', temp_impact)]:
        r, p, n = war_persistence_pearson(df)
        persistence_rows.append({'label': label, 'pearson_r': r,
                                  'r_squared': r ** 2, 'p_value': p,
                                  'n_pairs': n})
    persist_df = pd.DataFrame(persistence_rows)
    with pd.option_context('display.float_format', '{:.4f}'.format):
        print(persist_df.to_string(index=False))

    # --- 3. Top/Bottom 10 overlap and rank correlation ---
    print('\n--- Career WAR rankings (coaches with >=3 seasons) ---')
    print('\nIMPUTATION TEST vs PRIMARY:')
    imp_overlap = top_bottom_overlap(primary_stats, imp_stats, n=10)
    print(f"  Common coaches: {imp_overlap['common_coaches']}")
    print(f"  Top 10 overlap: {imp_overlap['top10_overlap']} / 10")
    print(f"  Bottom 10 overlap: {imp_overlap['bottom10_overlap']} / 10")
    print(f"  Spearman rank correlation (career WAR): "
          f"rho={imp_overlap['spearman_rho']:.4f}, p={imp_overlap['spearman_p']:.2e}")

    print('\nTEMPORAL TEST vs PRIMARY:')
    temp_overlap = top_bottom_overlap(primary_stats, temp_stats, n=10)
    print(f"  Common coaches: {temp_overlap['common_coaches']}")
    print(f"  Top 10 overlap: {temp_overlap['top10_overlap']} / 10")
    print(f"  Bottom 10 overlap: {temp_overlap['bottom10_overlap']} / 10")
    print(f"  Spearman rank correlation (career WAR): "
          f"rho={temp_overlap['spearman_rho']:.4f}, p={temp_overlap['spearman_p']:.2e}")

    # --- 4. Side-by-side top-10 / bottom-10 tables ---
    print('\n--- Top 10 coaches (Avg_WAR per 16-game season) ---')
    top_table = build_top_bottom_table(primary_stats, imp_stats, temp_stats,
                                        which='top', n=10)
    print(top_table.to_string(index=False))

    print('\n--- Bottom 10 coaches (Avg_WAR per 16-game season) ---')
    bot_table = build_top_bottom_table(primary_stats, imp_stats, temp_stats,
                                        which='bottom', n=10)
    print(bot_table.to_string(index=False))

    # --- Save summary CSV ---
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for label, m in zip(['Primary', 'Imputation test', 'Temporal test'],
                        rows):
        row = {'analysis': label, **m}
        row.pop('label')
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.merge(persist_df.rename(columns={'label': 'analysis'}),
                                    on='analysis')
    summary_df.to_csv(OUT_CSV, index=False)
    print(f'\nSummary saved to: {OUT_CSV}')

    # Save top/bottom side-by-side
    combined = pd.concat([
        top_table.assign(group='top'),
        bot_table.assign(group='bottom'),
    ], ignore_index=True)
    combined.to_csv(OUT_TOP_BOTTOM, index=False)
    print(f'Top/Bottom comparison saved to: {OUT_TOP_BOTTOM}')


def build_top_bottom_table(primary, imp, temp, which='top', n=10):
    """Build a side-by-side table of top or bottom N coaches from each analysis."""
    sort_asc = (which == 'bottom')
    rows = []
    for label, df in [('Primary', primary), ('Imp test', imp), ('Temp test', temp)]:
        d = df.sort_values('Avg_WAR', ascending=sort_asc).head(n).copy()
        d = d.reset_index()
        for rank, r in d.iterrows():
            rows.append({
                'analysis': label,
                'rank': rank + 1,
                'coach': r['Primary_Coach'],
                'avg_war_per_16': round(r['Avg_WAR'] * 16, 2),
                'seasons': int(r['Seasons']),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    main()
