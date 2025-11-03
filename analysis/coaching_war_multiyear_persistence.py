"""
Multi-Year WAR Persistence Analysis
Tests whether 2-year, 3-year, and 4-year rolling WAR averages are more stable
and predictive than single-year WAR.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_coaching_data():
    """Load coaching WAR data."""
    print("Loading coaching WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    df = df.sort_values(['Primary_Coach', 'Team', 'Year']).reset_index(drop=True)
    print(f"Loaded {len(df)} coach-season records")
    return df

def calculate_rolling_averages(df, windows=[2, 3, 4]):
    """Calculate rolling average WAR for each coach-team combination."""
    print("\nCalculating rolling average WAR...")

    # Group by coach and team for rolling calculations
    df_copy = df.copy()

    for window in windows:
        col_name = f'WAR_Rolling_{window}yr'
        df_copy[col_name] = np.nan

        # Calculate rolling average for each coach-team combination
        for (coach, team), group in df_copy.groupby(['Primary_Coach', 'Team']):
            if len(group) >= window:
                # Sort by year
                group = group.sort_values('Year')
                indices = group.index

                # Calculate rolling average
                rolling_avg = group['Coaching_WAR'].rolling(window=window, min_periods=window).mean()
                df_copy.loc[indices, col_name] = rolling_avg

        print(f"  {window}-year rolling average: {df_copy[col_name].notna().sum()} valid observations")

    return df_copy

def create_lagged_datasets(df):
    """Create multiple lagged datasets for different time horizons."""
    print("\nCreating lagged datasets...")

    datasets = {}

    # Single year
    print("\n1-YEAR LAG (baseline):")
    single_year = create_single_year_lag(df)
    datasets['1-year'] = single_year
    print(f"  {len(single_year)} consecutive season pairs")

    # 2-year average
    print("\n2-YEAR AVERAGE:")
    two_year = create_multiyear_lag(df, 'WAR_Rolling_2yr', 2)
    datasets['2-year avg'] = two_year
    print(f"  {len(two_year)} observations with 2-year average")

    # 3-year average
    print("\n3-YEAR AVERAGE:")
    three_year = create_multiyear_lag(df, 'WAR_Rolling_3yr', 3)
    datasets['3-year avg'] = three_year
    print(f"  {len(three_year)} observations with 3-year average")

    # 4-year average
    print("\n4-YEAR AVERAGE:")
    four_year = create_multiyear_lag(df, 'WAR_Rolling_4yr', 4)
    datasets['4-year avg'] = four_year
    print(f"  {len(four_year)} observations with 4-year average")

    return datasets

def create_single_year_lag(df):
    """Create Year N -> Year N+1 dataset."""
    lagged_records = []

    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    for idx in range(len(df) - 1):
        current = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        if (current['Team'] == next_row['Team'] and
            current['Primary_Coach'] == next_row['Primary_Coach'] and
            next_row['Year'] == current['Year'] + 1):

            lagged_records.append({
                'Coach': current['Primary_Coach'],
                'Team': current['Team'],
                'Year_N': int(current['Year']),
                'Predictor': current['Coaching_WAR'],
                'Outcome': next_row['Coaching_WAR'],
                'Background': current['Background']
            })

    return pd.DataFrame(lagged_records)

def create_multiyear_lag(df, rolling_col, window):
    """Create N-year average -> Year N+1 dataset."""
    lagged_records = []

    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    for idx in range(len(df) - 1):
        current = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        # Check consecutive years, same coach/team, and valid rolling average
        if (current['Team'] == next_row['Team'] and
            current['Primary_Coach'] == next_row['Primary_Coach'] and
            next_row['Year'] == current['Year'] + 1 and
            pd.notna(current[rolling_col])):

            lagged_records.append({
                'Coach': current['Primary_Coach'],
                'Team': current['Team'],
                'Year_N': int(current['Year']),
                'Predictor': current[rolling_col],
                'Outcome': next_row['Coaching_WAR'],
                'Background': current['Background']
            })

    return pd.DataFrame(lagged_records)

def analyze_persistence(datasets):
    """Analyze persistence for each time horizon."""
    print("\n" + "="*80)
    print("PERSISTENCE COMPARISON: Single Year vs Multi-Year Averages")
    print("="*80)

    results = []

    for name, df in datasets.items():
        if len(df) > 0:
            # Correlation
            corr, p_value = stats.pearsonr(df['Predictor'], df['Outcome'])
            spearman_corr, spearman_p = stats.spearmanr(df['Predictor'], df['Outcome'])

            # Regression
            X = df[['Predictor']].values
            y = df['Outcome'].values
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            results.append({
                'Time_Horizon': name,
                'N': len(df),
                'Pearson_r': corr,
                'Pearson_p': p_value,
                'Spearman_rho': spearman_corr,
                'R2': r2,
                'RMSE': rmse,
                'Coefficient': model.coef_[0],
                'Intercept': model.intercept_
            })

            print(f"\n{name.upper()}:")
            print(f"  Sample size: {len(df)}")
            print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")
            print(f"  Spearman rho: {spearman_corr:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Regression coefficient: {model.coef_[0]:.4f} (persistence rate)")

    results_df = pd.DataFrame(results)

    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(results_df[['Time_Horizon', 'N', 'Pearson_r', 'R2', 'Coefficient', 'RMSE']].to_string(index=False))

    # Calculate improvements
    baseline_r2 = results_df[results_df['Time_Horizon'] == '1-year']['R2'].values[0]
    baseline_rmse = results_df[results_df['Time_Horizon'] == '1-year']['RMSE'].values[0]

    print("\n" + "="*80)
    print("IMPROVEMENT OVER SINGLE-YEAR BASELINE")
    print("="*80)

    for _, row in results_df.iterrows():
        if row['Time_Horizon'] != '1-year':
            r2_improvement = ((row['R2'] - baseline_r2) / baseline_r2) * 100
            rmse_improvement = ((baseline_rmse - row['RMSE']) / baseline_rmse) * 100
            print(f"\n{row['Time_Horizon']}:")
            print(f"  R² improvement: {r2_improvement:+.1f}%")
            print(f"  RMSE improvement: {rmse_improvement:+.1f}%")
            print(f"  Persistence coefficient: {row['Coefficient']:.3f} vs {results_df[results_df['Time_Horizon']=='1-year']['Coefficient'].values[0]:.3f}")

    return results_df

def create_comparison_plots(datasets, results_df):
    """Create visualizations comparing different time horizons."""
    print("\nCreating comparison visualizations...")

    # Plot 1: Scatter plots for each time horizon
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('WAR Persistence by Time Horizon', fontsize=15, fontweight='bold')

    time_horizons = ['1-year', '2-year avg', '3-year avg', '4-year avg']

    for idx, name in enumerate(time_horizons):
        ax = axes[idx // 2, idx % 2]
        df = datasets[name]
        result = results_df[results_df['Time_Horizon'] == name].iloc[0]

        # Scatter
        ax.scatter(df['Predictor'], df['Outcome'], alpha=0.4, s=30, color='steelblue')

        # Regression line
        X = df[['Predictor']].values
        y = df['Outcome'].values
        model = LinearRegression()
        model.fit(X, y)
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Regression')

        # Perfect persistence line
        lim_min = min(df['Predictor'].min(), df['Outcome'].min())
        lim_max = max(df['Predictor'].max(), df['Outcome'].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=1.5,
               label='Perfect persistence', alpha=0.7)

        # Zero lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

        ax.set_xlabel(f'{name} WAR', fontsize=11, fontweight='bold')
        ax.set_ylabel('Year N+1 WAR', fontsize=11, fontweight='bold')
        ax.set_title(f'{name}: r={result["Pearson_r"]:.3f}, R²={result["R2"]:.3f} (n={result["N"]})',
                    fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_war_multiyear_persistence_scatter.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_war_multiyear_persistence_scatter.png")
    plt.close()

    # Plot 2: Summary bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(results_df))

    # R² comparison
    ax1.bar(x_pos, results_df['R2'], color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Time Horizon', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² (Variance Explained)', fontsize=12, fontweight='bold')
    ax1.set_title('Predictive Power by Time Horizon', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Time_Horizon'])
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(results_df['R2']):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    # Persistence coefficient comparison
    ax2.bar(x_pos, results_df['Coefficient'], color='coral', alpha=0.8, edgecolor='black')
    ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Perfect persistence')
    ax2.set_xlabel('Time Horizon', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Regression Coefficient', fontsize=12, fontweight='bold')
    ax2.set_title('Persistence Rate by Time Horizon', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['Time_Horizon'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(results_df['Coefficient']):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_war_multiyear_persistence_summary.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_war_multiyear_persistence_summary.png")
    plt.close()

def fishers_z_test(r1, n1, r2, n2):
    """
    Fisher's Z-transformation test for comparing two independent correlations.

    Returns:
        z_stat: Test statistic
        p_value: Two-tailed p-value
    """
    # Fisher's Z transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Standard error
    se = np.sqrt(1/(n1-3) + 1/(n2-3))

    # Test statistic
    z_stat = (z1 - z2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value

def analyze_by_background(datasets):
    """Analyze whether persistence differs by coaching background."""
    print("\n" + "="*80)
    print("PERSISTENCE BY COACHING BACKGROUND")
    print("="*80)

    all_results = []

    for time_horizon, df in datasets.items():
        if len(df) > 0:
            print(f"\n{time_horizon.upper()}:")

            for background in ['Offensive', 'Defensive', 'Other']:
                subset = df[df['Background'] == background]

                if len(subset) > 20:  # Minimum sample size
                    corr, p_value = stats.pearsonr(subset['Predictor'], subset['Outcome'])

                    all_results.append({
                        'Time_Horizon': time_horizon,
                        'Background': background,
                        'N': len(subset),
                        'Correlation': corr,
                        'R2': corr**2,
                        'P_value': p_value
                    })

                    print(f"  {background}: r={corr:.3f}, R²={corr**2:.3f}, n={len(subset)}, p={p_value:.4f}")

    results_df = pd.DataFrame(all_results)

    # Test if offensive vs defensive differs
    print("\n" + "="*80)
    print("OFFENSIVE vs DEFENSIVE COMPARISON (with Statistical Tests)")
    print("="*80)

    for time_horizon in ['1-year', '2-year avg', '3-year avg', '4-year avg']:
        off_data = results_df[(results_df['Time_Horizon'] == time_horizon) &
                              (results_df['Background'] == 'Offensive')]
        def_data = results_df[(results_df['Time_Horizon'] == time_horizon) &
                              (results_df['Background'] == 'Defensive')]

        if len(off_data) > 0 and len(def_data) > 0:
            off_r2 = off_data['R2'].values[0]
            def_r2 = def_data['R2'].values[0]
            off_r = off_data['Correlation'].values[0]
            def_r = def_data['Correlation'].values[0]
            off_n = off_data['N'].values[0]
            def_n = def_data['N'].values[0]

            # Fisher's Z test
            z_stat, p_value = fishers_z_test(off_r, off_n, def_r, def_n)

            print(f"\n{time_horizon}:")
            print(f"  Offensive: R²={off_r2:.4f}, r={off_r:.4f}, n={off_n}")
            print(f"  Defensive: R²={def_r2:.4f}, r={def_r:.4f}, n={def_n}")
            print(f"  Difference: {off_r2 - def_r2:+.4f}")
            print(f"  Fisher's Z-test: Z={z_stat:.4f}, p={p_value:.4f}")

            if p_value < 0.05:
                print(f"  [*] SIGNIFICANT (p < 0.05)")
            elif p_value < 0.10:
                print(f"  [*] MARGINALLY SIGNIFICANT (p < 0.10)")
            else:
                print(f"  [*] NOT SIGNIFICANT (p >= 0.10)")

    return results_df

def main():
    """Run multi-year persistence analysis."""

    # Load data
    df = load_coaching_data()

    # Calculate rolling averages
    df_with_rolling = calculate_rolling_averages(df)

    # Create lagged datasets
    datasets = create_lagged_datasets(df_with_rolling)

    # Analyze persistence
    results_df = analyze_persistence(datasets)

    # Create visualizations
    create_comparison_plots(datasets, results_df)

    # Analyze by background
    background_results = analyze_by_background(datasets)

    # Save results
    results_df.to_csv('analysis/outputs/csv/coaching_war_multiyear_persistence_results.csv', index=False)
    background_results.to_csv('analysis/outputs/csv/coaching_war_multiyear_persistence_by_background.csv', index=False)

    # Save detailed data
    for name, data in datasets.items():
        filename = f"analysis/outputs/csv/coaching_war_{name.replace(' ', '_').replace('-', '_')}_data.csv"
        data.to_csv(filename, index=False)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")

    baseline_r2 = results_df[results_df['Time_Horizon'] == '1-year']['R2'].values[0]
    best_r2 = results_df['R2'].max()
    best_horizon = results_df[results_df['R2'] == best_r2]['Time_Horizon'].values[0]
    improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100

    print(f"\n  Baseline (1-year): R² = {baseline_r2:.4f}")
    print(f"  Best ({best_horizon}): R² = {best_r2:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")

    baseline_coef = results_df[results_df['Time_Horizon'] == '1-year']['Coefficient'].values[0]
    best_coef = results_df[results_df['R2'] == best_r2]['Coefficient'].values[0]

    print(f"\n  Persistence coefficient:")
    print(f"    1-year: {baseline_coef:.3f}")
    print(f"    {best_horizon}: {best_coef:.3f}")

    if best_r2 > baseline_r2 * 1.3:
        print(f"\n  [+] Multi-year averaging SUBSTANTIALLY improves stability")
        print(f"      This suggests aggregation helps separate skill from noise")
    elif best_r2 > baseline_r2 * 1.1:
        print(f"\n  [~] Multi-year averaging provides MODEST improvement")
        print(f"      Some benefit to aggregation but still high variability")
    else:
        print(f"\n  [-] Multi-year averaging provides MINIMAL improvement")
        print(f"      Suggests fundamental instability in coaching effectiveness")

    print("\nResults saved to analysis/outputs/csv/ and analysis/outputs/png/")

if __name__ == '__main__':
    main()
