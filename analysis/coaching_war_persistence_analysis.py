"""
Coaching WAR Persistence Analysis
Tests whether coaching WAR is stable across consecutive seasons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from figure_utils import configure_matplotlib_fonts
import warnings
warnings.filterwarnings('ignore')

def load_coaching_data():
    """Load coaching WAR data."""
    print("Loading coaching WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_lagged_dataset(df):
    """Create dataset with Year N metrics matched to Year N+1 outcomes."""
    print("\nCreating consecutive season pairs...")

    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    lagged_records = []

    for idx in range(len(df) - 1):
        current = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        # Consecutive years with same coach and team
        if (current['Team'] == next_row['Team'] and
            current['Primary_Coach'] == next_row['Primary_Coach'] and
            next_row['Year'] == current['Year'] + 1):

            lagged_records.append({
                'Team': current['Team'],
                'Coach': current['Primary_Coach'],
                'Year_N': int(current['Year']),
                'Year_N_Plus_1': int(next_row['Year']),
                'Year_N_WAR': current['Coaching_WAR'],
                'Year_N_Plus_1_WAR': next_row['Coaching_WAR'],
                'Year_N_Win_Pct': current['Actual_Win_Pct'],
                'Year_N_Plus_1_Win_Pct': next_row['Actual_Win_Pct'],
                'Background': current['Background'],
                'Same_Team': True
            })

    lagged_df = pd.DataFrame(lagged_records)
    print(f"Created {len(lagged_df)} consecutive season pairs (same coach, same team)")
    print(f"Years: {int(lagged_df['Year_N'].min())}-{int(lagged_df['Year_N'].max())}")
    print(f"Unique coaches: {lagged_df['Coach'].nunique()}")

    return lagged_df

def analyze_war_persistence(lagged_df):
    """Comprehensive WAR persistence analysis."""
    print("\n" + "="*80)
    print("WAR PERSISTENCE ANALYSIS: Year N WAR -> Year N+1 WAR")
    print("="*80)

    # Overall correlation
    corr, p_value = stats.pearsonr(lagged_df['Year_N_WAR'], lagged_df['Year_N_Plus_1_WAR'])
    spearman_corr, spearman_p = stats.spearmanr(lagged_df['Year_N_WAR'], lagged_df['Year_N_Plus_1_WAR'])

    # Regression
    X = lagged_df[['Year_N_WAR']].values
    y = lagged_df['Year_N_Plus_1_WAR'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\nOverall Persistence:")
    print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")
    print(f"  Spearman rho: {spearman_corr:.4f} (p={spearman_p:.6f})")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Regression: Year N+1 WAR = {model.coef_[0]:.4f} * Year N WAR + {model.intercept_:.4f}")

    # Interpret
    print(f"\nInterpretation:")
    if r2 > 0.20:
        print(f"  Strong persistence (R² = {r2:.3f})")
        print(f"  WAR appears to capture stable coaching skill")
    elif r2 > 0.10:
        print(f"  Moderate persistence (R² = {r2:.3f})")
        print(f"  Some stability but substantial year-to-year variation")
    else:
        print(f"  Weak persistence (R² = {r2:.3f})")
        print(f"  High year-to-year variability")

    # Calculate split-half reliability
    print(f"\nRegression to the mean:")
    print(f"  Coefficient: {model.coef_[0]:.4f}")
    if model.coef_[0] < 0.5:
        print(f"  Strong regression to mean - extreme Year N values don't persist")
    elif model.coef_[0] < 0.7:
        print(f"  Moderate regression to mean - some persistence of extremes")
    else:
        print(f"  Weak regression to mean - Year N performance largely persists")

    return corr, r2, model, p_value

def create_persistence_scatter(lagged_df, corr, r2, model, p_value):
    """Create detailed scatter plot of WAR persistence."""
    print("\nCreating WAR persistence scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Convert WAR to games (multiply by 16)
    year_n_games = lagged_df['Year_N_WAR'] * 16
    year_n_plus_1_games = lagged_df['Year_N_Plus_1_WAR'] * 16

    # Scatter plot
    ax.scatter(year_n_games, year_n_plus_1_games,
              alpha=0.4, s=50, color='steelblue', edgecolor='black', linewidth=0.5)

    # Regression line (convert to games)
    x_range = np.linspace(lagged_df['Year_N_WAR'].min(),
                         lagged_df['Year_N_WAR'].max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range * 16, y_pred * 16, 'r-', linewidth=3, label='Regression line', zorder=5)

    # Perfect persistence line (y=x)
    lim_min = min(year_n_games.min(), year_n_plus_1_games.min())
    lim_max = max(year_n_games.max(), year_n_plus_1_games.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2,
           label='Perfect persistence (y=x)', zorder=5)

    # Zero lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel('Year N Coaching WAR (Games per Season)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Year N+1 Coaching WAR (Games per Season)', fontsize=18, fontweight='bold')

    # Add stats box (equation in games scale to match axes)
    # Since model was fit on WAR scale (0-1), convert to games for display
    # Y_games = coef * X_games + intercept_games
    # where Y_games = model.predict(X_war) * 16 and X_games = X_war * 16
    coef_games = model.coef_[0]  # Coefficient stays the same
    intercept_games = model.intercept_ * 16  # Intercept scales by 16

    # Use mathematical notation with subscripts
    stats_text = f"$WAR_{{N+1}} = {coef_games:.3f} \\cdot WAR_N + {intercept_games:.3f}$ games\n"
    stats_text += f"$R^2 = {r2:.3f}$\n"
    stats_text += f"$n = {len(lagged_df)}$ season pairs\n"
    if p_value < 0.001:
        stats_text += f"$p < 0.001$"
    else:
        stats_text += f"$p = {p_value:.3f}$"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=14, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='lower right', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = 'analysis/outputs/png/coaching_war_persistence_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

def analyze_by_war_level(lagged_df):
    """Analyze persistence for different WAR levels."""
    print("\n" + "="*80)
    print("PERSISTENCE BY WAR LEVEL")
    print("="*80)

    # Create categories based on Year N WAR
    lagged_df['WAR_Category'] = pd.cut(lagged_df['Year_N_WAR'],
                                       bins=[-np.inf, -0.05, 0.05, np.inf],
                                       labels=['Below Replacement', 'Replacement Level', 'Above Replacement'])

    print("\nSample sizes:")
    print(lagged_df['WAR_Category'].value_counts().sort_index())

    results = []
    for category in ['Below Replacement', 'Replacement Level', 'Above Replacement']:
        subset = lagged_df[lagged_df['WAR_Category'] == category]
        if len(subset) > 10:
            corr, p_value = stats.pearsonr(subset['Year_N_WAR'], subset['Year_N_Plus_1_WAR'])
            mean_year_n = subset['Year_N_WAR'].mean()
            mean_year_n_plus_1 = subset['Year_N_Plus_1_WAR'].mean()

            results.append({
                'Category': category,
                'N': len(subset),
                'Mean_Year_N_WAR': mean_year_n,
                'Mean_Year_N_Plus_1_WAR': mean_year_n_plus_1,
                'Correlation': corr,
                'P_value': p_value,
                'R2': corr**2
            })

            print(f"\n{category} (n={len(subset)}):")
            print(f"  Year N mean WAR: {mean_year_n:.4f}")
            print(f"  Year N+1 mean WAR: {mean_year_n_plus_1:.4f}")
            print(f"  Change: {mean_year_n_plus_1 - mean_year_n:.4f} (regression to mean)")
            print(f"  Correlation: r={corr:.4f} (p={p_value:.6f}), R²={corr**2:.4f}")

    results_df = pd.DataFrame(results)
    return results_df

def analyze_by_quintile(lagged_df):
    """Analyze how each quintile performs in Year N+1."""
    print("\n" + "="*80)
    print("QUINTILE ANALYSIS: REGRESSION TO THE MEAN")
    print("="*80)

    # Create quintiles based on Year N WAR
    lagged_df['WAR_Quintile'] = pd.qcut(lagged_df['Year_N_WAR'], q=5,
                                        labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'],
                                        duplicates='drop')

    quintile_stats = lagged_df.groupby('WAR_Quintile').agg({
        'Year_N_WAR': ['mean', 'std', 'count'],
        'Year_N_Plus_1_WAR': ['mean', 'std']
    }).round(4)

    quintile_stats.columns = ['Year_N_Mean', 'Year_N_Std', 'Count', 'Year_N_Plus_1_Mean', 'Year_N_Plus_1_Std']
    quintile_stats['Change'] = quintile_stats['Year_N_Plus_1_Mean'] - quintile_stats['Year_N_Mean']
    quintile_stats['Pct_Retained'] = (quintile_stats['Year_N_Plus_1_Mean'] / quintile_stats['Year_N_Mean'] * 100).round(1)

    print("\nQuintile performance:")
    print(quintile_stats.to_string())

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x_pos = np.arange(len(quintile_stats))
    width = 0.35

    # Plot 1: Year N vs Year N+1 means
    ax1.bar(x_pos - width/2, quintile_stats['Year_N_Mean'], width,
           label='Year N', color='steelblue', alpha=0.8, edgecolor='black')
    ax1.bar(x_pos + width/2, quintile_stats['Year_N_Plus_1_Mean'], width,
           label='Year N+1', color='coral', alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Coaching WAR', fontsize=12, fontweight='bold')
    ax1.set_title('Regression to the Mean by Quintile', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(quintile_stats.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Change from Year N to Year N+1
    colors = ['red' if x < 0 else 'green' for x in quintile_stats['Change']]
    ax2.bar(x_pos, quintile_stats['Change'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change in WAR (Year N+1 - Year N)', fontsize=12, fontweight='bold')
    ax2.set_title('Regression to the Mean: Change by Quintile', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(quintile_stats.index, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(quintile_stats['Change']):
        ax2.text(i, v + (0.005 if v > 0 else -0.005), f'{v:.3f}',
                ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    plt.tight_layout()
    output_path = 'analysis/outputs/png/coaching_war_regression_to_mean.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved quintile plot to: {output_path}")
    plt.close()

    return quintile_stats

def compare_to_win_pct_persistence(lagged_df):
    """Compare WAR persistence to Win% persistence as a baseline."""
    print("\n" + "="*80)
    print("COMPARISON: WAR vs Win% Persistence")
    print("="*80)

    # WAR persistence
    war_corr, war_p = stats.pearsonr(lagged_df['Year_N_WAR'], lagged_df['Year_N_Plus_1_WAR'])
    war_r2 = war_corr**2

    # Win% persistence
    winpct_corr, winpct_p = stats.pearsonr(lagged_df['Year_N_Win_Pct'], lagged_df['Year_N_Plus_1_Win_Pct'])
    winpct_r2 = winpct_corr**2

    print(f"\nWAR Persistence:")
    print(f"  r = {war_corr:.4f}, R² = {war_r2:.4f}, p = {war_p:.6f}")

    print(f"\nWin% Persistence:")
    print(f"  r = {winpct_corr:.4f}, R² = {winpct_r2:.4f}, p = {winpct_p:.6f}")

    print(f"\nComparison:")
    print(f"  WAR R² / Win% R² = {war_r2/winpct_r2:.2f}x")
    if war_r2 > winpct_r2:
        print(f"  WAR is MORE stable than Win% (better signal-to-noise)")
    else:
        print(f"  WAR is LESS stable than Win% (lower persistence)")
        print(f"  Note: This is expected if WAR successfully removes contextual luck")

def create_summary_table(lagged_df, overall_corr, overall_r2, quintile_stats):
    """Create summary statistics table."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    summary = {
        'Metric': [
            'Sample size (consecutive seasons)',
            'Time period',
            'Unique coaches',
            'Overall correlation (r)',
            'R² (variance explained)',
            'Mean Year N WAR',
            'Mean Year N+1 WAR',
            'SD Year N WAR',
            'SD Year N+1 WAR',
            'Mean absolute change',
            'Median absolute change'
        ],
        'Value': [
            len(lagged_df),
            f"{int(lagged_df['Year_N'].min())}-{int(lagged_df['Year_N'].max())}",
            lagged_df['Coach'].nunique(),
            f"{overall_corr:.4f}",
            f"{overall_r2:.4f}",
            f"{lagged_df['Year_N_WAR'].mean():.4f}",
            f"{lagged_df['Year_N_Plus_1_WAR'].mean():.4f}",
            f"{lagged_df['Year_N_WAR'].std():.4f}",
            f"{lagged_df['Year_N_Plus_1_WAR'].std():.4f}",
            f"{(lagged_df['Year_N_Plus_1_WAR'] - lagged_df['Year_N_WAR']).abs().mean():.4f}",
            f"{(lagged_df['Year_N_Plus_1_WAR'] - lagged_df['Year_N_WAR']).abs().median():.4f}"
        ]
    }

    summary_df = pd.DataFrame(summary)
    print("\n" + summary_df.to_string(index=False))

    return summary_df

def main(font_family='serif'):
    """Run WAR persistence analysis."""

    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)

    # Load data
    df = load_coaching_data()

    # Create lagged dataset
    lagged_df = create_lagged_dataset(df)

    # Overall persistence analysis
    corr, r2, model, p_value = analyze_war_persistence(lagged_df)

    # Create main scatter plot
    create_persistence_scatter(lagged_df, corr, r2, model, p_value)

    # Analyze by WAR level
    level_results = analyze_by_war_level(lagged_df)

    # Quintile analysis (regression to mean)
    quintile_stats = analyze_by_quintile(lagged_df)

    # Compare to Win% persistence
    compare_to_win_pct_persistence(lagged_df)

    # Summary table
    summary_df = create_summary_table(lagged_df, corr, r2, quintile_stats)

    # Save results
    lagged_df.to_csv('analysis/outputs/csv/coaching_war_persistence_data.csv', index=False)
    level_results.to_csv('analysis/outputs/csv/coaching_war_persistence_by_level.csv', index=False)
    quintile_stats.to_csv('analysis/outputs/csv/coaching_war_persistence_quintiles.csv')
    summary_df.to_csv('analysis/outputs/csv/coaching_war_persistence_summary.csv', index=False)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - analysis/outputs/csv/coaching_war_persistence_data.csv")
    print(f"  - analysis/outputs/csv/coaching_war_persistence_by_level.csv")
    print(f"  - analysis/outputs/csv/coaching_war_persistence_quintiles.csv")
    print(f"  - analysis/outputs/csv/coaching_war_persistence_summary.csv")
    print(f"  - analysis/outputs/png/coaching_war_persistence_scatter.png")
    print(f"  - analysis/outputs/png/coaching_war_regression_to_mean.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coaching WAR persistence analysis')
    parser.add_argument('--font', type=str, default='serif',
                       help='Font family to use (default: serif)')
    args = parser.parse_args()

    main(font_family=args.font)
