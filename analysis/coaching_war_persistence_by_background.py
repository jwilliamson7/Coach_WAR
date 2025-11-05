"""
Coaching WAR Persistence by Background
Compares year-to-year persistence for Offensive vs Defensive coaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from figure_utils import configure_matplotlib_fonts
import warnings
warnings.filterwarnings('ignore')

def load_coaching_data():
    """Load coaching WAR data with backgrounds."""
    print("Loading coaching WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    df = df.sort_values(['Primary_Coach', 'Team', 'Year']).reset_index(drop=True)
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_consecutive_pairs_with_background(df):
    """Create dataset of consecutive season pairs with background info."""
    print("\nCreating consecutive season pairs...")

    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    consecutive_pairs = []

    for idx in range(len(df) - 1):
        current = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        # Check if consecutive years with same coach and team
        if (current['Team'] == next_row['Team'] and
            current['Primary_Coach'] == next_row['Primary_Coach'] and
            next_row['Year'] == current['Year'] + 1):

            consecutive_pairs.append({
                'Coach': current['Primary_Coach'],
                'Team': current['Team'],
                'Background': current['Background'],
                'Year_N': int(current['Year']),
                'Year_N_WAR': current['Coaching_WAR'],
                'Year_N_Plus_1_WAR': next_row['Coaching_WAR']
            })

    result_df = pd.DataFrame(consecutive_pairs)
    print(f"Created {len(result_df)} consecutive season pairs")

    # Filter to Offensive and Defensive only
    result_df = result_df[result_df['Background'].isin(['Offensive', 'Defensive'])]
    print(f"Filtered to {len(result_df)} pairs (Offensive and Defensive only)")

    return result_df

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

def analyze_by_background(df):
    """Analyze persistence separately for each background."""
    print("\n" + "="*80)
    print("WAR PERSISTENCE BY COACHING BACKGROUND")
    print("="*80)

    results = {}

    for background in ['Offensive', 'Defensive']:
        subset = df[df['Background'] == background]

        print(f"\n{background.upper()} COACHES:")
        print(f"  Sample size: {len(subset)} consecutive season pairs")

        # Correlation
        corr, p_value = stats.pearsonr(subset['Year_N_WAR'], subset['Year_N_Plus_1_WAR'])

        # Regression
        X = subset[['Year_N_WAR']].values
        y = subset['Year_N_Plus_1_WAR'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f} WAR ({rmse * 16:.2f} games)")
        print(f"  Regression: Y = {model.coef_[0]:.3f}X + {model.intercept_:.3f}")
        print(f"  Persistence: {model.coef_[0]*100:.1f}% of Year N WAR persists to Year N+1")

        results[background] = {
            'data': subset,
            'corr': corr,
            'r2': r2,
            'rmse': rmse,
            'p_value': p_value,
            'model': model,
            'n': len(subset)
        }

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    # R² comparison
    r2_diff = results['Defensive']['r2'] - results['Offensive']['r2']
    print(f"\nDefensive R² - Offensive R²: {r2_diff:+.4f}")
    if r2_diff > 0:
        print(f"Defensive coaches show {abs(r2_diff/results['Offensive']['r2'])*100:.1f}% stronger persistence")
    else:
        print(f"Offensive coaches show {abs(r2_diff/results['Defensive']['r2'])*100:.1f}% stronger persistence")

    # Statistical test using Fisher's Z-transformation
    print("\nStatistical Test (Fisher's Z-transformation):")
    z_stat, p_value = fishers_z_test(
        results['Offensive']['corr'], results['Offensive']['n'],
        results['Defensive']['corr'], results['Defensive']['n']
    )
    print(f"  Z-statistic: {z_stat:.4f}")
    print(f"  P-value (two-tailed): {p_value:.4f}")

    if p_value < 0.05:
        print(f"  [*] SIGNIFICANT: Correlations are statistically different (p < 0.05)")
    elif p_value < 0.10:
        print(f"  [*] MARGINALLY SIGNIFICANT: Correlations are marginally different (p < 0.10)")
    else:
        print(f"  [*] NOT SIGNIFICANT: Correlations are not statistically different (p >= 0.10)")

    # Effect size (difference in correlations)
    corr_diff = results['Offensive']['corr'] - results['Defensive']['corr']
    print(f"\n  Effect size (r_offensive - r_defensive): {corr_diff:+.4f}")

    return results

def create_background_comparison_scatter(results):
    """Create scatter plot comparing Offensive vs Defensive persistence."""
    print("\nCreating background comparison scatter plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Color scheme
    colors = {
        'Offensive': '#d62728',  # Red
        'Defensive': '#1f77b4'   # Blue
    }

    # Plot each background
    for background in ['Offensive', 'Defensive']:
        data = results[background]['data']
        model = results[background]['model']
        r2 = results[background]['r2']
        n = results[background]['n']

        # Convert to games
        year_n_games = data['Year_N_WAR'] * 16
        year_n_plus_1_games = data['Year_N_Plus_1_WAR'] * 16

        # Scatter plot
        ax.scatter(year_n_games, year_n_plus_1_games,
                  alpha=0.5, s=60, color=colors[background],
                  edgecolor='black', linewidth=0.5,
                  label=f'{background} (n={n}, R²={r2:.3f})')

        # Regression line
        x_range = np.linspace(data['Year_N_WAR'].min(), data['Year_N_WAR'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range * 16, y_pred * 16,
               color=colors[background], linewidth=3,
               linestyle='-', zorder=5)

    # Perfect persistence line (y=x)
    all_data = pd.concat([results['Offensive']['data'], results['Defensive']['data']])
    lim_min = min((all_data['Year_N_WAR'] * 16).min(), (all_data['Year_N_Plus_1_WAR'] * 16).min())
    lim_max = max((all_data['Year_N_WAR'] * 16).max(), (all_data['Year_N_Plus_1_WAR'] * 16).max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'gray', linewidth=2,
           linestyle='--', label='Perfect persistence (y=x)', zorder=3, alpha=0.7)

    # Reference lines at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    ax.set_xlabel('Year N Coaching WAR (Games per Season)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Year N+1 Coaching WAR (Games per Season)', fontsize=18, fontweight='bold')

    # Stats box with mathematical notation
    stats_text = "Regression Equations:\n"
    for background in ['Offensive', 'Defensive']:
        model = results[background]['model']
        r2 = results[background]['r2']
        p_value = results[background]['p_value']
        coef = model.coef_[0]
        intercept = model.intercept_ * 16
        stats_text += f"\n{background}:\n"
        stats_text += f"  $WAR_{{N+1}} = {coef:.3f} \\cdot WAR_N + {intercept:.2f}$ games\n"
        stats_text += f"  $R^2 = {r2:.3f}$\n"
        if p_value < 0.001:
            stats_text += f"  $p < 0.001$\n"
        else:
            stats_text += f"  $p = {p_value:.3f}$\n"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=13, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.legend(loc='lower right', fontsize=14, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_war_persistence_by_background.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_war_persistence_by_background.png")
    plt.close()

def main(font_family='Helvetica'):
    """Run persistence analysis by background."""

    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)

    # Load data
    df = load_coaching_data()

    # Create consecutive pairs with background
    lagged_df = create_consecutive_pairs_with_background(df)

    # Analyze by background
    results = analyze_by_background(lagged_df)

    # Create visualization
    create_background_comparison_scatter(results)

    # Save results
    lagged_df.to_csv('analysis/outputs/csv/coaching_war_persistence_by_background.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOffensive Coaches: R² = {results['Offensive']['r2']:.4f}")
    print(f"Defensive Coaches: R² = {results['Defensive']['r2']:.4f}")
    print(f"\nDifference: {results['Defensive']['r2'] - results['Offensive']['r2']:+.4f}")
    print("\nResults saved to:")
    print("  - analysis/outputs/csv/coaching_war_persistence_by_background.csv")
    print("  - analysis/outputs/png/coaching_war_persistence_by_background.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coaching WAR persistence by background')
    parser.add_argument('--font', type=str, default='Helvetica',
                       help='Font family to use (default: Helvetica)')
    args = parser.parse_args()

    main(font_family=args.font)
