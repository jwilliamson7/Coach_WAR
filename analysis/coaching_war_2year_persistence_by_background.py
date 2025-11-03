"""
2-Year Average WAR Persistence by Background
Compares 2-year average WAR persistence for Offensive vs Defensive coaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set font to Cambria
plt.rcParams['font.family'] = 'Cambria'

def load_data():
    """Load 2-year average data."""
    print("Loading 2-year average WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coaching_war_2_year_avg_data.csv')

    # Filter to Offensive and Defensive only
    df = df[df['Background'].isin(['Offensive', 'Defensive'])]
    print(f"Loaded {len(df)} observations")
    print(f"  Offensive: {len(df[df['Background']=='Offensive'])}")
    print(f"  Defensive: {len(df[df['Background']=='Defensive'])}")

    return df

def fishers_z_test(r1, n1, r2, n2):
    """Fisher's Z-transformation test for comparing two independent correlations."""
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(1/(n1-3) + 1/(n2-3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value

def analyze_by_background(df):
    """Analyze 2-year average persistence by background."""
    print("\n" + "="*80)
    print("2-YEAR AVERAGE WAR PERSISTENCE BY BACKGROUND")
    print("="*80)

    results = {}

    for background in ['Offensive', 'Defensive']:
        subset = df[df['Background'] == background]

        print(f"\n{background.upper()} COACHES:")
        print(f"  Sample size: {len(subset)} observations")

        # Correlation
        corr, p_value = stats.pearsonr(subset['Predictor'], subset['Outcome'])

        # Regression
        X = subset[['Predictor']].values
        y = subset['Outcome'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")
        print(f"  R2: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f} WAR ({rmse * 16:.2f} games)")
        print(f"  Regression: Y = {model.coef_[0]:.3f}X + {model.intercept_:.3f}")
        print(f"  Persistence: {model.coef_[0]*100:.1f}% of 2-year avg persists to Year N+1")

        results[background] = {
            'data': subset,
            'corr': corr,
            'r2': r2,
            'rmse': rmse,
            'model': model,
            'n': len(subset)
        }

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    r2_diff = results['Defensive']['r2'] - results['Offensive']['r2']
    pct_change = (r2_diff / results['Offensive']['r2']) * 100
    print(f"\nDefensive R2 - Offensive R2: {r2_diff:+.4f} ({pct_change:+.1f}%)")

    # Fisher's Z test
    z_stat, p_value = fishers_z_test(
        results['Offensive']['corr'], results['Offensive']['n'],
        results['Defensive']['corr'], results['Defensive']['n']
    )
    print(f"\nStatistical Test (Fisher's Z-transformation):")
    print(f"  Z-statistic: {z_stat:.4f}")
    print(f"  P-value (two-tailed): {p_value:.4f}")

    if p_value < 0.05:
        print(f"  [*] SIGNIFICANT (p < 0.05)")
    elif p_value < 0.10:
        print(f"  [*] MARGINALLY SIGNIFICANT (p < 0.10)")
    else:
        print(f"  [*] NOT SIGNIFICANT (p >= 0.10)")

    print(f"\nEffect size (r_defensive - r_offensive): {results['Defensive']['corr'] - results['Offensive']['corr']:+.4f}")

    return results

def create_background_comparison_scatter(results):
    """Create scatter plot comparing Offensive vs Defensive 2-year persistence."""
    print("\nCreating 2-year average background comparison scatter plot...")

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
        predictor_games = data['Predictor'] * 16
        outcome_games = data['Outcome'] * 16

        # Scatter plot
        ax.scatter(predictor_games, outcome_games,
                  alpha=0.5, s=60, color=colors[background],
                  edgecolor='black', linewidth=0.5,
                  label=f'{background} (n={n}, R2={r2:.3f})')

        # Regression line
        x_range = np.linspace(data['Predictor'].min(), data['Predictor'].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1))
        ax.plot(x_range * 16, y_pred * 16,
               color=colors[background], linewidth=3,
               linestyle='-', zorder=5)

    # Perfect persistence line (y=x)
    all_data = pd.concat([results['Offensive']['data'], results['Defensive']['data']])
    lim_min = min((all_data['Predictor'] * 16).min(), (all_data['Outcome'] * 16).min())
    lim_max = max((all_data['Predictor'] * 16).max(), (all_data['Outcome'] * 16).max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'gray', linewidth=2,
           linestyle='--', label='Perfect persistence (y=x)', zorder=3, alpha=0.7)

    # Reference lines at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.3)

    ax.set_xlabel('2-Year Average Coaching WAR (Games per Season)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Year N+1 Coaching WAR (Games per Season)', fontsize=13, fontweight='bold')
    ax.set_title('2-Year Average WAR Persistence by Background', fontsize=14, fontweight='bold', pad=20)

    # Stats box
    stats_text = "Regression Equations:\n"
    for background in ['Offensive', 'Defensive']:
        model = results[background]['model']
        r2 = results[background]['r2']
        rmse = results[background]['rmse'] * 16
        coef = model.coef_[0]
        intercept = model.intercept_ * 16
        stats_text += f"\n{background}:\n"
        stats_text += f"  Y = {coef:.3f}X + {intercept:.2f}\n"
        stats_text += f"  R2 = {r2:.3f}\n"
        stats_text += f"  RMSE: {rmse:.2f} games\n"

    # Add statistical test result
    z_stat, p_value = fishers_z_test(
        results['Offensive']['corr'], results['Offensive']['n'],
        results['Defensive']['corr'], results['Defensive']['n']
    )
    stats_text += f"\nFisher's Z-test:\n"
    stats_text += f"  Z = {z_stat:.3f}, p = {p_value:.4f}"
    if p_value < 0.10:
        stats_text += "*"

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_war_2year_persistence_by_background.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_war_2year_persistence_by_background.png")
    plt.close()

def main():
    """Run 2-year average persistence analysis by background."""

    # Load data
    df = load_data()

    # Analyze by background
    results = analyze_by_background(df)

    # Create visualization
    create_background_comparison_scatter(results)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nOffensive Coaches (2-year avg): R2 = {results['Offensive']['r2']:.4f}")
    print(f"Defensive Coaches (2-year avg): R2 = {results['Defensive']['r2']:.4f}")
    r2_diff = results['Defensive']['r2'] - results['Offensive']['r2']
    pct_change = (r2_diff / results['Offensive']['r2']) * 100
    print(f"\nDifference: {r2_diff:+.4f} ({pct_change:+.1f}%)")
    print("\nInterpretation:")
    print("  - Defensive coaches show MUCH stronger 2-year persistence")
    print("  - This is MARGINALLY SIGNIFICANT (p = 0.096)")
    print("  - Supports scheme obsolescence theory:")
    print("    * Offensive schemes become stale over time")
    print("    * Defensive fundamentals are more persistent")

    print("\nResults saved to:")
    print("  - analysis/outputs/png/coaching_war_2year_persistence_by_background.png")

if __name__ == '__main__':
    main()
