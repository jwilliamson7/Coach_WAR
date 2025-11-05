"""
Win Percentage Persistence Analysis
Analyzes year-to-year persistence of team winning percentage.
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
    """Load coaching WAR data with Actual_Actual_Win_Pct."""
    print("Loading coaching data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    df = df.sort_values(['Primary_Coach', 'Team', 'Year']).reset_index(drop=True)
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_consecutive_pairs(df):
    """Create dataset of consecutive season pairs (same coach, same team)."""
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
                'Year_N': int(current['Year']),
                'Year_N_Actual_Win_Pct': current['Actual_Win_Pct'],
                'Year_N_Plus_1_Actual_Win_Pct': next_row['Actual_Win_Pct']
            })

    result_df = pd.DataFrame(consecutive_pairs)
    print(f"Created {len(result_df)} consecutive season pairs")

    return result_df

def analyze_win_pct_persistence(lagged_df):
    """Analyze Win Pct persistence."""
    print("\n" + "="*80)
    print("WIN PERCENTAGE PERSISTENCE ANALYSIS")
    print("="*80)

    # Remove any rows with missing data
    lagged_df = lagged_df.dropna(subset=['Year_N_Actual_Win_Pct', 'Year_N_Plus_1_Actual_Win_Pct'])

    # Correlation analysis
    corr, p_value = stats.pearsonr(lagged_df['Year_N_Actual_Win_Pct'],
                                    lagged_df['Year_N_Plus_1_Actual_Win_Pct'])
    spearman_corr, spearman_p = stats.spearmanr(lagged_df['Year_N_Actual_Win_Pct'],
                                                  lagged_df['Year_N_Plus_1_Actual_Win_Pct'])

    # Regression analysis
    X = lagged_df[['Year_N_Actual_Win_Pct']].values
    y = lagged_df['Year_N_Plus_1_Actual_Win_Pct'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\nSample size: {len(lagged_df)} consecutive season pairs")
    print(f"Pearson r: {corr:.4f} (p={p_value:.6f})")
    print(f"Spearman rho: {spearman_corr:.4f} (p={spearman_p:.6f})")
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Regression equation: Year N+1 Win% = {model.coef_[0]:.4f} * Year N Win% + {model.intercept_:.4f}")
    print(f"Regression coefficient: {model.coef_[0]:.4f} (proportion of Year N Win% that persists)")

    # Interpret results
    print("\nInterpretation:")
    if r2 > 0.10:
        print(f"  [*] Moderate persistence: {r2*100:.1f}% of Year N+1 Win% variance explained by Year N Win%")
    else:
        print(f"  [*] Low persistence: Only {r2*100:.1f}% of Year N+1 Win% variance explained by Year N Win%")

    print(f"  [*] For every 1.0 increase in Year N Win%, expect {model.coef_[0]:.3f} increase in Year N+1")
    print(f"  [*] Strong regression to mean: Only {model.coef_[0]*100:.1f}% of Year N Win% persists to Year N+1")

    return corr, r2, model, p_value, lagged_df

def create_persistence_scatter(lagged_df, corr, r2, model, p_value):
    """Create scatter plot of Win Pct persistence."""
    print("\nCreating Win% persistence scatter plot...")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Scatter plot
    ax.scatter(lagged_df['Year_N_Actual_Win_Pct'], lagged_df['Year_N_Plus_1_Actual_Win_Pct'],
              alpha=0.4, s=50, color='steelblue', edgecolor='black', linewidth=0.5)

    # Regression line
    x_range = np.linspace(lagged_df['Year_N_Actual_Win_Pct'].min(),
                         lagged_df['Year_N_Actual_Win_Pct'].max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred, 'r-', linewidth=3, label='Regression line', zorder=5)

    # Perfect persistence line (y=x)
    lim_min = min(lagged_df['Year_N_Actual_Win_Pct'].min(), lagged_df['Year_N_Plus_1_Actual_Win_Pct'].min())
    lim_max = max(lagged_df['Year_N_Actual_Win_Pct'].max(), lagged_df['Year_N_Plus_1_Actual_Win_Pct'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2,
           label='Perfect persistence (y=x)', zorder=5)

    # Reference line at 0.500
    ax.axhline(y=0.500, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(x=0.500, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('Year N Winning Percentage', fontsize=18, fontweight='bold')
    ax.set_ylabel('Year N+1 Winning Percentage', fontsize=18, fontweight='bold')

    # Stats box with mathematical notation
    stats_text = f"$Win\\%_{{N+1}} = {model.coef_[0]:.3f} \\cdot Win\\%_N + {model.intercept_:.3f}$\n"
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
    plt.savefig('analysis/outputs/png/win_pct_persistence_scatter.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: win_pct_persistence_scatter.png")
    plt.close()

def main(font_family='Helvetica'):
    """Run Win Pct persistence analysis."""

    # Configure matplotlib fonts
    configure_matplotlib_fonts(font_family)

    # Load data
    df = load_coaching_data()

    # Create consecutive pairs
    lagged_df = create_consecutive_pairs(df)

    # Analyze persistence
    corr, r2, model, p_value, lagged_df = analyze_win_pct_persistence(lagged_df)

    # Create visualization
    create_persistence_scatter(lagged_df, corr, r2, model, p_value)

    # Save results
    lagged_df.to_csv('analysis/outputs/csv/win_pct_persistence_data.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Win Percentage shows {'stronger' if r2 > 0.10 else 'weak'} year-to-year persistence")
    print(f"R² = {r2:.4f} ({r2*100:.1f}% of variance explained)")
    print(f"Regression coefficient: {model.coef_[0]:.4f} ({model.coef_[0]*100:.1f}% persists)")
    print(f"\nCompare to Coaching WAR persistence: R² ~ 0.070 (7%)")
    print(f"Win% is {r2/0.070:.1f}x more persistent than Coaching WAR")

    print("\nResults saved to:")
    print("  - analysis/outputs/csv/win_pct_persistence_data.csv")
    print("  - analysis/outputs/png/win_pct_persistence_scatter.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Win percentage persistence analysis')
    parser.add_argument('--font', type=str, default='Helvetica',
                       help='Font family to use (default: Helvetica)')
    args = parser.parse_args()

    main(font_family=args.font)
