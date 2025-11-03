"""
Coaching WAR Persistence with Fired Coaches Included
Tests persistence when accounting for coaches who got fired.
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

def load_coaching_data():
    """Load coaching WAR data."""
    print("Loading coaching WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    df = df.sort_values(['Primary_Coach', 'Team', 'Year']).reset_index(drop=True)
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_dataset_with_fired_penalty(df):
    """Create dataset including fired coaches with penalty."""
    print("\nCreating dataset with fired coach penalty...")

    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    records = []

    for idx in range(len(df)):
        current = df.iloc[idx]

        # Check if this coach continues with the same team next year
        next_year_rows = df[(df['Team'] == current['Team']) &
                           (df['Year'] == current['Year'] + 1)]

        if len(next_year_rows) > 0:
            next_year_coach = next_year_rows.iloc[0]['Primary_Coach']

            # Did the same coach continue?
            if next_year_coach == current['Primary_Coach']:
                survived = True
                year_n_plus_1_war = next_year_rows.iloc[0]['Coaching_WAR']
            else:
                survived = False
                year_n_plus_1_war = None  # Will assign penalty
        else:
            # No data for this team next year (skip)
            continue

        records.append({
            'Coach': current['Primary_Coach'],
            'Team': current['Team'],
            'Year_N': int(current['Year']),
            'Year_N_WAR': current['Coaching_WAR'],
            'Survived': survived,
            'Year_N_Plus_1_WAR': year_n_plus_1_war
        })

    result_df = pd.DataFrame(records)

    # Calculate 10th percentile penalty from survivors
    survivors = result_df[result_df['Survived'] == True]
    fired_penalty = survivors['Year_N_Plus_1_WAR'].quantile(0.10)

    print(f"\n10th percentile penalty: {fired_penalty:.4f} WAR ({fired_penalty * 16:.2f} games)")

    # Assign penalty to fired coaches
    result_df.loc[result_df['Survived'] == False, 'Year_N_Plus_1_WAR'] = fired_penalty

    print(f"\nTotal observations: {len(result_df)}")
    print(f"  Survived: {result_df['Survived'].sum()} ({result_df['Survived'].mean()*100:.1f}%)")
    print(f"  Fired/Left: {(~result_df['Survived']).sum()} ({(~result_df['Survived']).mean()*100:.1f}%)")

    return result_df, fired_penalty

def analyze_persistence_with_fired(df, fired_penalty):
    """Analyze persistence including fired coaches."""
    print("\n" + "="*80)
    print("WAR PERSISTENCE INCLUDING FIRED COACHES")
    print("="*80)

    # Overall correlation
    corr, p_value = stats.pearsonr(df['Year_N_WAR'], df['Year_N_Plus_1_WAR'])
    spearman_corr, spearman_p = stats.spearmanr(df['Year_N_WAR'], df['Year_N_Plus_1_WAR'])

    # Regression
    X = df[['Year_N_WAR']].values
    y = df['Year_N_Plus_1_WAR'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    print(f"\nWith Fired Coaches (10th percentile penalty):")
    print(f"  Sample size: {len(df)}")
    print(f"  Pearson r: {corr:.4f} (p={p_value:.6f})")
    print(f"  Spearman rho: {spearman_corr:.4f} (p={spearman_p:.6f})")
    print(f"  R²: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} WAR ({rmse * 16:.2f} games)")
    print(f"  Regression: Year N+1 WAR = {model.coef_[0]:.4f} * Year N WAR + {model.intercept_:.4f}")

    # Compare to survivors-only
    survivors_only = df[df['Survived'] == True]
    corr_surv, _ = stats.pearsonr(survivors_only['Year_N_WAR'], survivors_only['Year_N_Plus_1_WAR'])
    X_surv = survivors_only[['Year_N_WAR']].values
    y_surv = survivors_only['Year_N_Plus_1_WAR'].values
    model_surv = LinearRegression()
    model_surv.fit(X_surv, y_surv)
    r2_surv = r2_score(y_surv, model_surv.predict(X_surv))

    print(f"\nSurvivors Only (for comparison):")
    print(f"  Sample size: {len(survivors_only)}")
    print(f"  Pearson r: {corr_surv:.4f}")
    print(f"  R²: {r2_surv:.4f}")

    print(f"\nImpact of Including Fired Coaches:")
    print(f"  R² change: {r2 - r2_surv:+.4f} ({(r2 - r2_surv)/r2_surv * 100:+.1f}%)")
    print(f"  Correlation change: {corr - corr_surv:+.4f}")

    return corr, r2, model, corr_surv, r2_surv

def create_comparison_scatter(df, corr, r2, model, corr_surv, r2_surv, fired_penalty):
    """Create scatter plot comparing with/without fired coaches."""
    print("\nCreating comparison scatter plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Convert to games
    year_n_games = df['Year_N_WAR'] * 16
    year_n_plus_1_games = df['Year_N_Plus_1_WAR'] * 16
    fired_penalty_games = fired_penalty * 16

    # Left plot: Survivors only
    survivors = df[df['Survived'] == True]
    ax1.scatter(survivors['Year_N_WAR'] * 16, survivors['Year_N_Plus_1_WAR'] * 16,
               alpha=0.4, s=50, color='steelblue', edgecolor='black', linewidth=0.5,
               label=f'Survived (n={len(survivors)})')

    # Regression line for survivors
    X_surv = survivors[['Year_N_WAR']].values
    y_surv = survivors['Year_N_Plus_1_WAR'].values
    model_surv = LinearRegression()
    model_surv.fit(X_surv, y_surv)
    x_range = np.linspace(df['Year_N_WAR'].min(), df['Year_N_WAR'].max(), 100)
    y_pred_surv = model_surv.predict(x_range.reshape(-1, 1))
    ax1.plot(x_range * 16, y_pred_surv * 16, 'r-', linewidth=3, label='Regression line', zorder=5)

    # Perfect persistence line
    lim_min = min(year_n_games.min(), year_n_plus_1_games.min())
    lim_max = max(year_n_games.max(), year_n_plus_1_games.max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2,
            label='Perfect persistence', zorder=5)

    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Year N Coaching WAR (Games per Season)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Year N+1 Coaching WAR (Games per Season)', fontsize=12, fontweight='bold')
    ax1.set_title('Survivors Only', fontsize=13, fontweight='bold')

    # Stats box for survivors
    rmse_surv = np.sqrt(mean_squared_error(y_surv, model_surv.predict(X_surv))) * 16
    stats_text_surv = f"R² = {r2_surv:.3f}\n"
    stats_text_surv += f"n = {len(survivors)} season pairs\n"
    stats_text_surv += f"RMSE: {rmse_surv:.2f} games"
    ax1.text(0.05, 0.95, stats_text_surv, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right plot: Including fired coaches
    ax2.scatter(survivors['Year_N_WAR'] * 16, survivors['Year_N_Plus_1_WAR'] * 16,
               alpha=0.4, s=50, color='steelblue', edgecolor='black', linewidth=0.5,
               label=f'Survived (n={len(survivors)})')

    # Add fired coaches
    fired = df[df['Survived'] == False]
    ax2.scatter(fired['Year_N_WAR'] * 16, fired['Year_N_Plus_1_WAR'] * 16,
               alpha=0.6, s=50, color='red', marker='x', linewidth=2,
               label=f'Fired (n={len(fired)})', zorder=4)

    # Regression line including fired
    x_range = np.linspace(df['Year_N_WAR'].min(), df['Year_N_WAR'].max(), 100)
    y_pred = model.predict(x_range.reshape(-1, 1))
    ax2.plot(x_range * 16, y_pred * 16, 'r-', linewidth=3, label='Regression line', zorder=5)

    # Perfect persistence line
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'g--', linewidth=2,
            label='Perfect persistence', zorder=5)

    # Fired penalty line
    ax2.axhline(y=fired_penalty_games, color='darkred', linestyle=':', linewidth=2,
               label=f'Fired penalty: {fired_penalty_games:.2f} games', zorder=3)

    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Year N Coaching WAR (Games per Season)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Year N+1 Coaching WAR (Games per Season)', fontsize=12, fontweight='bold')
    ax2.set_title('Including Fired Coaches (10th Percentile Penalty)', fontsize=13, fontweight='bold')

    # Stats box for all coaches
    rmse_all = np.sqrt(mean_squared_error(df['Year_N_Plus_1_WAR'], model.predict(df[['Year_N_WAR']]))) * 16
    stats_text_all = f"R² = {r2:.3f}\n"
    stats_text_all += f"n = {len(df)} season pairs\n"
    stats_text_all += f"RMSE: {rmse_all:.2f} games"
    ax2.text(0.05, 0.95, stats_text_all, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Coaching WAR Persistence: Impact of Including Fired Coaches',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_war_persistence_with_fired_comparison.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_war_persistence_with_fired_comparison.png")
    plt.close()

def main():
    """Run persistence analysis with fired coaches."""

    # Load data
    df = load_coaching_data()

    # Create dataset with fired penalty
    analysis_df, fired_penalty = create_dataset_with_fired_penalty(df)

    # Analyze persistence
    corr, r2, model, corr_surv, r2_surv = analyze_persistence_with_fired(analysis_df, fired_penalty)

    # Create visualization
    create_comparison_scatter(analysis_df, corr, r2, model, corr_surv, r2_surv, fired_penalty)

    # Save results
    analysis_df.to_csv('analysis/outputs/csv/coaching_war_persistence_with_fired.csv', index=False)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nSurvivors Only: R² = {r2_surv:.4f}")
    print(f"Including Fired: R² = {r2:.4f}")
    print(f"Change: {r2 - r2_surv:+.4f} ({(r2 - r2_surv)/r2_surv * 100:+.1f}%)")

    if r2 < r2_surv:
        print(f"\n[*] Including fired coaches REDUCES R² by {abs((r2 - r2_surv)/r2_surv * 100):.1f}%")
        print(f"    This makes sense: fired coaches had poor Year N but assigned fixed penalty")
        print(f"    Adds noise/weakens the Year N -> Year N+1 relationship")
    else:
        print(f"\n[*] Including fired coaches INCREASES R² by {abs((r2 - r2_surv)/r2_surv * 100):.1f}%")
        print(f"    Fired coaches strengthen the negative relationship at low Year N WAR")

    print("\nResults saved to:")
    print("  - analysis/outputs/csv/coaching_war_persistence_with_fired.csv")
    print("  - analysis/outputs/png/coaching_war_persistence_with_fired_comparison.png")

if __name__ == '__main__':
    main()
