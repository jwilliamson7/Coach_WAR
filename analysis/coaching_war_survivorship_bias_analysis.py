"""
Coaching WAR Survivorship Bias Analysis
Tests whether regression to the mean is real or driven by survivorship bias.
Accounts for coaches who get fired after poor performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font to serif (Computer Modern for LaTeX compatibility)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern', 'DejaVu Serif', 'Times New Roman']

def load_coaching_data():
    """Load coaching WAR data."""
    print("Loading coaching WAR data...")
    df = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
    df = df.sort_values(['Primary_Coach', 'Team', 'Year']).reset_index(drop=True)
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_survivorship_dataset(df):
    """Create dataset tracking survival and Year N+1 outcomes."""
    print("\nAnalyzing coach survival patterns...")

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
                year_n_plus_1_war = np.nan
        else:
            # No data for this team next year (expansion, etc.)
            survived = None  # Exclude from analysis
            year_n_plus_1_war = np.nan

        records.append({
            'Coach': current['Primary_Coach'],
            'Team': current['Team'],
            'Year_N': int(current['Year']),
            'Year_N_WAR': current['Coaching_WAR'],
            'Year_N_Win_Pct': current['Actual_Win_Pct'],
            'Survived': survived,
            'Year_N_Plus_1_WAR': year_n_plus_1_war,
            'Background': current['Background']
        })

    survival_df = pd.DataFrame(records)

    # Remove cases where we can't determine survival (team gaps)
    survival_df = survival_df[survival_df['Survived'].notna()].copy()

    print(f"\nTotal seasons analyzed: {len(survival_df)}")
    print(f"Survived to Year N+1: {survival_df['Survived'].sum()} ({survival_df['Survived'].mean()*100:.1f}%)")
    print(f"Did not survive: {(~survival_df['Survived']).sum()} ({(~survival_df['Survived']).mean()*100:.1f}%)")

    return survival_df

def analyze_survival_by_performance(survival_df):
    """Analyze survival rates by performance quintile."""
    print("\n" + "="*80)
    print("SURVIVAL RATES BY YEAR N WAR QUINTILE")
    print("="*80)

    # Create quintiles
    survival_df['WAR_Quintile'] = pd.qcut(survival_df['Year_N_WAR'], q=5,
                                          labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'],
                                          duplicates='drop')

    quintile_survival = survival_df.groupby('WAR_Quintile').agg({
        'Year_N_WAR': ['mean', 'std'],
        'Survived': ['sum', 'count', 'mean']
    })

    # Flatten MultiIndex columns
    quintile_survival.columns = ['_'.join(col).strip() for col in quintile_survival.columns.values]
    quintile_survival.columns = ['Mean_WAR', 'Std_WAR', 'N_Survived', 'N_Total', 'Survival_Rate']

    # Ensure numeric types
    for col in quintile_survival.columns:
        quintile_survival[col] = pd.to_numeric(quintile_survival[col], errors='coerce')

    quintile_survival = quintile_survival.round(4)
    quintile_survival['N_Fired'] = quintile_survival['N_Total'] - quintile_survival['N_Survived']
    quintile_survival['Survival_Pct'] = (quintile_survival['Survival_Rate'] * 100).round(1)

    print("\n" + quintile_survival[['Mean_WAR', 'N_Total', 'N_Survived', 'N_Fired', 'Survival_Pct']].to_string())

    # Test for trend
    x = np.arange(len(quintile_survival))
    y = quintile_survival['Survival_Rate'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"\nLinear trend in survival rate:")
    print(f"  Slope: {slope:.4f} (change per quintile)")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.001:
        print(f"  HIGHLY significant relationship between performance and survival")
    elif p_value < 0.05:
        print(f"  Significant relationship between performance and survival")
    else:
        print(f"  No significant relationship")

    return quintile_survival, survival_df

def compare_with_without_survivorship(survival_df, quintile_survival):
    """Compare regression to mean with and without survivorship bias."""
    print("\n" + "="*80)
    print("REGRESSION TO MEAN: WITH vs WITHOUT SURVIVORSHIP BIAS")
    print("="*80)

    results = []

    for quintile in ['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)']:
        quintile_data = survival_df[survival_df['WAR_Quintile'] == quintile]

        # Survivors only (original analysis)
        survivors = quintile_data[quintile_data['Survived'] == True]
        mean_year_n = quintile_data['Year_N_WAR'].mean()

        if len(survivors) > 0:
            mean_year_n_plus_1_survivors = survivors['Year_N_Plus_1_WAR'].mean()
        else:
            mean_year_n_plus_1_survivors = np.nan

        # All coaches (assign penalty to fired coaches)
        # Use 10th percentile of all Year N+1 WAR as "fired" penalty
        fired_penalty = survival_df[survival_df['Survived'] == True]['Year_N_Plus_1_WAR'].quantile(0.10)

        # Create adjusted outcome
        adjusted_outcomes = []
        for _, row in quintile_data.iterrows():
            if row['Survived']:
                adjusted_outcomes.append(row['Year_N_Plus_1_WAR'])
            else:
                adjusted_outcomes.append(fired_penalty)

        mean_year_n_plus_1_adjusted = np.mean(adjusted_outcomes)

        # Calculate changes
        change_survivors_only = mean_year_n_plus_1_survivors - mean_year_n
        change_adjusted = mean_year_n_plus_1_adjusted - mean_year_n
        bias = change_survivors_only - change_adjusted

        results.append({
            'Quintile': quintile,
            'Year_N_Mean_WAR': mean_year_n,
            'Year_N_Plus_1_Survivors_Only': mean_year_n_plus_1_survivors,
            'Year_N_Plus_1_Adjusted': mean_year_n_plus_1_adjusted,
            'Change_Survivors_Only': change_survivors_only,
            'Change_Adjusted': change_adjusted,
            'Survivorship_Bias': bias,
            'N_Total': len(quintile_data),
            'N_Survived': len(survivors),
            'Survival_Rate': len(survivors) / len(quintile_data) if len(quintile_data) > 0 else 0
        })

    results_df = pd.DataFrame(results)

    print(f"\nFired coach penalty (10th percentile): {fired_penalty:.4f}")
    print("\nQuintile-by-quintile comparison:")
    print(results_df[['Quintile', 'Year_N_Mean_WAR', 'Year_N_Plus_1_Survivors_Only',
                     'Year_N_Plus_1_Adjusted', 'Survivorship_Bias']].to_string(index=False))

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    # Analyze Q1 specifically
    q1_results = results_df[results_df['Quintile'] == 'Q1 (Worst)'].iloc[0]
    print(f"\nQ1 (Worst Performers):")
    print(f"  Survivors only: {q1_results['Change_Survivors_Only']:+.4f} change")
    print(f"  With fired coaches: {q1_results['Change_Adjusted']:+.4f} change")
    print(f"  Survivorship bias: {q1_results['Survivorship_Bias']:+.4f}")
    print(f"  Survival rate: {q1_results['Survival_Rate']*100:.1f}%")

    if abs(q1_results['Survivorship_Bias']) > 0.02:
        print(f"\n  >> SUBSTANTIAL survivorship bias detected in Q1")
        print(f"  >> The apparent 'improvement' is partly due to fired coaches being excluded")
    else:
        print(f"\n  >> Minimal survivorship bias in Q1")
        print(f"  >> Regression to mean appears genuine")

    # Analyze Q5 specifically
    q5_results = results_df[results_df['Quintile'] == 'Q5 (Best)'].iloc[0]
    print(f"\nQ5 (Best Performers):")
    print(f"  Survivors only: {q5_results['Change_Survivors_Only']:+.4f} change")
    print(f"  With fired coaches: {q5_results['Change_Adjusted']:+.4f} change")
    print(f"  Survivorship bias: {q5_results['Survivorship_Bias']:+.4f}")
    print(f"  Survival rate: {q5_results['Survival_Rate']*100:.1f}%")

    return results_df, fired_penalty

def create_survivorship_visualizations(survival_df, quintile_survival, results_df, fired_penalty):
    """Create comprehensive visualizations."""
    print("\nCreating visualizations...")

    # Plot 1: Survival rates by quintile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x_pos = np.arange(len(quintile_survival))

    # Survival rate bar chart
    ax1.bar(x_pos, quintile_survival['Survival_Pct'], color='steelblue',
           alpha=0.7, edgecolor='black')
    ax1.axhline(y=survival_df['Survived'].mean()*100, color='red', linestyle='--',
               linewidth=2, label=f"Overall avg: {survival_df['Survived'].mean()*100:.1f}%")
    ax1.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Survival Rate to Year N+1 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Coach Survival Rates by Performance Quintile',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(quintile_survival.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(quintile_survival['Survival_Pct']):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Stacked bar: survived vs fired
    survived = quintile_survival['N_Survived'].values
    fired = quintile_survival['N_Fired'].values

    ax2.bar(x_pos, survived, label='Survived', color='green', alpha=0.7, edgecolor='black')
    ax2.bar(x_pos, fired, bottom=survived, label='Fired/Left', color='red',
           alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Coaches', fontsize=12, fontweight='bold')
    ax2.set_title('Coach Retention by Performance Quintile',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(quintile_survival.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_survival_by_performance.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_survival_by_performance.png")
    plt.close()

    # Plot 2: Simple regression to mean (Year N vs Year N+1 with fired penalty)
    # Convert to games (multiply by 16)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    x_pos = np.arange(len(results_df))
    width = 0.35

    # Convert WAR to games
    year_n_games = results_df['Year_N_Mean_WAR'] * 16
    year_n_plus_1_games = results_df['Year_N_Plus_1_Adjusted'] * 16
    fired_penalty_games = fired_penalty * 16

    # Top plot: Year N vs Year N+1 bars
    ax1.bar(x_pos - width/2, year_n_games, width,
           label='Year N', color='steelblue', alpha=0.8, edgecolor='black')
    ax1.bar(x_pos + width/2, year_n_plus_1_games, width,
           label='Year N+1', color='coral', alpha=0.8, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Coaching WAR (Games per Season)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Quintile'])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels to bars in top plot
    for i, (v_n, v_n1) in enumerate(zip(year_n_games, year_n_plus_1_games)):
        # Year N labels (left bars)
        label_n = f'+{v_n:.2f}' if v_n > 0 else f'{v_n:.2f}'
        ax1.text(i - width/2, v_n + (0.15 if v_n > 0 else -0.15), label_n,
                ha='center', va='bottom' if v_n > 0 else 'top', fontweight='bold', fontsize=10)
        # Year N+1 labels (right bars)
        label_n1 = f'+{v_n1:.2f}' if v_n1 > 0 else f'{v_n1:.2f}'
        ax1.text(i + width/2, v_n1 + (0.15 if v_n1 > 0 else -0.15), label_n1,
                ha='center', va='bottom' if v_n1 > 0 else 'top', fontweight='bold', fontsize=10)

    # Adjust y-axis limits to show labels without clipping
    ymin, ymax = ax1.get_ylim()
    y_range = ymax - ymin
    ax1.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    # Bottom plot: Change from Year N to Year N+1
    changes_games = year_n_plus_1_games - year_n_games
    colors = ['red' if x < 0 else 'green' for x in changes_games]
    ax2.bar(x_pos, changes_games, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change in WAR (Games)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['Quintile'])
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(changes_games):
        label = f'+{v:.2f}' if v > 0 else f'{v:.2f}'
        ax2.text(i, v + (0.08 if v > 0 else -0.08), label,
                ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    # Adjust y-axis limits to show labels
    ymin, ymax = ax2.get_ylim()
    y_range = ymax - ymin
    ax2.set_ylim(ymin - 0.1 * y_range, ymax + 0.1 * y_range)

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_regression_to_mean_survivorship_adjusted.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_regression_to_mean_survivorship_adjusted.png")
    plt.close()

    # Plot 3: Survivorship bias magnitude by quintile
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red' if x > 0 else 'green' for x in results_df['Survivorship_Bias']]
    ax.bar(x_pos, results_df['Survivorship_Bias'], color=colors, alpha=0.7,
          edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)

    ax.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survivorship Bias (WAR units)', fontsize=12, fontweight='bold')
    ax.set_title('Magnitude of Survivorship Bias by Quintile\n' +
                '(Positive = survivors-only analysis overstates improvement)',
                fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(results_df['Quintile'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(results_df['Survivorship_Bias']):
        ax.text(i, v + (0.003 if v > 0 else -0.003), f'{v:+.4f}',
               ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')

    plt.tight_layout()
    plt.savefig('analysis/outputs/png/coaching_survivorship_bias_magnitude.png',
               dpi=300, bbox_inches='tight')
    print("  Saved: coaching_survivorship_bias_magnitude.png")
    plt.close()

def analyze_by_tenure(survival_df):
    """Analyze whether survivorship bias differs by coach tenure."""
    print("\n" + "="*80)
    print("SURVIVORSHIP BIAS BY COACH TENURE")
    print("="*80)

    # Calculate tenure (years with current team up to Year N)
    survival_df['Tenure'] = 1  # Placeholder - will calculate properly

    # For each coach-team-year, count how many years they've been there
    for idx, row in survival_df.iterrows():
        coach = row['Coach']
        team = row['Team']
        year_n = row['Year_N']

        # Count prior years with same coach-team
        prior_years = survival_df[(survival_df['Coach'] == coach) &
                                 (survival_df['Team'] == team) &
                                 (survival_df['Year_N'] < year_n)]

        survival_df.at[idx, 'Tenure'] = len(prior_years) + 1

    # Categorize tenure
    survival_df['Tenure_Category'] = pd.cut(survival_df['Tenure'],
                                           bins=[0, 1, 2, 3, 100],
                                           labels=['Year 1', 'Year 2', 'Year 3', 'Year 4+'],
                                           right=True)

    print("\nSurvival rates by tenure:")
    tenure_survival = survival_df.groupby('Tenure_Category')['Survived'].agg(['mean', 'count'])
    tenure_survival.columns = ['Survival_Rate', 'N']
    # Ensure numeric types
    tenure_survival['Survival_Rate'] = pd.to_numeric(tenure_survival['Survival_Rate'], errors='coerce')
    tenure_survival['N'] = pd.to_numeric(tenure_survival['N'], errors='coerce')
    tenure_survival['Survival_Pct'] = (tenure_survival['Survival_Rate'] * 100).round(1)
    print(tenure_survival[['N', 'Survival_Pct']].to_string())

    # Analyze worst quintile by tenure
    print("\nQ1 (Worst) survival by tenure:")
    q1_data = survival_df[survival_df['WAR_Quintile'] == 'Q1 (Worst)']
    q1_tenure = q1_data.groupby('Tenure_Category')['Survived'].agg(['mean', 'count'])
    q1_tenure.columns = ['Survival_Rate', 'N']
    # Ensure numeric types
    q1_tenure['Survival_Rate'] = pd.to_numeric(q1_tenure['Survival_Rate'], errors='coerce')
    q1_tenure['N'] = pd.to_numeric(q1_tenure['N'], errors='coerce')
    q1_tenure['Survival_Pct'] = (q1_tenure['Survival_Rate'] * 100).round(1)
    print(q1_tenure[['N', 'Survival_Pct']].to_string())

    print("\nInsights:")
    year1_survival = q1_tenure.loc['Year 1', 'Survival_Pct'] if 'Year 1' in q1_tenure.index else np.nan
    year4plus_survival = q1_tenure.loc['Year 4+', 'Survival_Pct'] if 'Year 4+' in q1_tenure.index else np.nan

    if pd.notna(year1_survival) and pd.notna(year4plus_survival):
        print(f"  Year 1 coaches in Q1: {year1_survival:.1f}% survive")
        print(f"  Year 4+ coaches in Q1: {year4plus_survival:.1f}% survive")
        print(f"  Difference: {year4plus_survival - year1_survival:+.1f} percentage points")
        print(f"\n  Established coaches have more job security even after bad seasons")

    return survival_df

def main():
    """Run survivorship bias analysis."""

    # Load data
    df = load_coaching_data()

    # Create survivorship dataset
    survival_df = create_survivorship_dataset(df)

    # Analyze survival by performance
    quintile_survival, survival_df = analyze_survival_by_performance(survival_df)

    # Compare with/without survivorship bias
    results_df, fired_penalty = compare_with_without_survivorship(survival_df, quintile_survival)

    # Create visualizations
    create_survivorship_visualizations(survival_df, quintile_survival, results_df, fired_penalty)

    # Analyze by tenure
    survival_df = analyze_by_tenure(survival_df)

    # Save results
    survival_df.to_csv('analysis/outputs/csv/coaching_survivorship_data.csv', index=False)
    quintile_survival.to_csv('analysis/outputs/csv/coaching_survival_by_quintile.csv')
    results_df.to_csv('analysis/outputs/csv/coaching_regression_to_mean_survivorship_adjusted.csv', index=False)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")

    q1_survival = quintile_survival.loc['Q1 (Worst)', 'Survival_Pct']
    q5_survival = quintile_survival.loc['Q5 (Best)', 'Survival_Pct']

    print(f"\nSurvival rates:")
    print(f"  Q1 (Worst): {q1_survival:.1f}%")
    print(f"  Q5 (Best): {q5_survival:.1f}%")
    print(f"  Gap: {q5_survival - q1_survival:.1f} percentage points")

    q1_bias = results_df[results_df['Quintile'] == 'Q1 (Worst)']['Survivorship_Bias'].values[0]
    print(f"\nSurvivorship bias in Q1:")
    print(f"  {q1_bias:+.4f} WAR units")
    print(f"  ({q1_bias*16:+.2f} games over 16-game season)")

    if abs(q1_bias) > 0.02:
        print(f"\n  [!] SUBSTANTIAL bias - regression to mean is partly survivorship effect")
    else:
        print(f"\n  [~] Modest bias - regression to mean is mostly genuine")

    print("\nFiles saved to:")
    print("  - analysis/outputs/csv/coaching_survivorship_data.csv")
    print("  - analysis/outputs/csv/coaching_survival_by_quintile.csv")
    print("  - analysis/outputs/csv/coaching_regression_to_mean_survivorship_adjusted.csv")
    print("  - analysis/outputs/png/coaching_survival_by_performance.png")
    print("  - analysis/outputs/png/coaching_regression_to_mean_survivorship_adjusted.png")
    print("  - analysis/outputs/png/coaching_survivorship_bias_magnitude.png")

if __name__ == '__main__':
    main()
