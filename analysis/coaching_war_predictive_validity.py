"""
Coaching WAR Predictive Validity Analysis
Tests whether Year N coaching WAR predicts Year N+1 team performance.
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
    print(f"Loaded {len(df)} coach-season records")
    return df

def create_lagged_dataset(df):
    """Create dataset with Year N metrics matched to Year N+1 outcomes."""
    print("\nCreating lagged dataset...")

    # Sort by team, coach, and year
    df = df.sort_values(['Team', 'Primary_Coach', 'Year']).reset_index(drop=True)

    # Create lagged records
    lagged_records = []

    for idx in range(len(df) - 1):
        current = df.iloc[idx]
        next_row = df.iloc[idx + 1]

        # Check if consecutive years with same coach and team
        if (current['Team'] == next_row['Team'] and
            current['Primary_Coach'] == next_row['Primary_Coach'] and
            next_row['Year'] == current['Year'] + 1):

            lagged_records.append({
                'Team': current['Team'],
                'Coach': current['Primary_Coach'],
                'Year_N': int(current['Year']),
                'Year_N_Plus_1': int(next_row['Year']),
                # Year N predictors
                'Year_N_WAR': current['Coaching_WAR'],
                'Year_N_Win_Pct': current['Actual_Win_Pct'],
                'Year_N_Predicted_Win_Pct': current['Predicted_With_Coach'],
                'Year_N_Replacement_Prediction': current['Predicted_Replacement'],
                'Year_N_Predicted_Impact': current['Predicted_Impact'],
                # Year N+1 outcome
                'Year_N_Plus_1_Win_Pct': next_row['Actual_Win_Pct'],
                'Year_N_Plus_1_WAR': next_row['Coaching_WAR'],
                # Background info
                'Background': current['Background']
            })

    lagged_df = pd.DataFrame(lagged_records)
    print(f"Created {len(lagged_df)} consecutive season pairs")
    print(f"Years: {int(lagged_df['Year_N'].min())}-{int(lagged_df['Year_N'].max())}")
    print(f"Unique coaches: {lagged_df['Coach'].nunique()}")

    return lagged_df

def analyze_predictive_power(lagged_df):
    """Analyze correlation and predictive power of different metrics."""
    print("\n" + "="*80)
    print("PREDICTIVE VALIDITY ANALYSIS")
    print("="*80)

    # Define predictors to test
    predictors = {
        'Year N WAR': 'Year_N_WAR',
        'Year N Win%': 'Year_N_Win_Pct',
        'Year N Model Prediction': 'Year_N_Predicted_Win_Pct',
        'Year N Predicted Impact': 'Year_N_Predicted_Impact'
    }

    results = []

    for name, col in predictors.items():
        # Calculate correlation
        corr, p_value = stats.pearsonr(lagged_df[col], lagged_df['Year_N_Plus_1_Win_Pct'])

        # Calculate Spearman (rank) correlation
        spearman_corr, spearman_p = stats.spearmanr(lagged_df[col], lagged_df['Year_N_Plus_1_Win_Pct'])

        # Fit linear regression
        X = lagged_df[[col]].values
        y = lagged_df['Year_N_Plus_1_Win_Pct'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        results.append({
            'Predictor': name,
            'Pearson_r': corr,
            'Pearson_p': p_value,
            'Spearman_rho': spearman_corr,
            'Spearman_p': spearman_p,
            'R2': r2,
            'RMSE': rmse,
            'Coefficient': model.coef_[0],
            'Intercept': model.intercept_
        })

        print(f"\n{name} -> Year N+1 Win%:")
        print(f"  Pearson r: {corr:.4f} (p={p_value:.4f})")
        print(f"  Spearman rho: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Regression: Y = {model.coef_[0]:.4f}X + {model.intercept_:.4f}")

    results_df = pd.DataFrame(results)

    # Compare WAR to Win% baseline
    print("\n" + "="*80)
    print("COMPARISON: WAR vs Win% as Predictors")
    print("="*80)

    war_r2 = results_df[results_df['Predictor'] == 'Year N WAR']['R2'].values[0]
    winpct_r2 = results_df[results_df['Predictor'] == 'Year N Win%']['R2'].values[0]

    print(f"\nYear N WAR R²: {war_r2:.4f}")
    print(f"Year N Win% R²: {winpct_r2:.4f}")
    print(f"Difference: {war_r2 - winpct_r2:.4f}")

    if war_r2 > winpct_r2:
        print("\n[+] WAR is MORE predictive than prior year Win%")
        print("  This suggests WAR captures persistent coaching ability beyond luck")
    else:
        print("\n[-] WAR is LESS predictive than prior year Win%")
        print("  This suggests WAR may not fully separate skill from luck")

    return results_df, lagged_df

def create_scatter_plots(lagged_df, results_df):
    """Create scatter plots showing predictive relationships."""
    print("\nCreating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Coaching Metrics: Predictive Validity for Year N+1 Performance',
                 fontsize=14, fontweight='bold')

    predictors = [
        ('Year_N_WAR', 'Year N Coaching WAR'),
        ('Year_N_Win_Pct', 'Year N Win%'),
        ('Year_N_Predicted_Win_Pct', 'Year N Model Prediction'),
        ('Year_N_Predicted_Impact', 'Year N Predicted Impact')
    ]

    for idx, (col, title) in enumerate(predictors):
        ax = axes[idx // 2, idx % 2]

        # Scatter plot
        ax.scatter(lagged_df[col], lagged_df['Year_N_Plus_1_Win_Pct'],
                  alpha=0.5, s=30, color='steelblue')

        # Regression line
        X = lagged_df[[col]].values
        y = lagged_df['Year_N_Plus_1_Win_Pct'].values
        model = LinearRegression()
        model.fit(X, y)

        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax.plot(x_range, y_pred, 'r-', linewidth=2, label='Regression line')

        # Get statistics
        result = results_df[results_df['Predictor'].str.contains(title.split()[1])].iloc[0]

        # Labels and title
        ax.set_xlabel(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Year N+1 Win%', fontsize=11, fontweight='bold')
        ax.set_title(f'r = {result["Pearson_r"]:.3f}, R² = {result["R2"]:.3f}, p = {result["Pearson_p"]:.4f}',
                    fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_path = 'analysis/outputs/png/coaching_war_predictive_validity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()

def analyze_by_war_quintiles(lagged_df):
    """Analyze Year N+1 performance by Year N WAR quintiles."""
    print("\n" + "="*80)
    print("QUINTILE ANALYSIS: Year N+1 Performance by Year N WAR")
    print("="*80)

    # Create WAR quintiles
    lagged_df['WAR_Quintile'] = pd.qcut(lagged_df['Year_N_WAR'], q=5,
                                         labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'],
                                         duplicates='drop')

    # Calculate average Year N+1 performance by quintile
    quintile_stats = lagged_df.groupby('WAR_Quintile').agg({
        'Year_N_WAR': ['mean', 'count'],
        'Year_N_Win_Pct': 'mean',
        'Year_N_Plus_1_Win_Pct': 'mean'
    }).round(4)

    quintile_stats.columns = ['Avg_Year_N_WAR', 'Count', 'Avg_Year_N_Win_Pct', 'Avg_Year_N_Plus_1_Win_Pct']

    print("\nAverage Year N+1 Win% by Year N WAR Quintile:")
    print(quintile_stats.to_string())

    # Test for monotonic relationship
    quintile_means = lagged_df.groupby('WAR_Quintile')['Year_N_Plus_1_Win_Pct'].mean().values
    x = np.arange(len(quintile_means))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, quintile_means)

    print(f"\nLinear trend across quintiles:")
    print(f"  Slope: {slope:.4f} win% per quintile")
    print(f"  R²: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  [+] Significant monotonic relationship detected")
    else:
        print("  [-] No significant monotonic relationship")

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(quintile_stats))
    ax.bar(x_pos, quintile_stats['Avg_Year_N_Plus_1_Win_Pct'],
           color='steelblue', alpha=0.7, edgecolor='black')

    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='0.500 (Average)')
    ax.axhline(y=lagged_df['Year_N_Plus_1_Win_Pct'].mean(),
              color='green', linestyle='--', linewidth=1, label='Sample Mean')

    ax.set_xlabel('Year N WAR Quintile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Year N+1 Win%', fontsize=12, fontweight='bold')
    ax.set_title('Year N+1 Performance by Year N Coaching WAR Quintile',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(quintile_stats.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, v in enumerate(quintile_stats['Avg_Year_N_Plus_1_Win_Pct']):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_path = 'analysis/outputs/png/coaching_war_quintile_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved quintile plot to: {output_path}")
    plt.close()

    return quintile_stats

def analyze_war_persistence(lagged_df):
    """Analyze whether WAR itself persists year-to-year."""
    print("\n" + "="*80)
    print("WAR PERSISTENCE ANALYSIS")
    print("="*80)

    corr, p_value = stats.pearsonr(lagged_df['Year_N_WAR'], lagged_df['Year_N_Plus_1_WAR'])
    spearman_corr, spearman_p = stats.spearmanr(lagged_df['Year_N_WAR'], lagged_df['Year_N_Plus_1_WAR'])

    print(f"\nYear N WAR -> Year N+1 WAR:")
    print(f"  Pearson r: {corr:.4f} (p={p_value:.4f})")
    print(f"  Spearman rho: {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"  R²: {corr**2:.4f}")

    if corr**2 > 0.2:
        print("\n  [+] Strong WAR persistence detected")
        print("  This suggests coaching ability is relatively stable")
    elif corr**2 > 0.1:
        print("\n  [~] Moderate WAR persistence detected")
        print("  Some consistency but also substantial year-to-year variation")
    else:
        print("\n  [-] Weak WAR persistence")
        print("  High year-to-year variation suggests noise or context-dependence")

def main():
    """Run predictive validity analysis."""

    # Load data
    df = load_coaching_data()

    # Create lagged dataset
    lagged_df = create_lagged_dataset(df)

    # Analyze predictive power
    results_df, lagged_df = analyze_predictive_power(lagged_df)

    # Create visualizations
    create_scatter_plots(lagged_df, results_df)

    # Quintile analysis
    quintile_stats = analyze_by_war_quintiles(lagged_df)

    # WAR persistence analysis
    analyze_war_persistence(lagged_df)

    # Save results
    results_df.to_csv('analysis/outputs/csv/coaching_war_predictive_validity_results.csv', index=False)
    quintile_stats.to_csv('analysis/outputs/csv/coaching_war_quintile_analysis.csv')

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAnalyzed {len(lagged_df)} consecutive season pairs")
    print(f"Results saved to:")
    print(f"  - analysis/outputs/csv/coaching_war_predictive_validity_results.csv")
    print(f"  - analysis/outputs/csv/coaching_war_quintile_analysis.csv")
    print(f"  - analysis/outputs/png/coaching_war_predictive_validity.png")
    print(f"  - analysis/outputs/png/coaching_war_quintile_analysis.png")

if __name__ == '__main__':
    main()
