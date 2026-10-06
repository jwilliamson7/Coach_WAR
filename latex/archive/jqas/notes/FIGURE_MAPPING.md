# Figure Mapping Reference

This document maps each figure reference in the LaTeX document to its source and status.

## Main Body Figures

| Figure # | Label | Description | Status | Source File |
|----------|-------|-------------|--------|-------------|
| Figure 1 | `fig:three_coach_trajectories` | Kevin O'Connell, Don Shula, Matt Eberflus WAR Trajectories | **MISSING** | Need to create from `coach_war_trajectories.csv` |
| Figure 2 | `fig:career_distributions` | Coach WAR Career Distributions (scatter) | **MISSING** | Need to create from career summary data |
| Figure 3 | `fig:2024_coaches_avg` | 2024 Head Coach Avg WAR vs Career Length | ✓ COMPLETE | `coach_2024_matrix.png` |
| Figure 4 | `fig:2024_trajectories` | Coach WAR Trajectory for 2024 Head Coaches | ✓ COMPLETE | `coach_2024_trajectories.png` |
| Figure 5 | `fig:2024_single_year` | 2024 Coach WAR for 2024 Head Coaches | ✓ COMPLETE | `coach_2024_single_year_bar.png` |
| Figure 6 | `fig:background_trajectory` | Average Cumulative Coach WAR by Background | ✓ COMPLETE | `coach_background_from_history_15seasons.png` |
| Figure 7 | `fig:war_persistence` | Coach WAR in Year N+1 vs Year N (persistence) | ✓ COMPLETE | `coaching_war_persistence_scatter.png` |
| Figure 8 | `fig:war_quintiles` | Mean Coach WAR in Year N and N+1 by Quintile | ✓ COMPLETE | `coaching_regression_to_mean_survivorship_adjusted.png` |
| Figure 9 | `fig:war_changes` | Year-over-year change in Coach WAR by Quintile | ✓ COMPLETE | `coaching_survivorship_bias_magnitude.png` |
| Figure 10 | `fig:dashboard` | Coach WAR Exploration Dashboard | **PLACEHOLDER** | Screenshot needed from HTML dashboard |

## Appendix Figures

| Figure # | Label | Description | Status | Source File |
|----------|-------|-------------|--------|-------------|
| Figure F1 | `fig:winpct_persistence` | Coach Win Percentage in Year N+1 vs Year N | ✓ COMPLETE | `win_pct_persistence_scatter.png` |
| Figure G1 | `fig:war_persistence_background` | Coach WAR persistence by Background (single year) | ✓ COMPLETE | `coaching_war_persistence_by_background.png` |
| Figure G2 | `fig:war_multiyear_persistence` | Coach WAR persistence by Background (2-year avg) | ✓ COMPLETE | `coaching_war_multiyear_persistence_scatter.png` |

## Tables in Main Body

All tables have been successfully converted to LaTeX format:

- Table 1: Top 15 Coaches by Average WAR (`tab:top_coaches`)
- Table 2: Bottom 15 Coaches by Average WAR (`tab:bottom_coaches`)
- Table 3: Mann-Whitney U Test - Offensive vs Defensive Backgrounds (`tab:background_mann_whitney`)
- Table 4: Average Coach WAR by Background and Decade (`tab:background_by_decade`)
- Table 5: Mann-Whitney U Test by Decade (`tab:background_decade_stats`)
- Table 6: Top 10 Single-Season WAR Instances (`tab:top_single_season`)
- Table 7: Bottom 10 Single-Season WAR Instances (`tab:bottom_single_season`)

## Appendix Tables

All appendix tables have been successfully converted:

- Table A1: Feature Count Across Categories (`tab:feature_counts`)
- Table B1: Comprehensive Feature List (`tab:comprehensive_features`) - abbreviated with note
- Table C1: Selected Hyperparameters (`tab:hyperparameters`)
- Table D1: Validation Strategy Overview (`tab:validation`)
- Table E1: Feature Importance Ranking (`tab:feature_importance`)

## Summary

- **Total Figures**: 13
- **Complete**: 10 (77%)
- **Missing**: 2 (15%)
- **Placeholder**: 1 (8%)

- **Total Tables**: 12
- **Complete**: 12 (100%)

## Next Steps for Missing Figures

### 1. Three-Coach Trajectory Comparison (Figure 1)

**Python code to generate**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load trajectory data
df = pd.read_csv('../data/final/coach_war_trajectories.csv')

# Filter for three coaches
coaches = ['Kevin O\'Connell', 'Don Shula', 'Matt Eberflus']
df_filtered = df[df['Coach'].isin(coaches)]

# Plot
plt.figure(figsize=(10, 6))
for coach in coaches:
    coach_data = df_filtered[df_filtered['Coach'] == coach]
    plt.plot(coach_data['Season_Number'], coach_data['Cumulative_WAR'],
             marker='o', label=coach, linewidth=2)

plt.xlabel('Season Number', fontsize=12)
plt.ylabel('Cumulative WAR', fontsize=12)
plt.title('Coach WAR Trajectories: O\'Connell, Shula, and Eberflus', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('latex/figures/coach_trajectories_oconnell_shula_eberflus.png', dpi=300)
```

### 2. Career Distributions Scatter (Figure 2)

**Data needed**:
- X-axis: Number of seasons coached (career length)
- Y-axis: Average WAR per season
- Add median lines to create quadrants
- Color code or annotate notable coaches

Source data: `../data/final/coach_career_impact_stats.csv`

### 3. Dashboard Screenshot (Figure 10)

**Steps**:
1. Open any dashboard HTML file from `../analysis/outputs/html/`
2. Take a high-quality screenshot
3. Crop to show the key features
4. Save as `dashboard_placeholder.png` in figures directory
