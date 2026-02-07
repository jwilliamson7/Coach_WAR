"""
Coach Background Performance by Decade
Analyzes how offensive vs defensive coaching background (coordinators and position coaches) 
success has changed over different NFL eras.
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_coach_background_by_decade():
    """Create decade-based comparison table for coach backgrounds."""
    
    # Load the matched data from the previous analysis
    print("Loading coach background and WAR data...")
    try:
        matched_data = pd.read_csv('analysis/outputs/csv/coach_matched_war_background_data.csv')
        # Remove duplicates
        matched_data = matched_data.drop_duplicates(subset=['Team', 'Year', 'Primary_Coach'])
        # Filter out "Both" category for cleaner analysis
        matched_data = matched_data[matched_data['Background'] != 'Both']
        print(f"Loaded {len(matched_data)} coach-season records (after deduplication and filtering)")
    except FileNotFoundError:
        print("Error: Please run coach_background_from_history.py first to generate the matched data.")
        return None
    
    # Add decade classification based on year
    def classify_decade(year):
        if year < 1980:
            return "1970s"
        elif year < 1990:
            return "1980s"
        elif year < 2000:
            return "1990s"
        elif year < 2010:
            return "2000s"
        elif year < 2020:
            return "2010s"
        else:
            return "2020s"
    
    matched_data['Decade'] = matched_data['Year'].apply(classify_decade)
    
    # Convert WAR to games
    matched_data['WAR_Games'] = matched_data['Coaching_WAR'] * 16
    
    # Create summary statistics by background and decade
    summary_stats = []
    
    decades = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
    backgrounds = ['Offensive', 'Defensive', 'Other']
    
    for decade in decades:
        decade_data = matched_data[matched_data['Decade'] == decade]
        
        print(f"\n{decade} Analysis:")
        print("=" * 40)
        
        for background in backgrounds:
            bg_data = decade_data[decade_data['Background'] == background]
            
            if len(bg_data) > 0:
                unique_coaches = len(bg_data['Primary_Coach'].unique())
                total_seasons = len(bg_data)
                avg_war = bg_data['WAR_Games'].mean()
                median_war = bg_data['WAR_Games'].median()
                
                # Calculate career WAR for coaches in this decade
                career_wars = bg_data.groupby('Primary_Coach')['WAR_Games'].sum()
                avg_career_war = career_wars.mean()
                
                summary_stats.append({
                    'Decade': decade,
                    'Background': background,
                    'Unique_Coaches': unique_coaches,
                    'Total_Seasons': total_seasons,
                    'Avg_Season_WAR': avg_war,
                    'Median_Season_WAR': median_war,
                    'Avg_Career_WAR': avg_career_war,
                    'Sample_Size': f"{unique_coaches} coaches, {total_seasons} seasons"
                })
                
                print(f"{background:<12}: {unique_coaches:3d} coaches, {total_seasons:4d} seasons, "
                      f"Avg WAR: {avg_war:+.2f} games/season")
            else:
                summary_stats.append({
                    'Decade': decade,
                    'Background': background,
                    'Unique_Coaches': 0,
                    'Total_Seasons': 0,
                    'Avg_Season_WAR': np.nan,
                    'Median_Season_WAR': np.nan,
                    'Avg_Career_WAR': np.nan,
                    'Sample_Size': "No data"
                })
                print(f"{background:<12}: No data")
    
    # Convert to DataFrame for easier analysis
    summary_df = pd.DataFrame(summary_stats)
    
    # Create pivot tables for easier viewing
    print("\n" + "="*80)
    print("SUMMARY TABLE: Average Season WAR by Background and Decade")
    print("="*80)
    
    # Pivot table for average season WAR
    war_pivot = summary_df.pivot(index='Background', columns='Decade', values='Avg_Season_WAR').round(2)
    print(war_pivot.to_string(na_rep='--'))
    
    print("\n" + "="*80)
    print("COACH COUNT by Background and Decade")
    print("="*80)
    
    # Pivot table for coach counts
    count_pivot = summary_df.pivot(index='Background', columns='Decade', values='Unique_Coaches').fillna(0).astype(int)
    print(count_pivot.to_string())
    
    print("\n" + "="*80)
    print("SEASON COUNT by Background and Decade")
    print("="*80)
    
    # Pivot table for season counts
    season_pivot = summary_df.pivot(index='Background', columns='Decade', values='Total_Seasons').fillna(0).astype(int)
    print(season_pivot.to_string())
    
    # Calculate decade trends
    print("\n" + "="*80)
    print("DECADE TRENDS")
    print("="*80)
    
    for background in backgrounds:
        bg_summary = summary_df[summary_df['Background'] == background]
        bg_summary = bg_summary[bg_summary['Avg_Season_WAR'].notna()]  # Remove decades with no data
        
        if len(bg_summary) > 1:
            first_decade_war = bg_summary.iloc[0]['Avg_Season_WAR']
            last_decade_war = bg_summary.iloc[-1]['Avg_Season_WAR']
            trend = "improving" if last_decade_war > first_decade_war else "declining"
            
            print(f"{background} Coordinators: {first_decade_war:+.2f} to {last_decade_war:+.2f} games/season ({trend})")
        else:
            print(f"{background} Coordinators: Insufficient data for trend analysis")
    
    # Background hiring trends
    print("\n" + "="*80)
    print("HIRING TRENDS (% of coaches by background)")
    print("="*80)
    
    hiring_trends = []
    for decade in decades:
        decade_data = summary_df[summary_df['Decade'] == decade]
        total_coaches = decade_data['Unique_Coaches'].sum()
        
        if total_coaches > 0:
            for background in backgrounds:
                bg_count = decade_data[decade_data['Background'] == background]['Unique_Coaches'].iloc[0]
                percentage = (bg_count / total_coaches) * 100
                hiring_trends.append({
                    'Decade': decade,
                    'Background': background,
                    'Percentage': percentage
                })
    
    hiring_df = pd.DataFrame(hiring_trends)
    hiring_pivot = hiring_df.pivot(index='Background', columns='Decade', values='Percentage').round(1)
    print(hiring_pivot.to_string(na_rep='--') + '%')
    
    # Statistical comparison section
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: OFFENSIVE vs DEFENSIVE BY DECADE")
    print("="*80)

    # Store results for trend analysis and compact table
    decade_differences = []
    compact_table_data = {
        'coaches_off': [],
        'coaches_def': [],
        'seasons_off': [],
        'seasons_def': [],
        'mean_off': [],
        'mean_def': [],
        'difference': [],
        'diff_ci_lower': [],
        'diff_ci_upper': [],
        'u_stat': [],
        'p_value': [],
        'significance': []
    }
    
    for decade in decades:
        decade_data = matched_data[matched_data['Decade'] == decade]
        offensive_data = decade_data[decade_data['Background'] == 'Offensive']['WAR_Games']
        defensive_data = decade_data[decade_data['Background'] == 'Defensive']['WAR_Games']
        
        if len(offensive_data) > 0 and len(defensive_data) > 0:
            print(f"\n{decade}:")
            print("-" * 40)
            
            # Sample sizes
            off_coaches = len(decade_data[decade_data['Background'] == 'Offensive']['Primary_Coach'].unique())
            def_coaches = len(decade_data[decade_data['Background'] == 'Defensive']['Primary_Coach'].unique())
            
            print(f"Sample sizes:")
            print(f"  Offensive: {off_coaches} coaches, {len(offensive_data)} seasons")
            print(f"  Defensive: {def_coaches} coaches, {len(defensive_data)} seasons")
            
            # Descriptive stats
            print(f"\nMean WAR (games/season):")
            print(f"  Offensive: {offensive_data.mean():+.3f}")
            print(f"  Defensive: {defensive_data.mean():+.3f}")
            print(f"  Difference: {offensive_data.mean() - defensive_data.mean():+.3f}")
            
            # T-test
            t_stat, p_value = stats.ttest_ind(offensive_data, defensive_data, equal_var=False)

            # Mann-Whitney U test (non-parametric)
            u_stat, p_value_mw = stats.mannwhitneyu(offensive_data, defensive_data, alternative='two-sided')

            # Cohen's d
            pooled_std = np.sqrt(((len(offensive_data) - 1) * offensive_data.var() +
                                 (len(defensive_data) - 1) * defensive_data.var()) /
                                (len(offensive_data) + len(defensive_data) - 2))
            cohens_d = (offensive_data.mean() - defensive_data.mean()) / pooled_std if pooled_std > 0 else 0

            # Welch's t-interval for difference in means
            se_diff = np.sqrt(offensive_data.var() / len(offensive_data) + defensive_data.var() / len(defensive_data))
            # Welch-Satterthwaite degrees of freedom
            nu_num = (offensive_data.var() / len(offensive_data) + defensive_data.var() / len(defensive_data))**2
            nu_den = ((offensive_data.var() / len(offensive_data))**2 / (len(offensive_data) - 1) +
                      (defensive_data.var() / len(defensive_data))**2 / (len(defensive_data) - 1))
            df_welch = nu_num / nu_den if nu_den > 0 else min(len(offensive_data), len(defensive_data)) - 1
            t_crit_welch = stats.t.ppf(0.975, df=df_welch)
            diff_mean = offensive_data.mean() - defensive_data.mean()
            diff_ci_lower = diff_mean - t_crit_welch * se_diff
            diff_ci_upper = diff_mean + t_crit_welch * se_diff

            print(f"\nStatistical tests:")
            print(f"  Welch's t-test:")
            print(f"    t-statistic: {t_stat:.3f}")
            print(f"    p-value: {p_value:.4f}")
            print(f"  Mann-Whitney U test (non-parametric):")
            print(f"    U-statistic: {u_stat:.0f}")
            print(f"    p-value: {p_value_mw:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            print(f"  95% CI on difference (O-D): [{diff_ci_lower:+.3f}, {diff_ci_upper:+.3f}]")

            # Interpretation based on more conservative p-value
            p_conservative = max(p_value, p_value_mw)
            if p_conservative < 0.01:
                sig = "**"
                sig_text = "highly significant"
            elif p_conservative < 0.05:
                sig = "*"
                sig_text = "significant"
            elif p_conservative < 0.10:
                sig = "†"
                sig_text = "marginally significant"
            else:
                sig = ""
                sig_text = "not significant"

            print(f"  Result: {sig_text} {sig}")
            
            # Store for trend analysis
            decade_num = int(decade[:4])  # Get year from decade string
            decade_differences.append({
                'Decade': decade,
                'Year': decade_num,
                'Difference': offensive_data.mean() - defensive_data.mean(),
                'Diff_CI_Lower': diff_ci_lower,
                'Diff_CI_Upper': diff_ci_upper,
                'p_value_ttest': p_value,
                'p_value_mw': p_value_mw,
                'cohens_d': cohens_d,
                'n_offensive': len(offensive_data),
                'n_defensive': len(defensive_data)
            })

            # Store for compact table
            compact_table_data['coaches_off'].append(off_coaches)
            compact_table_data['coaches_def'].append(def_coaches)
            compact_table_data['seasons_off'].append(len(offensive_data))
            compact_table_data['seasons_def'].append(len(defensive_data))
            compact_table_data['mean_off'].append(offensive_data.mean())
            compact_table_data['mean_def'].append(defensive_data.mean())
            compact_table_data['difference'].append(offensive_data.mean() - defensive_data.mean())
            compact_table_data['diff_ci_lower'].append(diff_ci_lower)
            compact_table_data['diff_ci_upper'].append(diff_ci_upper)
            compact_table_data['u_stat'].append(int(u_stat))
            compact_table_data['p_value'].append(p_value_mw)

            # Determine significance
            if p_value_mw < 0.01:
                compact_table_data['significance'].append('Highly Significant')
            elif p_value_mw < 0.05:
                compact_table_data['significance'].append('Significant')
            elif p_value_mw < 0.10:
                compact_table_data['significance'].append('Marginally Significant')
            else:
                compact_table_data['significance'].append('-')
    
    # Print compact summary table
    print("\n" + "="*80)
    print("COMPACT SUMMARY TABLE: OFFENSIVE vs DEFENSIVE BY DECADE")
    print("="*80)
    print()

    # Create formatted compact table with proper column alignment
    decades_abbr = ['1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

    # Define column width for each decade
    col_width = 20

    # Header row with decade labels
    header = f"{'Background':<31}"
    for decade in ["1970's", "1980's", "1990's", "2000's", "2010's", "2020's"]:
        header += f"{decade:>{col_width}}"
    print(header)

    # Row 1: Sample Size, Coaches (O/D)
    row = f"{'Sample Size, Coaches (O/D)':<31}"
    for i in range(len(compact_table_data['coaches_off'])):
        val = f"{compact_table_data['coaches_off'][i]}/{compact_table_data['coaches_def'][i]}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 2: Sample Size, Seasons (O/D)
    row = f"{'Sample Size, Seasons (O/D)':<31}"
    for i in range(len(compact_table_data['seasons_off'])):
        val = f"{compact_table_data['seasons_off'][i]}/{compact_table_data['seasons_def'][i]}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 3: Mean WAR (O/D)
    row = f"{'Mean WAR (O/D)':<31}"
    for i in range(len(compact_table_data['mean_off'])):
        val = f"{compact_table_data['mean_off'][i]:+.3f} / {compact_table_data['mean_def'][i]:+.3f}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 4: Difference (O - D)
    row = f"{'Difference (O - D)':<31}"
    for i in range(len(compact_table_data['difference'])):
        val = f"{compact_table_data['difference'][i]:+.3f}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 4b: 95% CI on Difference
    row = f"{'95% CI (O - D)':<31}"
    for i in range(len(compact_table_data['diff_ci_lower'])):
        val = f"[{compact_table_data['diff_ci_lower'][i]:+.2f},{compact_table_data['diff_ci_upper'][i]:+.2f}]"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 5: U-statistic
    row = f"{'U-statistic':<31}"
    for i in range(len(compact_table_data['u_stat'])):
        val = f"{compact_table_data['u_stat'][i]}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 6: p-value
    row = f"{'p-value':<31}"
    for i in range(len(compact_table_data['p_value'])):
        val = f"{compact_table_data['p_value'][i]:.4f}"
        row += f"{val:>{col_width}}"
    print(row)

    # Row 7: Significance
    row = f"{'Significance':<31}"
    for i in range(len(compact_table_data['significance'])):
        val = f"{compact_table_data['significance'][i]}"
        row += f"{val:>{col_width}}"
    print(row)
    print()

    # 2020s projection analysis
    print("\n" + "="*80)
    print("2020s PROJECTION: FULL DECADE ANALYSIS")
    print("="*80)
    print()

    # Get 2020s data
    data_2020s = matched_data[matched_data['Decade'] == '2020s'].copy()
    offensive_2020s = data_2020s[data_2020s['Background'] == 'Offensive']['WAR_Games']
    defensive_2020s = data_2020s[data_2020s['Background'] == 'Defensive']['WAR_Games']

    print(f"Current 2020s data (2020-2024, 5 seasons):")
    print(f"  Offensive: {len(offensive_2020s)} season observations")
    print(f"  Defensive: {len(defensive_2020s)} season observations")
    print(f"  Mean WAR (O/D): {offensive_2020s.mean():+.3f} / {defensive_2020s.mean():+.3f}")
    print(f"  Difference: {offensive_2020s.mean() - defensive_2020s.mean():+.3f}")

    # Current test
    u_stat_current, p_value_current = stats.mannwhitneyu(offensive_2020s, defensive_2020s, alternative='two-sided')
    print(f"  Current U-statistic: {u_stat_current:.0f}")
    print(f"  Current p-value: {p_value_current:.4f}")

    print(f"\nProjected full decade (2020-2029, 10 seasons):")
    print(f"Assuming similar hiring patterns and performance distributions continue...")

    # Simply duplicate the existing data to simulate a full decade
    # Current: 5 years, Target: 10 years (double the samples)
    offensive_2020s_full = np.concatenate([offensive_2020s, offensive_2020s])
    defensive_2020s_full = np.concatenate([defensive_2020s, defensive_2020s])

    print(f"  Projected sample sizes: {len(offensive_2020s_full)} offensive, {len(defensive_2020s_full)} defensive seasons")
    print(f"  Mean WAR (O/D): {offensive_2020s_full.mean():+.3f} / {defensive_2020s_full.mean():+.3f}")
    print(f"  Difference: {offensive_2020s_full.mean() - defensive_2020s_full.mean():+.3f}")

    # Calculate projected statistics
    u_stat_projected, p_value_projected = stats.mannwhitneyu(offensive_2020s_full, defensive_2020s_full, alternative='two-sided')

    print(f"  Projected U-statistic: {u_stat_projected:.0f}")
    print(f"  Projected p-value: {p_value_projected:.4f}")

    if p_value_projected < 0.01:
        projection_result = "highly significant"
    elif p_value_projected < 0.05:
        projection_result = "significant"
    elif p_value_projected < 0.10:
        projection_result = "marginally significant"
    else:
        projection_result = "not significant"

    print(f"  Projected result: {projection_result}")
    print(f"\nNote: This projection assumes the current performance trends and hiring patterns")
    print(f"      continue through 2029 at the same rate as 2020-2024.")

    # Trend analysis over time
    print("\n" + "="*80)
    print("TREND ANALYSIS: CHANGE OVER TIME")
    print("="*80)
    
    if len(decade_differences) > 1:
        trend_df = pd.DataFrame(decade_differences)
        
        # Linear regression on the difference over time
        from scipy.stats import linregress
        
        # Use decade number for regression
        slope, intercept, r_value, p_value_trend, std_err = linregress(
            trend_df['Year'], 
            trend_df['Difference']
        )
        
        print(f"\nLinear trend in Offensive-Defensive difference:")
        print(f"  Slope: {slope:.4f} games per year")
        print(f"  Per decade: {slope * 10:.3f} games")
        print(f"  R²: {r_value**2:.3f}")
        print(f"  p-value: {p_value_trend:.4f}")
        
        if p_value_trend < 0.05:
            print(f"  Significant linear trend detected!")
            if slope > 0:
                print(f"  Offensive coaches improving relative to defensive over time")
            else:
                print(f"  Defensive coaches improving relative to offensive over time")
        else:
            print(f"  No significant linear trend")
        
        # Display the decade-by-decade differences
        print(f"\nDecade-by-decade differences (Offensive - Defensive):")
        print(f"{'Decade':<8} {'Difference':<12} {'t-test p':<10} {'M-W p':<10} {'Significance'}")
        print("-" * 60)

        for _, row in trend_df.iterrows():
            p_conservative = max(row['p_value_ttest'], row['p_value_mw'])
            if p_conservative < 0.01:
                sig = "**"
            elif p_conservative < 0.05:
                sig = "*"
            elif p_conservative < 0.10:
                sig = "†"
            else:
                sig = ""

            print(f"{row['Decade']:<8} {row['Difference']:+.3f} games  {row['p_value_ttest']:.4f}     {row['p_value_mw']:.4f}     {sig}")
        
        print(f"\nSignificance: ** p<0.01, * p<0.05, † p<0.10")
        
        # Save trend data
        trend_df.to_csv('analysis/outputs/csv/coach_background_decade_trend_analysis.csv', index=False)
        print(f"\nTrend analysis saved to: coach_background_decade_trend_analysis.csv")

    # Within-group era change analysis (1970s vs 2020s)
    print("\n" + "="*80)
    print("ERA CHANGE ANALYSIS: 1970s vs 2020s WITHIN EACH BACKGROUND")
    print("="*80)

    for background in ['Offensive', 'Defensive']:
        data_70s = matched_data[(matched_data['Decade'] == '1970s') & (matched_data['Background'] == background)]['WAR_Games']
        data_20s = matched_data[(matched_data['Decade'] == '2020s') & (matched_data['Background'] == background)]['WAR_Games']

        if len(data_70s) > 1 and len(data_20s) > 1:
            change = data_20s.mean() - data_70s.mean()
            se = np.sqrt(data_70s.var() / len(data_70s) + data_20s.var() / len(data_20s))
            nu_num = (data_70s.var() / len(data_70s) + data_20s.var() / len(data_20s))**2
            nu_den = ((data_70s.var() / len(data_70s))**2 / (len(data_70s) - 1) +
                      (data_20s.var() / len(data_20s))**2 / (len(data_20s) - 1))
            df_w = nu_num / nu_den if nu_den > 0 else min(len(data_70s), len(data_20s)) - 1
            t_crit = stats.t.ppf(0.975, df=df_w)
            ci_lo = change - t_crit * se
            ci_hi = change + t_crit * se
            t_stat_era, p_era = stats.ttest_ind(data_20s, data_70s, equal_var=False)

            print(f"\n{background} coaches:")
            print(f"  1970s: n={len(data_70s)} seasons, mean={data_70s.mean():+.3f}, sd={data_70s.std():+.3f}")
            print(f"  2020s: n={len(data_20s)} seasons, mean={data_20s.mean():+.3f}, sd={data_20s.std():+.3f}")
            print(f"  Change (2020s - 1970s): {change:+.3f} games/season")
            print(f"  95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
            print(f"  Welch's t = {t_stat_era:.3f}, df = {df_w:.1f}, p = {p_era:.4f}")

    # Save detailed results
    summary_df.to_csv('analysis/outputs/csv/coach_background_by_decade_summary.csv', index=False)
    war_pivot.to_csv('analysis/outputs/csv/coach_background_war_by_decade.csv')
    count_pivot.to_csv('analysis/outputs/csv/coach_background_counts_by_decade.csv')
    hiring_pivot.to_csv('analysis/outputs/csv/coach_background_hiring_trends_by_decade.csv')
    
    print(f"\nDetailed data saved:")
    print("  - coach_background_by_decade_summary.csv: Complete summary statistics")
    print("  - coach_background_war_by_decade.csv: WAR performance by decade")
    print("  - coach_background_counts_by_decade.csv: Coach counts by decade")
    print("  - coach_background_hiring_trends_by_decade.csv: Hiring percentages by decade")
    
    return summary_df, war_pivot, count_pivot, hiring_pivot

if __name__ == "__main__":
    print("Analyzing coach background performance by decade...")
    summary, war_table, count_table, hiring_table = analyze_coach_background_by_decade()
    print(f"\nAnalysis complete! Check the CSV outputs for detailed decade-by-decade breakdowns.")