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
        matched_data = pd.read_csv('analysis/coach_matched_war_background_data.csv')
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
    
    # Store results for trend analysis
    decade_differences = []
    
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
            
            # Cohen's d
            pooled_std = np.sqrt(((len(offensive_data) - 1) * offensive_data.var() + 
                                 (len(defensive_data) - 1) * defensive_data.var()) / 
                                (len(offensive_data) + len(defensive_data) - 2))
            cohens_d = (offensive_data.mean() - defensive_data.mean()) / pooled_std if pooled_std > 0 else 0
            
            print(f"\nStatistical test:")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {cohens_d:.3f}")
            
            # Interpretation
            if p_value < 0.01:
                sig = "**"
                sig_text = "highly significant"
            elif p_value < 0.05:
                sig = "*"
                sig_text = "significant"
            elif p_value < 0.10:
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
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_offensive': len(offensive_data),
                'n_defensive': len(defensive_data)
            })
    
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
        print(f"{'Decade':<8} {'Difference':<12} {'p-value':<10} {'Significance'}")
        print("-" * 50)
        
        for _, row in trend_df.iterrows():
            if row['p_value'] < 0.01:
                sig = "**"
            elif row['p_value'] < 0.05:
                sig = "*"
            elif row['p_value'] < 0.10:
                sig = "†"
            else:
                sig = ""
            
            print(f"{row['Decade']:<8} {row['Difference']:+.3f} games  {row['p_value']:.4f}     {sig}")
        
        print(f"\nSignificance: ** p<0.01, * p<0.05, † p<0.10")
        
        # Save trend data
        trend_df.to_csv('analysis/coach_background_decade_trend_analysis.csv', index=False)
        print(f"\nTrend analysis saved to: coach_background_decade_trend_analysis.csv")
    
    # Save detailed results
    summary_df.to_csv('analysis/coach_background_by_decade_summary.csv', index=False)
    war_pivot.to_csv('analysis/coach_background_war_by_decade.csv')
    count_pivot.to_csv('analysis/coach_background_counts_by_decade.csv')
    hiring_pivot.to_csv('analysis/coach_background_hiring_trends_by_decade.csv')
    
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