"""
Coach Background WAR Analysis
Identifies coach backgrounds (offensive coordinator, defensive coordinator, or other)
and analyzes average cumulative WAR trajectories by background type.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def identify_coach_background(impact_df):
    """Identify each coach's background based on available metrics."""
    
    # Get unique coaches
    coaches = impact_df['Primary_Coach'].unique()
    
    coach_backgrounds = []
    
    for coach in coaches:
        coach_data = impact_df[impact_df['Primary_Coach'] == coach]
        
        # Check for OC metrics (offensive coordinator background)
        oc_columns = [col for col in coach_data.columns if '__oc_Norm' in col]
        oc_seasons = coach_data[oc_columns].notna().any(axis=1).sum() if oc_columns else 0
        
        # Check for DC metrics (defensive coordinator background)  
        dc_columns = [col for col in coach_data.columns if '__dc_Norm' in col]
        dc_seasons = coach_data[dc_columns].notna().any(axis=1).sum() if dc_columns else 0
        
        total_seasons = len(coach_data)
        
        # Calculate the predominant background based on where they have more coordinator data
        # Require at least 25% of seasons to have coordinator data to be classified as that background
        min_seasons_threshold = max(1, total_seasons * 0.25)
        
        # Classify background based on predominant coordinator experience
        if oc_seasons >= min_seasons_threshold and dc_seasons >= min_seasons_threshold:
            # If both are significant, classify by which is more frequent
            if oc_seasons > dc_seasons:
                background = 'Offensive'
            elif dc_seasons > oc_seasons:
                background = 'Defensive'
            else:
                background = 'Both'  # True tie - very rare
        elif oc_seasons >= min_seasons_threshold:
            background = 'Offensive'
        elif dc_seasons >= min_seasons_threshold:
            background = 'Defensive'
        else:
            background = 'Other'
            
        coach_backgrounds.append({
            'Coach': coach,
            'Background': background,
            'Has_OC_Data': has_oc_data,
            'Has_DC_Data': has_dc_data,
            'Total_Seasons': len(coach_data)
        })
    
    return pd.DataFrame(coach_backgrounds)

def calculate_cumulative_war_by_background(impact_df, coach_backgrounds):
    """Calculate average cumulative WAR trajectories by coach background."""
    
    # Merge background data
    analysis_df = impact_df.merge(coach_backgrounds[['Coach', 'Background']], 
                                left_on='Primary_Coach', right_on='Coach', how='left')
    
    # Calculate cumulative WAR for each coach
    coach_trajectories = []
    
    for coach in analysis_df['Primary_Coach'].unique():
        coach_data = analysis_df[analysis_df['Primary_Coach'] == coach].sort_values('Year')
        if len(coach_data) == 0:
            continue
            
        # Calculate cumulative WAR in games
        coach_data = coach_data.copy()
        coach_data['Cumulative_WAR_Games'] = (coach_data['Coaching_WAR'].cumsum() * 16)
        coach_data['Season_Number'] = range(1, len(coach_data) + 1)
        
        background = coach_data['Background'].iloc[0]
        
        for _, row in coach_data.iterrows():
            coach_trajectories.append({
                'Coach': coach,
                'Background': background,
                'Season_Number': row['Season_Number'],
                'Cumulative_WAR_Games': row['Cumulative_WAR_Games'],
                'Single_Season_WAR': row['Coaching_WAR'] * 16
            })
    
    return pd.DataFrame(coach_trajectories)

def create_background_war_analysis():
    """Create comprehensive analysis of coach backgrounds and WAR performance."""
    
    # Load the combined dataset which has coordinator metrics
    print("Loading combined final dataset...")
    combined_df = pd.read_csv('data/final/combined_final_dataset.csv')
    
    # Load the coaching impact analysis data
    print("Loading coaching impact analysis data...")
    impact_df = pd.read_csv('data/final/coaching_impact_analysis.csv')
    
    # Merge datasets to get coordinator metrics with coaching WAR
    impact_df = impact_df.merge(combined_df[['Team', 'Year'] + [col for col in combined_df.columns if '__oc_Norm' in col or '__dc_Norm' in col]], 
                               on=['Team', 'Year'], how='left')
    
    print(f"Loaded data for {len(impact_df)} coach-season records")
    print(f"Unique coaches: {impact_df['Primary_Coach'].nunique()}")
    
    # Identify coach backgrounds
    print("\nIdentifying coach backgrounds...")
    coach_backgrounds = identify_coach_background(impact_df)
    
    # Print background distribution
    background_counts = coach_backgrounds['Background'].value_counts()
    print(f"\nCoach Background Distribution:")
    print("=" * 40)
    for background, count in background_counts.items():
        pct = (count / len(coach_backgrounds)) * 100
        print(f"{background:<12}: {count:3d} coaches ({pct:5.1f}%)")
    
    # Calculate trajectories
    print("\nCalculating cumulative WAR trajectories...")
    trajectories_df = calculate_cumulative_war_by_background(impact_df, coach_backgrounds)
    
    # Calculate average trajectories by background and season
    avg_trajectories = trajectories_df.groupby(['Background', 'Season_Number']).agg({
        'Cumulative_WAR_Games': ['mean', 'std', 'count'],
        'Single_Season_WAR': 'mean'
    }).round(2)
    
    # Flatten column names
    avg_trajectories.columns = ['Avg_Cumulative_WAR', 'Std_Cumulative_WAR', 'Coach_Count', 'Avg_Single_Season_WAR']
    avg_trajectories = avg_trajectories.reset_index()
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Define colors for backgrounds
    colors = {
        'Offensive': '#FF6B35',  # Orange
        'Defensive': '#004E89',  # Dark Blue  
        'Both': '#7209B7',       # Purple
        'Other': '#808080'       # Gray
    }
    
    # Plot 1: Average Cumulative WAR Trajectories
    for background in avg_trajectories['Background'].unique():
        if pd.isna(background):
            continue
            
        bg_data = avg_trajectories[avg_trajectories['Background'] == background]
        
        ax1.plot(bg_data['Season_Number'], bg_data['Avg_Cumulative_WAR'], 
                color=colors.get(background, '#000000'), linewidth=3, 
                label=f"{background} ({background_counts.get(background, 0)} coaches)",
                marker='o', markersize=6, alpha=0.8)
        
        # Add confidence bands (±1 std)
        if len(bg_data) > 1:
            ax1.fill_between(bg_data['Season_Number'], 
                           bg_data['Avg_Cumulative_WAR'] - bg_data['Std_Cumulative_WAR'],
                           bg_data['Avg_Cumulative_WAR'] + bg_data['Std_Cumulative_WAR'],
                           color=colors.get(background, '#000000'), alpha=0.2)
    
    # Customize plot 1
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Season Number in Career', fontsize=16, family='Cambria')
    ax1.set_ylabel('Average Cumulative WAR (Wins Above Replacement)', fontsize=16, family='Cambria')
    ax1.set_title('Average Cumulative WAR Trajectories by Coach Background', fontsize=18, fontweight='bold', family='Cambria')
    ax1.grid(True, alpha=0.3)
    legend1 = ax1.legend(fontsize=12, loc='upper left')
    for text in legend1.get_texts():
        text.set_family('Cambria')
    ax1.tick_params(axis='both', labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_family('Cambria')
    
    # Plot 2: Average Single Season WAR by Background and Experience
    season_ranges = [(1, 3), (4, 7), (8, 15), (16, 30)]
    range_labels = ['1-3 seasons', '4-7 seasons', '8-15 seasons', '16+ seasons']
    
    background_performance = []
    
    for background in avg_trajectories['Background'].unique():
        if pd.isna(background):
            continue
            
        bg_traj = trajectories_df[trajectories_df['Background'] == background]
        
        for i, (min_season, max_season) in enumerate(season_ranges):
            range_data = bg_traj[(bg_traj['Season_Number'] >= min_season) & 
                               (bg_traj['Season_Number'] <= max_season)]
            
            if len(range_data) > 0:
                avg_war = range_data['Single_Season_WAR'].mean()
                background_performance.append({
                    'Background': background,
                    'Experience_Range': range_labels[i],
                    'Avg_Single_Season_WAR': avg_war,
                    'Sample_Size': len(range_data)
                })
    
    perf_df = pd.DataFrame(background_performance)
    
    # Create grouped bar chart
    x_pos = np.arange(len(range_labels))
    bar_width = 0.2
    
    for i, background in enumerate(['Offensive', 'Defensive', 'Both', 'Other']):
        if background not in perf_df['Background'].values:
            continue
            
        bg_perf = perf_df[perf_df['Background'] == background]
        values = []
        
        for range_label in range_labels:
            match = bg_perf[bg_perf['Experience_Range'] == range_label]
            if len(match) > 0:
                values.append(match['Avg_Single_Season_WAR'].iloc[0])
            else:
                values.append(0)
        
        ax2.bar(x_pos + i * bar_width, values, bar_width, 
               color=colors.get(background, '#000000'), alpha=0.8,
               label=background, edgecolor='black', linewidth=0.5)
    
    # Customize plot 2
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Experience Level', fontsize=16, family='Cambria')
    ax2.set_ylabel('Average Single Season WAR (Wins Above Replacement)', fontsize=16, family='Cambria')
    ax2.set_title('Average Single Season WAR by Coach Background and Experience', fontsize=18, fontweight='bold', family='Cambria')
    ax2.set_xticks(x_pos + bar_width * 1.5)
    ax2.set_xticklabels(range_labels, fontsize=12, family='Cambria')
    ax2.grid(True, alpha=0.3, axis='y')
    legend2 = ax2.legend(fontsize=12)
    for text in legend2.get_texts():
        text.set_family('Cambria')
    ax2.tick_params(axis='both', labelsize=12)
    for label in ax2.get_yticklabels():
        label.set_family('Cambria')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'analysis/outputs/png/coach_background_war_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nPlot saved as: {output_file}")
    
    # Print detailed statistics
    print(f"\nDetailed Background Analysis:")
    print("=" * 60)
    
    for background in ['Offensive', 'Defensive', 'Both', 'Other']:
        if background not in coach_backgrounds['Background'].values:
            continue
            
        bg_coaches = coach_backgrounds[coach_backgrounds['Background'] == background]
        bg_traj = trajectories_df[trajectories_df['Background'] == background]
        
        if len(bg_traj) > 0:
            avg_career_war = bg_traj.groupby('Coach')['Cumulative_WAR_Games'].max().mean()
            avg_single_war = bg_traj['Single_Season_WAR'].mean()
            
            print(f"\n{background} Coaches ({len(bg_coaches)} total):")
            print(f"  Average career WAR: {avg_career_war:+.1f} games")
            print(f"  Average single season WAR: {avg_single_war:+.2f} games")
            
            # Top 3 coaches by career WAR
            career_wars = bg_traj.groupby('Coach')['Cumulative_WAR_Games'].max().sort_values(ascending=False)
            print(f"  Top 3 by career WAR:")
            for coach, war in career_wars.head(3).items():
                print(f"    {coach}: {war:+.1f} games")
    
    # Save detailed data
    coach_backgrounds.to_csv('analysis/outputs/csv/coach_backgrounds.csv', index=False)
    avg_trajectories.to_csv('analysis/outputs/csv/coach_background_trajectories.csv', index=False)
    perf_df.to_csv('analysis/outputs/csv/coach_background_performance.csv', index=False)
    
    print(f"\nDetailed data saved:")
    print("  - coach_backgrounds.csv: Coach background classifications")
    print("  - coach_background_trajectories.csv: Average trajectories by background")
    print("  - coach_background_performance.csv: Performance by experience level")
    
    plt.close()
    
    return coach_backgrounds, avg_trajectories, perf_df

if __name__ == "__main__":
    print("Analyzing coach backgrounds and WAR performance...")
    backgrounds, trajectories, performance = create_background_war_analysis()
    print(f"\nAnalysis complete! Check the PNG file and CSV outputs for detailed results.")