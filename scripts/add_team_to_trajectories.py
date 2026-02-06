"""
Add Team column to coach_war_trajectories.csv by merging with coaching_impact_analysis.csv
"""

import pandas as pd

def main():
    print("Loading data...")

    # Load both files
    trajectories = pd.read_csv('data/final/coach_war_trajectories.csv')
    impact = pd.read_csv('data/final/coaching_impact_analysis.csv')

    print(f"Trajectories: {len(trajectories)} rows")
    print(f"Impact analysis: {len(impact)} rows")

    # Get Team mapping from impact analysis
    team_mapping = impact[['Team', 'Year', 'Primary_Coach']].copy()
    team_mapping = team_mapping.rename(columns={'Primary_Coach': 'Coach'})

    # Merge to add Team column
    merged = trajectories.merge(team_mapping, on=['Coach', 'Year'], how='left')

    # Reorder columns to put Team near the beginning
    cols = ['Coach', 'Team', 'Year', 'Season_Number', 'Annual_WAR', 'Annual_Games',
            'Cumulative_WAR', 'Cumulative_Games', 'Win_Pct', 'Predicted_Impact',
            'Predicted_Replacement', 'Total_Seasons']
    merged = merged[cols]

    # Check for any missing teams
    missing = merged['Team'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} rows have missing Team values")

    # Save result
    output_file = 'data/final/coach_war_trajectories_with_team.csv'
    merged.to_csv(output_file, index=False)

    print(f"\nSaved to: {output_file}")
    print(f"Total rows: {len(merged)}")
    print(f"\nSample output:")
    print(merged.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
