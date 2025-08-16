import pandas as pd
from pathlib import Path

# Load Andy Reid's data
history_df = pd.read_csv("C:/Personal/Coach_WAR/data/raw/Coaches/Andy Reid/all_coaching_history.csv")
ranks_df = pd.read_csv("C:/Personal/Coach_WAR/data/raw/Coaches/Andy Reid/all_coaching_ranks.csv")

print("Andy Reid History (selected years):")
andy_oc = history_df[(history_df['Year'] >= 1997) & (history_df['Year'] <= 1998)]
print(andy_oc[['Year', 'Level', 'Role', 'Employer']])

print("\nAndy Reid Ranks (selected years):")
andy_ranks = ranks_df[(ranks_df['Year'] >= 1997) & (ranks_df['Year'] <= 1998)]
print(andy_ranks[['Year', 'Tm', 'Role']])

# Check what team PHI had in 1997
league_dir = Path("C:/Personal/Coach_WAR/data/processed/League Data/1997")
if league_dir.exists():
    team_file = league_dir / "league_team_data_normalized.csv"
    if team_file.exists():
        team_df = pd.read_csv(team_file)
        phi_row = team_df[team_df['Team Abbreviation'] == 'phi']
        if not phi_row.empty:
            print(f"\nPHI data exists in 1997: {not phi_row.empty}")
            print(f"Sample PHI values in 1997:")
            for col in ['PF (Points For)', 'Yds', 'Y/P']:
                if col in phi_row.columns:
                    print(f"  {col}: {phi_row[col].iloc[0]}")
        else:
            print("\nNo PHI data found in 1997")
            print(f"Available teams: {sorted(team_df['Team Abbreviation'].unique())}")