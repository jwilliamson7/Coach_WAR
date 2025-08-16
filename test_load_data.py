import pandas as pd
from pathlib import Path

# Test loading normalized data
year_dir = Path("C:/Personal/Coach_WAR/data/processed/League Data/1970")
team_file = year_dir / "league_team_data_normalized.csv"
team_df = pd.read_csv(team_file)

print("Columns in normalized data:")
for col in team_df.columns[:20]:
    print(f"  {col}")

# Test team lookup
team = "chi"
team_row = team_df[team_df['Team Abbreviation'] == team]
print(f"\nLooking for team '{team}': Found = {not team_row.empty}")

if not team_row.empty:
    print("\nSample values for CHI in 1970:")
    for col in ["PF (Points For)", "Yds", "Y/P"]:
        if col in team_row.columns:
            value = team_row[col].iloc[0]
            print(f"  {col}: {value}")
            
# Check if BASE_TEAM_STATISTICS are present
from crawlers.utils.data_constants import BASE_TEAM_STATISTICS

print("\nChecking BASE_TEAM_STATISTICS presence:")
for stat in BASE_TEAM_STATISTICS[:5]:
    present = stat in team_df.columns
    print(f"  '{stat}': {present}")