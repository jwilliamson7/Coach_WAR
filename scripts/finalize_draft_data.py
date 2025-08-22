import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import TEAM_FRANCHISE_MAPPINGS


class DraftDataFinalizer:
    """Creates final draft dataset with rolling averages for coaching analysis"""
    
    def __init__(self, processed_dir: str = "data/processed/Draft", 
                 output_dir: str = "data/final"):
        """Initialize finalizer with input and output directories"""
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_comprehensive_data(self) -> pd.DataFrame:
        """Load the comprehensive draft data and apply team mappings"""
        filepath = self.processed_dir / "draft_picks_by_team_round_all_years.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Comprehensive draft data not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Create a direct mapping for all teams to their current PFR abbreviations
        # This ensures we get exactly 32 teams
        # Special handling: 'hou' represents both Houston Oilers (1970-1996) and Houston Texans (2002+)
        def map_houston_team(row):
            if row['Team'] == 'hou':
                if row['Draft_Year'] <= 1996:
                    return 'oti'  # Houston Oilers -> Tennessee Titans
                else:
                    return 'htx'  # Houston Texans (2002+)
            return row['Team']
        
        # Apply Houston mapping first
        df['Team'] = df.apply(map_houston_team, axis=1)
        
        team_mapping = {
            # Historical teams that moved/changed
            'bos': 'nwe',  # Boston Patriots -> New England Patriots
            'pho': 'crd',  # Phoenix Cardinals -> Arizona Cardinals
            'ten': 'oti',  # Tennessee Oilers/Titans -> oti
            'bal': 'rav',  # Baltimore Ravens
            'ari': 'crd',  # Arizona Cardinals -> crd
            'lac': 'sdg',  # LA Chargers -> sdg
            
            # Current teams - keep as is with PFR conventions
            'atl': 'atl',  # Atlanta Falcons
            'buf': 'buf',  # Buffalo Bills
            'car': 'car',  # Carolina Panthers
            'chi': 'chi',  # Chicago Bears
            'cin': 'cin',  # Cincinnati Bengals
            'cle': 'cle',  # Cleveland Browns
            'clt': 'clt',  # Indianapolis Colts
            'crd': 'crd',  # Arizona Cardinals (already)
            'dal': 'dal',  # Dallas Cowboys
            'den': 'den',  # Denver Broncos
            'det': 'det',  # Detroit Lions
            'gnb': 'gnb',  # Green Bay Packers
            'htx': 'htx',  # Houston Texans (expansion team, keep separate)
            'jax': 'jax',  # Jacksonville Jaguars
            'kan': 'kan',  # Kansas City Chiefs
            'mia': 'mia',  # Miami Dolphins
            'min': 'min',  # Minnesota Vikings
            'nor': 'nor',  # New Orleans Saints
            'nwe': 'nwe',  # New England Patriots
            'nyg': 'nyg',  # New York Giants
            'nyj': 'nyj',  # New York Jets
            'oti': 'oti',  # Tennessee Titans (already)
            'phi': 'phi',  # Philadelphia Eagles
            'pit': 'pit',  # Pittsburgh Steelers
            'rai': 'rai',  # Raiders (Oakland/Las Vegas)
            'ram': 'ram',  # Rams (LA/St. Louis)
            'rav': 'rav',  # Baltimore Ravens (already)
            'sdg': 'sdg',  # San Diego/LA Chargers (already)
            'sea': 'sea',  # Seattle Seahawks
            'sfo': 'sfo',  # San Francisco 49ers
            'tam': 'tam',  # Tampa Bay Buccaneers
            'was': 'was',  # Washington
        }
        
        # Apply mappings
        df['Team'] = df['Team'].map(team_mapping).fillna(df['Team'])
        
        # Aggregate data for teams that have been consolidated
        # Group by Team and Draft_Year, summing pick counts
        pick_cols = [col for col in df.columns if col.endswith('_Picks')]
        agg_dict = {col: 'sum' for col in pick_cols}
        
        df = df.groupby(['Team', 'Draft_Year'], as_index=False).agg(agg_dict)
        
        # Now combine high rounds after aggregation
        df = self._combine_high_rounds(df)
        
        unique_teams = sorted(df['Team'].unique())
        print(f"After team mapping: {len(unique_teams)} unique teams")
        print(f"Teams: {unique_teams}")
        
        return df
    
    def _filter_years(self, df: pd.DataFrame, start_year: int = 2003, 
                     end_year: int = 2024) -> pd.DataFrame:
        """Filter data to specified year range"""
        return df[(df['Draft_Year'] >= start_year) & (df['Draft_Year'] <= end_year)].copy()
    
    def _get_round_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of round pick columns after combining rounds 7+"""
        round_cols = []
        for round_num in range(1, 7):  # Rounds 1-6 individually
            col = f'Round_{round_num}_Picks'
            if col in df.columns:
                round_cols.append(col)
        
        # Add combined rounds 7+ column if it exists
        if 'Round_7Plus_Picks' in df.columns:
            round_cols.append('Round_7Plus_Picks')
        
        return sorted(round_cols)
    
    def _calculate_rolling_averages(self, df: pd.DataFrame, team: str, 
                                   round_cols: List[str]) -> pd.DataFrame:
        """Calculate rolling averages for a specific team"""
        team_data = df[df['Team'] == team].copy()
        team_data = team_data.sort_values('Draft_Year')
        
        result_rows = []
        
        for idx, row in team_data.iterrows():
            current_year = row['Draft_Year']
            result_row = {
                'Draft_Year': current_year,
                'Team': team
            }
            
            # Add current year data (all rounds including 7+)
            for col in round_cols:
                if col in row.index:
                    result_row[f'Current_{col}'] = row[col]
                else:
                    result_row[f'Current_{col}'] = 0
            
            # Add individual year lookbacks
            for years_back in range(1, 5):  # 1, 2, 3, 4 years ago
                target_year = current_year - years_back
                year_data = team_data[team_data['Draft_Year'] == target_year]
                
                if not year_data.empty:
                    if years_back == 4:
                        # Only Round 1 for 4 years ago
                        if 'Round_1_Picks' in year_data.columns:
                            result_row[f'Prev_{years_back}Yr_Round_1_Picks'] = year_data.iloc[0]['Round_1_Picks']
                        else:
                            result_row[f'Prev_{years_back}Yr_Round_1_Picks'] = 0
                    else:
                        # All rounds (including 7+) for 1, 2, 3 years ago
                        for col in round_cols:
                            if col in year_data.columns:
                                result_row[f'Prev_{years_back}Yr_{col}'] = year_data.iloc[0][col]
                            else:
                                result_row[f'Prev_{years_back}Yr_{col}'] = 0
                else:
                    # No data available for this year
                    if years_back == 4:
                        result_row[f'Prev_{years_back}Yr_Round_1_Picks'] = 0
                    else:
                        for col in round_cols:
                            result_row[f'Prev_{years_back}Yr_{col}'] = 0
            
            result_rows.append(result_row)
        
        return pd.DataFrame(result_rows)
    
    def _create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final dataset with rolling averages for all teams"""
        round_cols = self._get_round_columns(df)
        teams = sorted(df['Team'].unique())
        
        all_results = []
        
        print(f"Processing {len(teams)} teams...")
        print(f"Round columns found: {round_cols}")
        
        for i, team in enumerate(teams, 1):
            print(f"Processing team {i}/{len(teams)}: {team}")
            team_results = self._calculate_rolling_averages(df, team, round_cols)
            all_results.append(team_results)
        
        # Combine all team results
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by year and team
        final_df = final_df.sort_values(['Draft_Year', 'Team'])
        
        return final_df
    
    def _combine_high_rounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine rounds 7+ into a single feature"""
        df = df.copy()
        
        # Find all high round columns (Round 7 and above)
        high_round_cols = [col for col in df.columns 
                          if col.startswith('Round_') and col.endswith('_Picks')]
        high_round_cols = [col for col in high_round_cols 
                          if int(col.split('_')[1]) >= 7]
        
        if high_round_cols:
            # Sum all rounds 7+ picks
            df['Round_7Plus_Picks'] = df[high_round_cols].sum(axis=1)
            
            # Drop individual high round columns
            df = df.drop(columns=high_round_cols)
            
            print(f"Combined {len(high_round_cols)} high round columns into Round_7Plus_Picks")
        
        return df
    
    def _add_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary statistics and convert data types"""
        # Convert all pick count columns to integers
        pick_cols = [col for col in df.columns if '_Picks' in col]
        for col in pick_cols:
            df[col] = df[col].astype(int)
        
        # Ensure Draft_Year is integer
        df['Draft_Year'] = df['Draft_Year'].astype(int)
        
        return df
    
    def _save_final_dataset(self, df: pd.DataFrame, filename: str = "draft_picks_final.csv"):
        """Save the final dataset"""
        filepath = self.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved final draft dataset to {filepath}")
            
            # Also create a summary file
            self._create_dataset_summary(df, filename.replace('.csv', '_summary.txt'))
            
        except Exception as e:
            print(f"Error saving final dataset: {e}")
    
    def _create_dataset_summary(self, df: pd.DataFrame, summary_filename: str):
        """Create a summary file describing the dataset"""
        summary_path = self.output_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            f.write("Draft Picks Final Dataset Summary\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Years covered: {df['Draft_Year'].min()}-{df['Draft_Year'].max()}\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Teams: {len(df['Team'].unique())}\n")
            f.write(f"Years per team: {len(df) // len(df['Team'].unique())}\n\n")
            
            f.write("Column Categories:\n")
            f.write("- Current_*: Draft picks by round in the current year\n")
            f.write("- Prev_1Yr_*: Draft picks by round 1 year ago\n")
            f.write("- Prev_2Yr_*: Draft picks by round 2 years ago\n")
            f.write("- Prev_3Yr_*: Draft picks by round 3 years ago\n")
            f.write("- Prev_4Yr_Round_1_Picks: Round 1 draft picks 4 years ago only\n\n")
            
            f.write("Round Columns (1-6 + 7+):\n")
            for round_num in range(1, 7):
                current_col = f'Current_Round_{round_num}_Picks'
                if current_col in df.columns:
                    avg_picks = df[current_col].mean()
                    f.write(f"- Round {round_num}: Average {avg_picks:.2f} picks per team per year\n")
            
            # Add Round 7+ summary
            current_col = 'Current_Round_7Plus_Picks'
            if current_col in df.columns:
                avg_picks = df[current_col].mean()
                f.write(f"- Rounds 7+: Average {avg_picks:.2f} picks per team per year\n")
            
            f.write(f"\nTotal columns in dataset: {len(df.columns)}\n")
            
            # Sample of data availability
            f.write(f"\nData availability by year:\n")
            year_counts = df['Draft_Year'].value_counts().sort_index()
            for year, count in year_counts.items():
                f.write(f"- {year}: {count} teams\n")
        
        print(f"Saved dataset summary to {summary_path}")
    
    def finalize_draft_data(self, start_year: int = 1970, end_year: int = 2024):
        """Main function to create final draft dataset"""
        print(f"Creating final draft dataset for {start_year}-{end_year}")
        
        # Load comprehensive data (all available years for lookback calculations)
        print("Loading comprehensive draft data...")
        df_all = self._load_comprehensive_data()
        
        print(f"All data loaded: {len(df_all)} records covering {len(df_all['Draft_Year'].unique())} years ({df_all['Draft_Year'].min()}-{df_all['Draft_Year'].max()})")
        
        # Create final dataset with rolling averages using all data
        print("Calculating rolling averages using all available data...")
        final_df_all = self._create_final_dataset(df_all)
        
        # Filter final results to specified output years
        print(f"Filtering output to years {start_year}-{end_year}...")
        final_df = final_df_all[
            (final_df_all['Draft_Year'] >= start_year) & 
            (final_df_all['Draft_Year'] <= end_year)
        ].copy()
        
        print(f"Final data: {len(final_df)} records covering {len(final_df['Draft_Year'].unique())} years")
        
        # Add summary statistics
        print("Adding summary statistics...")
        final_df_with_stats = self._add_summary_statistics(final_df)
        
        # Save the final dataset
        filename = "draft_picks_final.csv"
        self._save_final_dataset(final_df_with_stats, filename)
        
        print(f"\nFinal dataset created successfully!")
        print(f"Records: {len(final_df_with_stats)}")
        print(f"Columns: {len(final_df_with_stats.columns)}")
        print(f"Years: {start_year}-{end_year}")
        print(f"Teams: {len(final_df_with_stats['Team'].unique())}")
        
        return final_df_with_stats
    
    def show_sample_data(self, df: pd.DataFrame, team: str = "dal", year: int = 2024):
        """Show sample data for a specific team and year"""
        sample = df[(df['Team'] == team) & (df['Draft_Year'] == year)]
        
        if sample.empty:
            print(f"No data found for {team} in {year}")
            return
        
        print(f"\nSample data for {team.upper()} in {year}:")
        print("-" * 50)
        
        # Show key columns only for readability
        key_cols = ['Draft_Year', 'Team', 'Current_Round_1_Picks', 
                   'Prev_1Yr_Round_1_Picks', 'Rolling_4Yr_Total_Round_1_Picks']
        
        available_cols = [col for col in key_cols if col in sample.columns]
        print(sample[available_cols].T)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Finalize draft data with rolling averages')
    parser.add_argument('--start-year', type=int, default=1970, 
                       help='Start year for analysis (default: 1970)')
    parser.add_argument('--end-year', type=int, default=2024, 
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--sample', type=str, 
                       help='Show sample data for specific team (e.g., "dal")')
    parser.add_argument('--sample-year', type=int, default=2024,
                       help='Year for sample data (default: 2024)')
    
    args = parser.parse_args()
    
    # Initialize finalizer
    finalizer = DraftDataFinalizer()
    
    # Create final dataset
    final_df = finalizer.finalize_draft_data(args.start_year, args.end_year)
    
    # Show sample if requested
    if args.sample:
        finalizer.show_sample_data(final_df, args.sample.lower(), args.sample_year)
    
    print("\nDraft data finalization completed!")


if __name__ == "__main__":
    main()