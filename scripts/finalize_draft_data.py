import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


class DraftDataFinalizer:
    """Creates final draft dataset with rolling averages for coaching analysis"""
    
    def __init__(self, processed_dir: str = "data/processed/Draft", 
                 output_dir: str = "data/final"):
        """Initialize finalizer with input and output directories"""
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_comprehensive_data(self) -> pd.DataFrame:
        """Load the comprehensive draft data"""
        filepath = self.processed_dir / "draft_picks_by_team_round_all_years.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Comprehensive draft data not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def _filter_years(self, df: pd.DataFrame, start_year: int = 2003, 
                     end_year: int = 2024) -> pd.DataFrame:
        """Filter data to specified year range"""
        return df[(df['Draft_Year'] >= start_year) & (df['Draft_Year'] <= end_year)].copy()
    
    def _get_round_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of round pick columns"""
        round_cols = [col for col in df.columns 
                     if col.startswith('Round_') and col.endswith('_Picks')]
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
            
            # Add current year data (all rounds)
            for col in round_cols:
                result_row[f'Current_{col}'] = row[col]
            
            # Add individual year lookbacks
            for years_back in range(1, 5):  # 1, 2, 3, 4 years ago
                target_year = current_year - years_back
                year_data = team_data[team_data['Draft_Year'] == target_year]
                
                if not year_data.empty:
                    if years_back == 4:
                        # Only Round 1 for 4 years ago
                        result_row[f'Prev_{years_back}Yr_Round_1_Picks'] = year_data.iloc[0]['Round_1_Picks']
                    else:
                        # All rounds for 1, 2, 3 years ago
                        for col in round_cols:
                            result_row[f'Prev_{years_back}Yr_{col}'] = year_data.iloc[0][col]
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
        
        for i, team in enumerate(teams, 1):
            print(f"Processing team {i}/{len(teams)}: {team}")
            team_results = self._calculate_rolling_averages(df, team, round_cols)
            all_results.append(team_results)
        
        # Combine all team results
        final_df = pd.concat(all_results, ignore_index=True)
        
        # Sort by year and team
        final_df = final_df.sort_values(['Draft_Year', 'Team'])
        
        return final_df
    
    def _add_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add summary statistics to the final dataset"""
        # No additional statistics needed - return dataframe as-is
        return df
    
    def _save_final_dataset(self, df: pd.DataFrame, filename: str = "draft_picks_final_2003_2024.csv"):
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
            
            f.write("Round Columns (1-7):\n")
            for round_num in range(1, 8):
                current_col = f'Current_Round_{round_num}_Picks'
                if current_col in df.columns:
                    avg_picks = df[current_col].mean()
                    f.write(f"- Round {round_num}: Average {avg_picks:.2f} picks per team per year\n")
            
            f.write(f"\nTotal columns in dataset: {len(df.columns)}\n")
            
            # Sample of data availability
            f.write(f"\nData availability by year:\n")
            year_counts = df['Draft_Year'].value_counts().sort_index()
            for year, count in year_counts.items():
                f.write(f"- {year}: {count} teams\n")
        
        print(f"Saved dataset summary to {summary_path}")
    
    def finalize_draft_data(self, start_year: int = 2003, end_year: int = 2024):
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
        filename = f"draft_picks_final_{start_year}_{end_year}.csv"
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
    parser.add_argument('--start-year', type=int, default=2003, 
                       help='Start year for analysis (default: 2003)')
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