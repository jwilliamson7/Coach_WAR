import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional


class DraftDataTransformer:
    """Transforms draft pick data into team-by-round summary statistics"""
    
    def __init__(self, draft_dir: str = "data/raw/Draft", 
                 output_dir: str = "data/processed/Draft"):
        """Initialize transformer with input and output directories"""
        self.draft_dir = Path(draft_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _discover_draft_files(self) -> List[Path]:
        """Find all draft CSV files"""
        return list(self.draft_dir.glob("draft_*.csv"))
    
    def _extract_year_from_filename(self, filename: str) -> int:
        """Extract year from draft filename"""
        # Assumes format: draft_YYYY.csv
        try:
            year_part = filename.split('_')[1].replace('.csv', '')
            return int(year_part)
        except (IndexError, ValueError):
            return 0
    
    def _load_draft_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load and validate a draft file"""
        try:
            df = pd.read_csv(file_path)
            
            # Check for required columns
            required_cols = ['Team', 'Round']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Warning: Missing columns in {file_path.name}: {missing_cols}")
                return None
                
            return df
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return None
    
    def _standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize team abbreviations to match PFR format"""
        df_clean = df.copy()
        
        # Common team name mappings for draft data
        team_mappings = {
            # Handle variations in team abbreviations (uppercase to lowercase)
            'NWE': 'nwe',  # New England
            'NOR': 'nor',  # New Orleans
            'TAM': 'tam',  # Tampa Bay
            'GNB': 'gnb',  # Green Bay
            'KAN': 'kan',  # Kansas City
            'SFO': 'sfo',  # San Francisco
            'LAR': 'ram',  # Los Angeles Rams
            'LAC': 'lac',  # Los Angeles Chargers
            'LVR': 'rai',  # Las Vegas Raiders
            'RAI': 'rai',  # Raiders (Oakland/Las Vegas)
            'CRD': 'crd',  # Arizona Cardinals
            # Handle legacy team names - maintain franchise continuity
            'STL': 'ram',  # St. Louis Rams -> LA Rams
            'SD': 'lac',   # San Diego -> LA Chargers
            'SDG': 'lac',  # San Diego Chargers -> LA Chargers  
            'OAK': 'rai',  # Oakland -> Las Vegas Raiders
            # Additional franchise mappings
            'IND': 'clt',  # Indianapolis uses 'clt' in PFR
        }
        
        # Convert to lowercase and apply mappings
        df_clean['Team'] = df_clean['Team'].str.upper()
        df_clean['Team'] = df_clean['Team'].replace(team_mappings)
        df_clean['Team'] = df_clean['Team'].str.lower()
        
        return df_clean
    
    def _calculate_picks_by_team_round(self, df: pd.DataFrame, year: int) -> pd.DataFrame:
        """Calculate number of picks by team and round for a given year"""
        try:
            # Standardize team names
            df_clean = self._standardize_team_names(df)
            
            # Group by team and round, count picks
            picks_summary = df_clean.groupby(['Team', 'Round']).size().reset_index(name='Picks')
            
            # Add year column
            picks_summary.insert(0, 'Draft_Year', year)
            
            # Create pivot table with rounds as columns
            pivot_df = picks_summary.pivot_table(
                index=['Draft_Year', 'Team'], 
                columns='Round', 
                values='Picks', 
                fill_value=0
            ).reset_index()
            
            # Flatten column names
            pivot_df.columns.name = None
            round_cols = [col for col in pivot_df.columns if isinstance(col, int)]
            
            # Rename round columns to be more descriptive
            rename_dict = {}
            for round_num in round_cols:
                rename_dict[round_num] = f'Round_{round_num}_Picks'
            
            pivot_df = pivot_df.rename(columns=rename_dict)
            
            # Add total picks column
            round_pick_cols = [col for col in pivot_df.columns if col.startswith('Round_') and col.endswith('_Picks')]
            pivot_df['Total_Picks'] = pivot_df[round_pick_cols].sum(axis=1)
            
            return pivot_df
            
        except Exception as e:
            print(f"Error calculating picks summary for year {year}: {e}")
            return None
    
    def _create_comprehensive_summary(self, all_data: List[pd.DataFrame]) -> pd.DataFrame:
        """Create comprehensive summary across all years"""
        if not all_data:
            return pd.DataFrame()
        
        # Combine all yearly data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Fill missing round columns with 0
        round_cols = [col for col in combined_df.columns if col.startswith('Round_') and col.endswith('_Picks')]
        for col in round_cols:
            combined_df[col] = combined_df[col].fillna(0)
        
        return combined_df
    
    def _save_yearly_summary(self, df: pd.DataFrame, year: int):
        """Save yearly draft pick summary"""
        filename = f"draft_picks_by_team_round_{year}.csv"
        filepath = self.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved {year} draft pick summary to {filepath}")
        except Exception as e:
            print(f"Error saving {year} summary: {e}")
    
    def _save_comprehensive_summary(self, df: pd.DataFrame):
        """Save comprehensive multi-year summary"""
        filename = "draft_picks_by_team_round_all_years.csv"
        filepath = self.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved comprehensive summary to {filepath}")
            
            # Also create a summary statistics file
            self._create_summary_statistics(df)
            
        except Exception as e:
            print(f"Error saving comprehensive summary: {e}")
    
    def _create_summary_statistics(self, df: pd.DataFrame):
        """Create summary statistics about draft picks"""
        stats_data = []
        
        # Overall statistics
        total_years = df['Draft_Year'].nunique()
        total_teams = df['Team'].nunique() 
        avg_picks_per_team_per_year = df['Total_Picks'].mean()
        
        stats_data.append({
            'Metric': 'Total Years',
            'Value': total_years
        })
        stats_data.append({
            'Metric': 'Unique Teams',
            'Value': total_teams
        })
        stats_data.append({
            'Metric': 'Avg Picks Per Team Per Year',
            'Value': round(avg_picks_per_team_per_year, 2)
        })
        
        # Round-specific statistics
        round_cols = [col for col in df.columns if col.startswith('Round_') and col.endswith('_Picks')]
        for col in sorted(round_cols):
            round_num = col.split('_')[1]
            avg_picks = df[col].mean()
            stats_data.append({
                'Metric': f'Avg Round {round_num} Picks Per Team Per Year',
                'Value': round(avg_picks, 2)
            })
        
        # Save statistics
        stats_df = pd.DataFrame(stats_data)
        stats_filepath = self.output_dir / "draft_pick_statistics.csv"
        stats_df.to_csv(stats_filepath, index=False)
        print(f"Saved draft statistics to {stats_filepath}")
    
    def transform_single_year(self, year: int) -> bool:
        """Transform draft data for a single year"""
        draft_file = self.draft_dir / f"draft_{year}.csv"
        
        if not draft_file.exists():
            print(f"Draft file for {year} not found: {draft_file}")
            return False
        
        print(f"Processing draft data for {year}...")
        
        # Load the draft file
        df = self._load_draft_file(draft_file)
        if df is None:
            return False
        
        # Calculate picks by team and round
        summary_df = self._calculate_picks_by_team_round(df, year)
        if summary_df is None:
            return False
        
        # Save the summary
        self._save_yearly_summary(summary_df, year)
        
        print(f"Processed {len(summary_df)} team records for {year}")
        return True
    
    def transform_all_years(self) -> None:
        """Transform draft data for all available years"""
        draft_files = self._discover_draft_files()
        
        if not draft_files:
            print("No draft files found")
            return
        
        print(f"Found {len(draft_files)} draft files to process")
        
        all_summaries = []
        processed_count = 0
        failed_count = 0
        
        for draft_file in sorted(draft_files):
            year = self._extract_year_from_filename(draft_file.name)
            if year == 0:
                print(f"Could not extract year from {draft_file.name}")
                failed_count += 1
                continue
            
            print(f"Processing {draft_file.name} (year: {year})...")
            
            # Load and process the file
            df = self._load_draft_file(draft_file)
            if df is None:
                failed_count += 1
                continue
            
            # Calculate summary
            summary_df = self._calculate_picks_by_team_round(df, year)
            if summary_df is None:
                failed_count += 1
                continue
            
            # Save yearly summary
            self._save_yearly_summary(summary_df, year)
            all_summaries.append(summary_df)
            processed_count += 1
            
            print(f"  * Processed {len(summary_df)} team records")
        
        # Create and save comprehensive summary
        if all_summaries:
            comprehensive_df = self._create_comprehensive_summary(all_summaries)
            self._save_comprehensive_summary(comprehensive_df)
        
        print(f"\nTransformation Summary:")
        print(f"Successfully processed: {processed_count} years")
        print(f"Failed: {failed_count} years")
    
    def show_sample_data(self, year: int = 2024) -> None:
        """Show sample of transformed data for a specific year"""
        draft_file = self.draft_dir / f"draft_{year}.csv"
        
        if not draft_file.exists():
            print(f"Draft file for {year} not found")
            return
        
        df = self._load_draft_file(draft_file)
        if df is None:
            return
        
        summary_df = self._calculate_picks_by_team_round(df, year)
        if summary_df is None:
            return
        
        print(f"\nSample draft pick summary for {year}:")
        print(summary_df.head(10).to_string(index=False))


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Transform draft data into team-by-round summaries')
    parser.add_argument('--year', type=int, help='Process specific year only')
    parser.add_argument('--sample', type=int, help='Show sample data for specific year')
    
    args = parser.parse_args()
    
    # Initialize transformer
    transformer = DraftDataTransformer()
    
    if args.sample:
        transformer.show_sample_data(args.sample)
        return
    
    if args.year:
        # Process single year
        success = transformer.transform_single_year(args.year)
        if not success:
            sys.exit(1)
    else:
        # Process all years
        transformer.transform_all_years()
    
    print("\nDraft data transformation completed!")


if __name__ == "__main__":
    main()