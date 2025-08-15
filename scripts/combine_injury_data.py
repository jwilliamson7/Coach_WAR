#!/usr/bin/env python3
"""
Injury Data Combination Script

This script combines all individual team/year injury CSV files from the raw data directory
into a single consolidated dataset in the final data directory for analysis.

Usage:
    python scripts/combine_injury_data.py
    python scripts/combine_injury_data.py --input-dir "custom/input/path"
    python scripts/combine_injury_data.py --output-dir "custom/output/path"
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional, Dict
import glob

class InjuryDataCombiner:
    """Combines injury data files into a single dataset for analysis"""
    
    def __init__(self, input_dir: str = "data/raw/Injuries", output_dir: str = "data/final"):
        """
        Initialize the combiner
        
        Args:
            input_dir: Directory containing raw injury CSV files
            output_dir: Directory to save combined dataset
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def find_injury_files(self) -> List[Path]:
        """
        Find all injury CSV files in the input directory and team subfolders
        
        Returns:
            List of Path objects for injury CSV files
        """
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        injury_files = []
        
        # Look for files in team subfolders: {TEAM}/{team}_{year}_injuries.csv
        for team_dir in self.input_dir.iterdir():
            if team_dir.is_dir():
                pattern = str(team_dir / "*_*_injuries.csv")
                team_files = [Path(f) for f in glob.glob(pattern)]
                injury_files.extend(team_files)
                if team_files:
                    self.logger.debug(f"Found {len(team_files)} files in {team_dir.name}/")
        
        # Also look for files in root directory (for backward compatibility)
        pattern = str(self.input_dir / "*_*_injuries.csv")
        root_files = [Path(f) for f in glob.glob(pattern)]
        injury_files.extend(root_files)
        if root_files:
            self.logger.debug(f"Found {len(root_files)} files in root directory")
        
        self.logger.info(f"Found {len(injury_files)} total injury files")
        return injury_files
    
    def load_injury_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate a single injury CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with injury data or None if loading failed
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['Team', 'Year']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"File {file_path.name} missing required columns: {missing_columns}")
                return None
            
            # Add source file info for debugging
            df['Source_File'] = file_path.name
            df['Source_Team_Dir'] = file_path.parent.name if file_path.parent.name != 'Injuries' else 'ROOT'
            
            self.logger.debug(f"Loaded {len(df)} records from {file_path.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path.name}: {e}")
            return None
    
    def combine_injury_data(self, injury_files: List[Path]) -> pd.DataFrame:
        """
        Combine all injury CSV files into a single DataFrame
        
        Args:
            injury_files: List of injury file paths
            
        Returns:
            Combined DataFrame with all injury data
        """
        if not injury_files:
            self.logger.error("No injury files provided to combine")
            return pd.DataFrame()
        
        all_dataframes = []
        failed_files = []
        
        self.logger.info(f"Loading {len(injury_files)} injury files...")
        
        for i, file_path in enumerate(injury_files, 1):
            if i % 50 == 0:  # Progress update every 50 files
                self.logger.info(f"Processing file {i}/{len(injury_files)}")
            
            df = self.load_injury_file(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
            else:
                failed_files.append(file_path.name)
        
        if not all_dataframes:
            self.logger.error("No valid injury files found to combine")
            return pd.DataFrame()
        
        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} files: {failed_files[:5]}...")
        
        # Combine all DataFrames
        self.logger.info(f"Combining {len(all_dataframes)} successful DataFrames...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        self.logger.info(f"Combined {len(all_dataframes)} files into {len(combined_df)} total records")
        
        return combined_df
    
    def clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize the combined injury data
        
        Args:
            df: Combined injury DataFrame
            
        Returns:
            Cleaned and standardized DataFrame
        """
        if df.empty:
            return df
        
        self.logger.info("Cleaning and standardizing combined data...")
        cleaned_df = df.copy()
        
        # Standardize team names (ensure uppercase)
        cleaned_df['Team'] = cleaned_df['Team'].str.upper()
        
        # Ensure Year is integer
        cleaned_df['Year'] = pd.to_numeric(cleaned_df['Year'], errors='coerce')
        
        # Drop rows with invalid years
        invalid_years = cleaned_df['Year'].isna()
        if invalid_years.any():
            self.logger.warning(f"Dropping {invalid_years.sum()} rows with invalid years")
            cleaned_df = cleaned_df[~invalid_years]
        
        # Fill missing numeric columns with 0
        numeric_columns = [
            'Questionable', 'Doubtful', 'Out', 'IR', 'PUP', 
            'Total_Weeks_Missed', 'Total_Players_Injured'
        ]
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
        
        # Sort by Team and Year
        cleaned_df = cleaned_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        # Remove duplicates (same team/year combination, keep latest by source file date if available)
        before_count = len(cleaned_df)
        
        # Check for duplicates
        duplicate_mask = cleaned_df.duplicated(subset=['Team', 'Year'], keep=False)
        if duplicate_mask.any():
            duplicates_df = cleaned_df[duplicate_mask].copy()
            
            # If we have scraped dates, keep the most recent
            if 'Scraped_Date' in duplicates_df.columns:
                duplicates_df['Scraped_Date'] = pd.to_datetime(duplicates_df['Scraped_Date'], errors='coerce')
                # Keep the row with the latest scraped date for each team/year
                latest_idx = duplicates_df.groupby(['Team', 'Year'])['Scraped_Date'].idxmax()
                keep_indices = latest_idx.dropna().astype(int)
                
                # Remove all duplicates, then add back the ones we want to keep
                cleaned_df = cleaned_df[~duplicate_mask]
                cleaned_df = pd.concat([cleaned_df, duplicates_df.loc[keep_indices]], ignore_index=True)
            else:
                # If no scraped date, just keep the last occurrence
                cleaned_df = cleaned_df.drop_duplicates(subset=['Team', 'Year'], keep='last')
        
        after_count = len(cleaned_df)
        if before_count != after_count:
            self.logger.warning(f"Removed {before_count - after_count} duplicate team/year records")
        
        # Add processing metadata
        cleaned_df['Combined_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Sort final output
        cleaned_df = cleaned_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        self.logger.info(f"Cleaned data: {len(cleaned_df)} records for {cleaned_df['Team'].nunique()} teams "
                        f"across years {cleaned_df['Year'].min()}-{cleaned_df['Year'].max()}")
        
        return cleaned_df
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the combined injury data
        
        Args:
            df: Cleaned injury DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        summary_stats = {
            'total_records': len(df),
            'teams_count': df['Team'].nunique(),
            'years_range': f"{df['Year'].min()}-{df['Year'].max()}",
            'years_covered': sorted(df['Year'].unique().tolist()),
            'teams_covered': sorted(df['Team'].unique().tolist()),
            'avg_records_per_team': len(df) / df['Team'].nunique(),
            'avg_records_per_year': len(df) / df['Year'].nunique()
        }
        
        # Calculate aggregate injury statistics
        if 'Total_Weeks_Missed' in df.columns:
            summary_stats['total_weeks_missed_all_teams'] = df['Total_Weeks_Missed'].sum()
            summary_stats['avg_weeks_missed_per_team_year'] = df['Total_Weeks_Missed'].mean()
        
        if 'Total_Players_Injured' in df.columns:
            summary_stats['total_players_injured_all_teams'] = df['Total_Players_Injured'].sum()
            summary_stats['avg_players_injured_per_team_year'] = df['Total_Players_Injured'].mean()
        
        return summary_stats
    
    def save_combined_data(self, df: pd.DataFrame, summary_stats: Dict) -> bool:
        """
        Save the combined injury dataset and metadata
        
        Args:
            df: Combined and cleaned DataFrame
            summary_stats: Summary statistics dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Save main combined dataset
            main_output_path = self.output_dir / "injury_data_combined.csv"
            df.to_csv(main_output_path, index=False)
            self.logger.info(f"Saved combined injury dataset to {main_output_path}")
            
            # Save summary statistics
            summary_df = pd.DataFrame([summary_stats])
            summary_output_path = self.output_dir / "injury_data_summary.csv"
            summary_df.to_csv(summary_output_path, index=False)
            self.logger.info(f"Saved summary statistics to {summary_output_path}")
            
            # Save team-year coverage matrix for quick reference
            if not df.empty:
                coverage_matrix = df.pivot_table(
                    index='Team', 
                    columns='Year', 
                    values='Total_Weeks_Missed',
                    aggfunc='first',
                    fill_value=None
                )
                coverage_output_path = self.output_dir / "injury_data_coverage_matrix.csv"
                coverage_matrix.to_csv(coverage_output_path)
                self.logger.info(f"Saved coverage matrix to {coverage_output_path}")
            
            # Save detailed metadata
            metadata = {
                'combination_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                **summary_stats
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_output_path = self.output_dir / "injury_data_combination_metadata.csv"
            metadata_df.to_csv(metadata_output_path, index=False)
            self.logger.info(f"Saved combination metadata to {metadata_output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving combined data: {e}")
            return False
    
    def combine_all_injury_data(self) -> bool:
        """
        Main processing function - combines all injury data into final dataset
        
        Returns:
            True if processing completed successfully, False otherwise
        """
        self.logger.info("Starting injury data combination for final dataset...")
        
        # Find all injury files
        injury_files = self.find_injury_files()
        if not injury_files:
            self.logger.error("No injury files found to combine")
            return False
        
        # Combine all files
        combined_df = self.combine_injury_data(injury_files)
        if combined_df.empty:
            self.logger.error("No valid data found in injury files")
            return False
        
        # Clean and standardize
        cleaned_df = self.clean_and_standardize(combined_df)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_stats(cleaned_df)
        
        # Save combined data
        success = self.save_combined_data(cleaned_df, summary_stats)
        
        if success:
            self.logger.info("Injury data combination completed successfully!")
            self.logger.info(f"Final dataset: {len(cleaned_df)} records, "
                           f"{cleaned_df['Team'].nunique()} teams, "
                           f"{cleaned_df['Year'].nunique()} years "
                           f"({cleaned_df['Year'].min()}-{cleaned_df['Year'].max()})")
            
            # Log top injury statistics
            if 'Total_Weeks_Missed' in cleaned_df.columns:
                top_injury_teams = cleaned_df.groupby('Team')['Total_Weeks_Missed'].sum().sort_values(ascending=False).head(5)
                self.logger.info(f"Teams with most total weeks missed: {top_injury_teams.to_dict()}")
        
        return success


def main():
    """Main function to run the injury data combiner"""
    parser = argparse.ArgumentParser(
        description='Combine all NFL injury data files into a single dataset for analysis'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw/Injuries',
        help='Input directory containing raw injury CSV files (default: data/raw/Injuries)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for combined dataset (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = InjuryDataCombiner(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Combine all injury data
    success = combiner.combine_all_injury_data()
    
    if success:
        print("\nInjury data combination completed successfully!")
        print(f"Combined dataset saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - injury_data_combined.csv (main dataset)")
        print("  - injury_data_summary.csv (summary statistics)")
        print("  - injury_data_coverage_matrix.csv (team-year coverage)")
        print("  - injury_data_combination_metadata.csv (processing metadata)")
    else:
        print("\nInjury data combination failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()