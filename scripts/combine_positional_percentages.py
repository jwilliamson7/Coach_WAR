#!/usr/bin/env python3
"""
Positional Percentage Data Combiner Script

This script combines all yearly positional percentage files from Spotrac data
into a single consolidated dataset, removing Spotrac team abbreviations and
adding year columns.

Usage:
    python scripts/combine_positional_percentages.py
"""

import pandas as pd
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import glob
import re

class PositionalPercentageCombiner:
    """Combines yearly positional percentage files into a single dataset"""
    
    def __init__(self, input_dir: str = "data/processed/Spotrac/positional_percentages", 
                 output_dir: str = "data/final"):
        """
        Initialize the combiner
        
        Args:
            input_dir: Directory containing yearly positional percentage files
            output_dir: Directory to save combined data
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
    
    def extract_year_from_filename(self, filename: str) -> int:
        """
        Extract year from filename pattern: positional_percentages_YYYY.csv
        
        Args:
            filename: Filename to extract year from
            
        Returns:
            Year as integer, or None if not found
        """
        match = re.search(r'positional_percentages_(\d{4})\.csv', filename)
        if match:
            return int(match.group(1))
        return None
    
    def clean_and_filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove cap figures, metadata columns, and keep only essential data
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            DataFrame with only team, year, and percentage columns
        """
        # Define columns to keep: PFR_Team, Year, and all percentage columns
        columns_to_keep = ['PFR_Team', 'Year']
        
        # Add all percentage columns (end with '_Pct')
        for col in df.columns:
            if col.endswith('_Pct'):
                columns_to_keep.append(col)
        
        # Filter to only the columns we want
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df_filtered = df[existing_columns]
        
        self.logger.debug(f"Kept {len(existing_columns)} columns: {existing_columns}")
        
        return df_filtered
    
    def load_and_process_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load and process a single positional percentage file
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Processed DataFrame or None if error
        """
        try:
            # Extract year from filename
            year = self.extract_year_from_filename(file_path.name)
            if year is None:
                self.logger.warning(f"Could not extract year from filename: {file_path.name}")
                return None
            
            # Load the file
            df = pd.read_csv(file_path)
            self.logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            
            # Add year column if it doesn't exist or update it
            df['Year'] = year
            
            # Ensure PFR_Team column exists
            if 'PFR_Team' not in df.columns:
                self.logger.error(f"No PFR_Team column found in {file_path.name}")
                return None
            
            # Clean up PFR_Team column (ensure uppercase)
            df['PFR_Team'] = df['PFR_Team'].str.upper().str.strip()
            
            # Filter columns to keep only team, year, and percentages
            df = self.clean_and_filter_columns(df)
            
            self.logger.info(f"Processed {file_path.name}: {len(df)} teams for year {year}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def combine_all_files(self) -> pd.DataFrame:
        """
        Combine all positional percentage files into a single DataFrame
        
        Returns:
            Combined DataFrame
        """
        # Find all positional percentage files
        pattern = self.input_dir / "positional_percentages_*.csv"
        files = list(glob.glob(str(pattern)))
        
        if not files:
            self.logger.error(f"No positional percentage files found in {self.input_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(files)} positional percentage files")
        
        all_dataframes = []
        processed_years = []
        
        for file_path in sorted(files):
            file_path = Path(file_path)
            df = self.load_and_process_file(file_path)
            
            if df is not None and not df.empty:
                all_dataframes.append(df)
                year = df['Year'].iloc[0]
                processed_years.append(year)
                self.logger.info(f"Added {len(df)} rows for year {year}")
        
        if not all_dataframes:
            self.logger.error("No valid data found in any files")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Reorder columns to put Year as second column after PFR_Team
        columns = list(combined_df.columns)
        if 'Year' in columns and 'PFR_Team' in columns:
            columns.remove('Year')
            pfr_index = columns.index('PFR_Team')
            columns.insert(pfr_index + 1, 'Year')
            combined_df = combined_df[columns]
        
        # Sort by year and team
        combined_df = combined_df.sort_values(['Year', 'PFR_Team'], ignore_index=True)
        
        self.logger.info(f"Combined data: {len(combined_df)} total rows across {len(processed_years)} years")
        self.logger.info(f"Years processed: {sorted(processed_years)}")
        self.logger.info(f"Teams per year: {len(combined_df) // len(processed_years) if processed_years else 0}")
        
        return combined_df
    
    def save_combined_data(self, combined_df: pd.DataFrame) -> bool:
        """
        Save combined positional percentage data
        
        Args:
            combined_df: Combined DataFrame to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if combined_df.empty:
                self.logger.warning("No data to save")
                return False
            
            # Save main combined file
            output_file = self.output_dir / "positional_percentages_combined.csv"
            combined_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved combined positional percentages to {output_file}")
            
            # Create metadata
            years = sorted(combined_df['Year'].unique())
            teams = sorted(combined_df['PFR_Team'].unique())
            
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Rows': len(combined_df),
                'Years_Included': len(years),
                'Year_Range': f"{min(years)}-{max(years)}" if years else "N/A",
                'Teams_Included': len(teams),
                'Team_List': ', '.join(teams),
                'Columns_Total': len(combined_df.columns),
                'Source_Directory': str(self.input_dir),
                'Files_Processed': len(years)
            }
            
            # Save metadata
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "positional_percentages_combined_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            # Print summary statistics
            print(f"\nCombined Positional Percentages Summary:")
            print(f"  Total rows: {len(combined_df):,}")
            print(f"  Years: {min(years)}-{max(years)} ({len(years)} years)")
            print(f"  Teams: {len(teams)} unique teams")
            print(f"  Columns: {len(combined_df.columns)}")
            print(f"  Average rows per year: {len(combined_df) // len(years)}")
            
            # Show sample of data structure
            print(f"\nSample data structure:")
            print(combined_df[['PFR_Team', 'Year', 'QB_Pct', 'RB_Pct', 'WR_Pct', 'TE_Pct', 'OL_Pct']].head())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving combined data: {e}")
            return False
    
    def run_combination(self) -> bool:
        """
        Run the complete combination process
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting positional percentage data combination")
        
        # Combine all files
        combined_df = self.combine_all_files()
        
        if combined_df.empty:
            self.logger.error("No data to combine")
            return False
        
        # Save combined data
        success = self.save_combined_data(combined_df)
        
        if success:
            self.logger.info("Positional percentage combination completed successfully")
        else:
            self.logger.error("Failed to save combined data")
        
        return success


def main():
    """Main function to run positional percentage combination"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combine NFL positional percentage data files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed/Spotrac/positional_percentages',
        help='Input directory containing positional percentage files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for combined data'
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = PositionalPercentageCombiner(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Run combination
    success = combiner.run_combination()
    
    if success:
        print(f"\nSuccess! Combined positional percentage data saved to: {args.output_dir}")
        print("Generated files:")
        print("  - positional_percentages_combined.csv (main dataset)")
        print("  - positional_percentages_combined_metadata.csv (processing metadata)")
    else:
        print("Failed to combine positional percentage data")
        sys.exit(1)


if __name__ == "__main__":
    main()