#!/usr/bin/env python3
"""
Salary Cap Total View Data Combiner Script

This script combines all yearly salary cap total view files from Spotrac data
into a single consolidated dataset, removing Spotrac team abbreviations and
adding year columns.

Usage:
    python scripts/combine_salary_cap_totals.py
"""

import pandas as pd
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import glob
import re

class SalaryCapTotalCombiner:
    """Combines yearly salary cap total view files into a single dataset"""
    
    def __init__(self, input_dir: str = "data/processed/Spotrac/total_view", 
                 output_dir: str = "data/final"):
        """
        Initialize the combiner
        
        Args:
            input_dir: Directory containing yearly salary cap total files
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
        
        # Minimal team mapping for remaining historical duplicates
        self.historical_team_map = {
            'SDSD': 'SDG',    # San Diego -> Los Angeles Chargers (normalize to SDG)
            'STLSTL': 'RAM',  # St. Louis -> Los Angeles Rams (normalize to RAM)
        }
    
    def extract_year_from_filename(self, filename: str) -> int:
        """
        Extract year from filename pattern: salary_cap_YYYY_processed.csv
        
        Args:
            filename: Filename to extract year from
            
        Returns:
            Year as integer, or None if not found
        """
        match = re.search(r'salary_cap_(\d{4})_processed\.csv', filename)
        if match:
            return int(match.group(1))
        return None
    
    def clean_and_filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove Spotrac team abbreviations and keep only percentage data
        
        Args:
            df: DataFrame with all columns
            
        Returns:
            DataFrame with PFR_Team, Year, and only percentage columns
        """
        # Columns to drop - remove absolute values and unwanted metadata
        columns_to_drop = []
        
        # Always drop these columns if they exist
        unwanted_columns = [
            'Team', 'Rank', 'Record', 'PlayersActive', 'Avg AgeTeam',
            'Total CapAllocations', 'Cap SpaceAll', 'Active53-Man', 
            'ReservesIR/PUP/NFI/SUSP', 'DeadCap'
        ]
        
        for col in unwanted_columns:
            if col in df.columns:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            self.logger.debug(f"Dropped columns: {columns_to_drop}")
        
        # Ensure PFR_Team is uppercase
        if 'PFR_Team' in df.columns:
            df['PFR_Team'] = df['PFR_Team'].str.upper().str.strip()
            
            # Apply minimal historical team mapping for remaining duplicates
            df['PFR_Team'] = df['PFR_Team'].map(self.historical_team_map).fillna(df['PFR_Team'])
        
        return df
    
    def load_and_process_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load and process a single salary cap total file
        
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
            
            # Add year column
            df['Year'] = year
            
            # Ensure PFR_Team column exists
            if 'PFR_Team' not in df.columns:
                self.logger.error(f"No PFR_Team column found in {file_path.name}")
                return None
            
            # Clean team columns and filter data
            df = self.clean_and_filter_columns(df)
            
            self.logger.info(f"Processed {file_path.name}: {len(df)} teams for year {year}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def combine_all_files(self) -> pd.DataFrame:
        """
        Combine all salary cap total files into a single DataFrame
        
        Returns:
            Combined DataFrame
        """
        # Find all salary cap total files
        pattern = self.input_dir / "salary_cap_*_processed.csv"
        files = list(glob.glob(str(pattern)))
        
        if not files:
            self.logger.error(f"No salary cap total files found in {self.input_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(files)} salary cap total files")
        
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
        Save combined salary cap total data
        
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
            output_file = self.output_dir / "salary_cap_totals_combined.csv"
            combined_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved combined salary cap totals to {output_file}")
            
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
            metadata_file = self.output_dir / "salary_cap_totals_combined_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            # Print summary statistics
            print(f"\nCombined Salary Cap Totals Summary:")
            print(f"  Total rows: {len(combined_df):,}")
            print(f"  Years: {min(years)}-{max(years)} ({len(years)} years)")
            print(f"  Teams: {len(teams)} unique teams")
            print(f"  Columns: {len(combined_df.columns)}")
            print(f"  Average rows per year: {len(combined_df) // len(years)}")
            
            # Show sample of data structure
            print(f"\nSample data structure:")
            sample_cols = ['PFR_Team', 'Year', 'Total CapAllocations_Pct', 'Cap SpaceAll_Pct', 'DeadCap_Pct']
            available_cols = [col for col in sample_cols if col in combined_df.columns]
            print(combined_df[available_cols].head())
            
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
        self.logger.info("Starting salary cap total data combination")
        
        # Combine all files
        combined_df = self.combine_all_files()
        
        if combined_df.empty:
            self.logger.error("No data to combine")
            return False
        
        # Save combined data
        success = self.save_combined_data(combined_df)
        
        if success:
            self.logger.info("Salary cap total combination completed successfully")
        else:
            self.logger.error("Failed to save combined data")
        
        return success


def main():
    """Main function to run salary cap total combination"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combine NFL salary cap total view data files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed/Spotrac/total_view',
        help='Input directory containing salary cap total files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for combined data'
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = SalaryCapTotalCombiner(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Run combination
    success = combiner.run_combination()
    
    if success:
        print(f"\nSuccess! Combined salary cap total data saved to: {args.output_dir}")
        print("Generated files:")
        print("  - salary_cap_totals_combined.csv (main dataset)")
        print("  - salary_cap_totals_combined_metadata.csv (processing metadata)")
    else:
        print("Failed to combine salary cap total data")
        sys.exit(1)


if __name__ == "__main__":
    main()