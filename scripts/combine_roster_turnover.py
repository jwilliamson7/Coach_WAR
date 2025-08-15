#!/usr/bin/env python3
"""
Roster Turnover Combination Script

This script combines individual team roster turnover files into consolidated datasets
for analysis. It processes both detailed year-to-year data and summary statistics.

Usage:
    python scripts/combine_roster_turnover.py
    python scripts/combine_roster_turnover.py --input-dir "custom/input/path"
    python scripts/combine_roster_turnover.py --output-dir "custom/output/path"
"""

import pandas as pd
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Optional
import glob

class RosterTurnoverCombiner:
    """Combines individual team roster turnover files into consolidated datasets"""
    
    def __init__(self, input_dir: str = "data/processed/RosterTurnover", output_dir: str = "data/final"):
        """
        Initialize the combiner
        
        Args:
            input_dir: Directory containing individual team turnover files
            output_dir: Directory to save combined datasets
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
        
        # Input subdirectories
        self.detailed_dir = self.input_dir / "detailed"
        self.summary_dir = self.input_dir / "summary"
    
    def find_turnover_files(self, file_type: str) -> List[Path]:
        """
        Find all turnover files of a specific type
        
        Args:
            file_type: Either 'detailed' or 'summary'
            
        Returns:
            List of file paths
        """
        if file_type == 'detailed':
            search_dir = self.detailed_dir
            pattern = "*_roster_turnover_detailed.csv"
        elif file_type == 'summary':
            search_dir = self.summary_dir
            pattern = "*_roster_turnover_summary.csv"
        else:
            raise ValueError("file_type must be 'detailed' or 'summary'")
        
        if not search_dir.exists():
            self.logger.warning(f"Directory does not exist: {search_dir}")
            return []
        
        files = list(search_dir.glob(pattern))
        self.logger.info(f"Found {len(files)} {file_type} files in {search_dir}")
        return files
    
    def load_turnover_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load a single turnover CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame or None if loading failed
        """
        try:
            df = pd.read_csv(file_path)
            
            # Add source file info
            df['Source_File'] = file_path.name
            
            self.logger.debug(f"Loaded {len(df)} records from {file_path.name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path.name}: {e}")
            return None
    
    def combine_files(self, files: List[Path], file_type: str) -> pd.DataFrame:
        """
        Combine multiple turnover files into a single DataFrame
        
        Args:
            files: List of file paths to combine
            file_type: Type of files being combined
            
        Returns:
            Combined DataFrame
        """
        if not files:
            self.logger.warning(f"No {file_type} files to combine")
            return pd.DataFrame()
        
        all_dataframes = []
        
        for file_path in files:
            df = self.load_turnover_file(file_path)
            if df is not None and not df.empty:
                all_dataframes.append(df)
        
        if not all_dataframes:
            self.logger.error(f"No valid {file_type} data found")
            return pd.DataFrame()
        
        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by team and year
        if 'Team' in combined_df.columns:
            sort_columns = ['Team']
            if 'Year_From' in combined_df.columns:
                sort_columns.extend(['Year_From', 'Year_To'])
            elif 'First_Year' in combined_df.columns:
                sort_columns.append('First_Year')
            
            if 'Position_Group' in combined_df.columns:
                sort_columns.append('Position_Group')
            
            combined_df = combined_df.sort_values(sort_columns, ignore_index=True)
        
        # Add combination metadata
        combined_df['Combined_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        combined_df['Files_Combined'] = len(all_dataframes)
        
        self.logger.info(f"Combined {len(all_dataframes)} {file_type} files into {len(combined_df)} total records")
        
        return combined_df
    
    def clean_combined_data(self, df: pd.DataFrame, file_type: str) -> pd.DataFrame:
        """
        Clean and standardize the combined data
        
        Args:
            df: Combined DataFrame
            file_type: Type of data being cleaned
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        cleaned_df = df.copy()
        
        # Standardize team names
        if 'Team' in cleaned_df.columns:
            cleaned_df['Team'] = cleaned_df['Team'].str.upper()
        
        # Ensure numeric columns are properly typed
        numeric_columns = []
        if file_type == 'detailed':
            numeric_columns = [
                'Year_From', 'Year_To', 'Players_Year1', 'Players_Year2', 
                'Players_Retained', 'Players_Departed', 'Players_New',
                'Retention_Rate_Pct', 'Departure_Rate_Pct', 'New_Player_Rate_Pct', 'Net_Change'
            ]
        elif file_type == 'summary':
            numeric_columns = [
                'First_Year', 'Last_Year', 'Years_Analyzed',
                'Avg_Players_Year1', 'Avg_Players_Year2',
                'Avg_Retention_Rate_Pct', 'Avg_Departure_Rate_Pct', 'Avg_New_Player_Rate_Pct',
                'Avg_Net_Change'
            ]
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Remove duplicates if any
        before_count = len(cleaned_df)
        if file_type == 'detailed':
            # For detailed data, duplicates are same team, years, and position
            duplicate_cols = ['Team', 'Year_From', 'Year_To', 'Position_Group']
        else:
            # For summary data, duplicates are same team and position
            duplicate_cols = ['Team', 'Position_Group']
        
        # Only check for duplicates if all required columns exist
        if all(col in cleaned_df.columns for col in duplicate_cols):
            cleaned_df = cleaned_df.drop_duplicates(subset=duplicate_cols, keep='last')
            after_count = len(cleaned_df)
            
            if before_count != after_count:
                self.logger.warning(f"Removed {before_count - after_count} duplicate records")
        
        self.logger.info(f"Cleaned {file_type} data: {len(cleaned_df)} records for "
                        f"{cleaned_df['Team'].nunique() if 'Team' in cleaned_df.columns else 'unknown'} teams")
        
        return cleaned_df
    
    def save_combined_data(self, detailed_df: pd.DataFrame, summary_df: pd.DataFrame) -> bool:
        """
        Save the combined datasets
        
        Args:
            detailed_df: Combined detailed DataFrame
            summary_df: Combined summary DataFrame
            
        Returns:
            True if saved successfully
        """
        try:
            success_count = 0
            
            # Save detailed data
            if not detailed_df.empty:
                detailed_file = self.output_dir / "roster_turnover_detailed_combined.csv"
                detailed_df.to_csv(detailed_file, index=False)
                self.logger.info(f"Saved combined detailed data to {detailed_file}")
                success_count += 1
            
            # Save summary data
            if not summary_df.empty:
                summary_file = self.output_dir / "roster_turnover_summary_combined.csv"
                summary_df.to_csv(summary_file, index=False)
                self.logger.info(f"Saved combined summary data to {summary_file}")
                success_count += 1
            
            # Save metadata
            metadata = {
                'Combination_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Input_Directory': str(self.input_dir),
                'Output_Directory': str(self.output_dir),
                'Detailed_Records': len(detailed_df),
                'Summary_Records': len(summary_df),
                'Teams_Detailed': detailed_df['Team'].nunique() if not detailed_df.empty else 0,
                'Teams_Summary': summary_df['Team'].nunique() if not summary_df.empty else 0
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "roster_turnover_combination_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved combination metadata to {metadata_file}")
            
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error saving combined data: {e}")
            return False
    
    def combine_all_turnover_data(self) -> bool:
        """
        Main function to combine all roster turnover data
        
        Returns:
            True if combination completed successfully
        """
        self.logger.info("Starting roster turnover data combination...")
        
        # Find files
        detailed_files = self.find_turnover_files('detailed')
        summary_files = self.find_turnover_files('summary')
        
        if not detailed_files and not summary_files:
            self.logger.error("No turnover files found to combine")
            return False
        
        # Combine detailed files
        detailed_df = pd.DataFrame()
        if detailed_files:
            detailed_df = self.combine_files(detailed_files, 'detailed')
            if not detailed_df.empty:
                detailed_df = self.clean_combined_data(detailed_df, 'detailed')
        
        # Combine summary files
        summary_df = pd.DataFrame()
        if summary_files:
            summary_df = self.combine_files(summary_files, 'summary')
            if not summary_df.empty:
                summary_df = self.clean_combined_data(summary_df, 'summary')
        
        # Save combined data
        success = self.save_combined_data(detailed_df, summary_df)
        
        if success:
            self.logger.info("Roster turnover data combination completed successfully!")
            if not detailed_df.empty:
                self.logger.info(f"Detailed data: {len(detailed_df)} records, "
                               f"{detailed_df['Team'].nunique()} teams")
            if not summary_df.empty:
                self.logger.info(f"Summary data: {len(summary_df)} records, "
                               f"{summary_df['Team'].nunique()} teams")
        
        return success


def main():
    """Main function to run the roster turnover combiner"""
    parser = argparse.ArgumentParser(
        description='Combine individual team roster turnover files into consolidated datasets'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/processed/RosterTurnover',
        help='Input directory containing individual team turnover files (default: data/processed/RosterTurnover)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for combined datasets (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = RosterTurnoverCombiner(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Combine all turnover data
    success = combiner.combine_all_turnover_data()
    
    if success:
        print("\n✅ Roster turnover data combination completed successfully!")
        print(f"📁 Combined datasets saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - roster_turnover_detailed_combined.csv (all year-to-year comparisons)")
        print("  - roster_turnover_summary_combined.csv (all team position averages)")
        print("  - roster_turnover_combination_metadata.csv (combination info)")
    else:
        print("\n❌ Roster turnover data combination failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()