#!/usr/bin/env python3
"""
Injury Data Processing Script

This script combines all individual team/year injury CSV files from the raw data directory
into a single consolidated dataset in the processed data directory.

Usage:
    python scripts/process_injury_data.py
    python scripts/process_injury_data.py --input-dir "custom/input/path"
    python scripts/process_injury_data.py --output-dir "custom/output/path"
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

class InjuryDataProcessor:
    """Processes and combines injury data files"""
    
    def __init__(self, input_dir: str = "data/raw/Injuries", output_dir: str = "data/processed/Injury"):
        """
        Initialize the processor
        
        Args:
            input_dir: Directory containing raw injury CSV files
            output_dir: Directory to save processed data
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
        Find all injury CSV files in the input directory
        
        Returns:
            List of Path objects for injury CSV files
        """
        if not self.input_dir.exists():
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            return []
        
        # Look for files matching pattern: {team}_{year}_injuries.csv
        pattern = str(self.input_dir / "*_*_injuries.csv")
        injury_files = [Path(f) for f in glob.glob(pattern)]
        
        self.logger.info(f"Found {len(injury_files)} injury files in {self.input_dir}")
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
            
            # Add source file info
            df['Source_File'] = file_path.name
            
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
        all_dataframes = []
        
        for file_path in injury_files:
            df = self.load_injury_file(file_path)
            if df is not None:
                all_dataframes.append(df)
        
        if not all_dataframes:
            self.logger.error("No valid injury files found to combine")
            return pd.DataFrame()
        
        # Combine all DataFrames
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
        
        cleaned_df = df.copy()
        
        # Standardize team names (ensure uppercase)
        cleaned_df['Team'] = cleaned_df['Team'].str.upper()
        
        # Ensure Year is integer
        cleaned_df['Year'] = pd.to_numeric(cleaned_df['Year'], errors='coerce')
        
        # Fill missing numeric columns with 0
        numeric_columns = ['Questionable', 'Doubtful', 'Out', 'IR', 'PUP', 
                          'Total_Weeks_Missed', 'Total_Players_Injured']
        
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
        
        # Sort by Team and Year
        cleaned_df = cleaned_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        # Remove duplicates (same team/year combination)
        duplicates = cleaned_df.duplicated(subset=['Team', 'Year'], keep='last')
        if duplicates.any():
            num_duplicates = duplicates.sum()
            self.logger.warning(f"Removing {num_duplicates} duplicate team/year records (keeping latest)")
            cleaned_df = cleaned_df[~duplicates]
        
        # Add processing metadata
        cleaned_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        self.logger.info(f"Cleaned data: {len(cleaned_df)} records for {cleaned_df['Team'].nunique()} teams "
                        f"across {cleaned_df['Year'].nunique()} years")
        
        return cleaned_df
    
    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the injury data
        
        Args:
            df: Cleaned injury DataFrame
            
        Returns:
            DataFrame with summary statistics
        """
        if df.empty:
            return pd.DataFrame()
        
        # Team-level summary (average across years)
        team_summary = df.groupby('Team').agg({
            'Year': ['min', 'max', 'count'],
            'Questionable': 'mean',
            'Doubtful': 'mean', 
            'Out': 'mean',
            'IR': 'mean',
            'PUP': 'mean',
            'Total_Weeks_Missed': 'mean',
            'Total_Players_Injured': 'mean'
        }).round(2)
        
        # Flatten column names
        team_summary.columns = ['_'.join(col).strip() for col in team_summary.columns]
        team_summary = team_summary.rename(columns={
            'Year_min': 'First_Year',
            'Year_max': 'Last_Year', 
            'Year_count': 'Years_Available',
            'Questionable_mean': 'Avg_Questionable',
            'Doubtful_mean': 'Avg_Doubtful',
            'Out_mean': 'Avg_Out',
            'IR_mean': 'Avg_IR',
            'PUP_mean': 'Avg_PUP',
            'Total_Weeks_Missed_mean': 'Avg_Total_Weeks_Missed',
            'Total_Players_Injured_mean': 'Avg_Total_Players_Injured'
        })
        
        team_summary = team_summary.reset_index()
        
        return team_summary
    
    def save_processed_data(self, df: pd.DataFrame, summary_df: pd.DataFrame) -> bool:
        """
        Save the processed injury data and summary statistics
        
        Args:
            df: Main processed DataFrame
            summary_df: Summary statistics DataFrame
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Save main processed data
            main_output_path = self.output_dir / "injury_data_combined.csv"
            df.to_csv(main_output_path, index=False)
            self.logger.info(f"Saved combined injury data to {main_output_path}")
            
            # Save summary statistics
            if not summary_df.empty:
                summary_output_path = self.output_dir / "injury_data_team_summary.csv"
                summary_df.to_csv(summary_output_path, index=False)
                self.logger.info(f"Saved team summary statistics to {summary_output_path}")
            
            # Save processing metadata
            metadata = {
                'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Records': len(df),
                'Teams_Count': df['Team'].nunique() if not df.empty else 0,
                'Years_Range': f"{df['Year'].min()}-{df['Year'].max()}" if not df.empty else "N/A",
                'Input_Directory': str(self.input_dir),
                'Output_Directory': str(self.output_dir)
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_output_path = self.output_dir / "processing_metadata.csv"
            metadata_df.to_csv(metadata_output_path, index=False)
            self.logger.info(f"Saved processing metadata to {metadata_output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {e}")
            return False
    
    def process_all_injury_data(self) -> bool:
        """
        Main processing function - combines and processes all injury data
        
        Returns:
            True if processing completed successfully, False otherwise
        """
        self.logger.info("Starting injury data processing...")
        
        # Find all injury files
        injury_files = self.find_injury_files()
        if not injury_files:
            self.logger.error("No injury files found to process")
            return False
        
        # Combine all files
        combined_df = self.combine_injury_data(injury_files)
        if combined_df.empty:
            self.logger.error("No valid data found in injury files")
            return False
        
        # Clean and standardize
        cleaned_df = self.clean_and_standardize(combined_df)
        
        # Generate summary statistics
        summary_df = self.generate_summary_stats(cleaned_df)
        
        # Save processed data
        success = self.save_processed_data(cleaned_df, summary_df)
        
        if success:
            self.logger.info("Injury data processing completed successfully!")
            self.logger.info(f"Final dataset: {len(cleaned_df)} records, "
                           f"{cleaned_df['Team'].nunique()} teams, "
                           f"{cleaned_df['Year'].nunique()} years")
        
        return success


def main():
    """Main function to run the injury data processor"""
    parser = argparse.ArgumentParser(
        description='Process and combine NFL injury data files'
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
        default='data/processed/Injury',
        help='Output directory for processed data (default: data/processed/Injury)'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = InjuryDataProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process all injury data
    success = processor.process_all_injury_data()
    
    if success:
        print("\n✅ Injury data processing completed successfully!")
        print(f"📁 Processed data saved to: {args.output_dir}")
        print("\nGenerated files:")
        print("  - injury_data_combined.csv (main dataset)")
        print("  - injury_data_team_summary.csv (team averages)")
        print("  - processing_metadata.csv (processing info)")
    else:
        print("\n❌ Injury data processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()