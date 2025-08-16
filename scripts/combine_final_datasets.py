#!/usr/bin/env python3
"""
Combine All Final Datasets into Single Team-Year Table

This script combines all non-metadata CSV files from data/final/ into a single dataset
using salary_cap_totals_combined.csv as the base table. Only team-year combinations
that exist in the base table will be included in the final output.

Usage:
    python scripts/combine_final_datasets.py
    python scripts/combine_final_datasets.py --output-dir data/analysis
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob

class FinalDatasetCombiner:
    """Combines all final datasets into a single team-year table"""
    
    def __init__(self, final_dir: str = "data/final", 
                 output_dir: str = "data/final"):
        """
        Initialize the combiner
        
        Args:
            final_dir: Directory containing final datasets
            output_dir: Directory to save combined dataset
        """
        self.final_dir = Path(final_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Base table file
        self.base_file = "salary_cap_totals_combined.csv"
        
        # Files to combine (excluding metadata and specific files)
        self.files_to_combine = [
            "positional_percentages_combined.csv",
            "roster_turnover_crosstab.csv", 
            "starters_turnover_crosstab.csv",
            "starters_games_missed_crosstab.csv",
            "age_experience_metrics_crosstab.csv",
            "av_metrics_crosstab.csv",
            "penalty_interception_metrics.csv",
            "sos_winning_percentage.csv"
        ]
        
        # Columns to exclude from joins (date/metadata columns)
        self.exclude_columns = [
            'Analysis_Date', 'Extraction_Date', 'Creation_Date',
            'Last_Updated', 'Generated_Date', 'Processed_Date'
        ]
    
    def load_base_table(self) -> Optional[pd.DataFrame]:
        """
        Load the base salary cap table
        
        Returns:
            Base DataFrame or None if failed
        """
        base_path = self.final_dir / self.base_file
        
        if not base_path.exists():
            self.logger.error(f"Base file not found: {base_path}")
            return None
        
        try:
            df = pd.read_csv(base_path)
            
            # Check for required columns
            if 'PFR_Team' not in df.columns or 'Year' not in df.columns:
                self.logger.error("Base table missing required columns: PFR_Team, Year")
                return None
            
            # Standardize team column name
            df = df.rename(columns={'PFR_Team': 'Team'})
            
            # Ensure Team and Year are properly formatted
            df['Team'] = df['Team'].str.upper()
            df['Year'] = df['Year'].astype(int)
            
            self.logger.info(f"Loaded base table: {len(df)} rows, {len(df.columns)} columns")
            self.logger.info(f"Team-Year range: {df['Year'].min()}-{df['Year'].max()}")
            self.logger.info(f"Teams: {df['Team'].nunique()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading base table: {e}")
            return None
    
    def load_dataset(self, filename: str) -> Optional[pd.DataFrame]:
        """
        Load a dataset file and prepare it for joining
        
        Args:
            filename: Name of the file to load
            
        Returns:
            Prepared DataFrame or None if failed
        """
        file_path = self.final_dir / filename
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                self.logger.warning(f"Empty file: {filename}")
                return None
            
            # Standardize team column names
            team_columns = ['Team', 'PFR_Team']
            team_col = None
            for col in team_columns:
                if col in df.columns:
                    team_col = col
                    break
            
            if team_col is None:
                self.logger.warning(f"No team column found in {filename}")
                return None
            
            # Rename to standard 'Team' if needed
            if team_col != 'Team':
                df = df.rename(columns={team_col: 'Team'})
            
            # Check for Year column
            if 'Year' not in df.columns:
                # Special handling for turnover files that use Year_To
                if 'Year_To' in df.columns:
                    df = df.rename(columns={'Year_To': 'Year'})
                    # Drop Year_From as it's redundant
                    if 'Year_From' in df.columns:
                        df = df.drop(columns=['Year_From'])
                else:
                    self.logger.warning(f"No Year column found in {filename}")
                    return None
            
            # Standardize data types
            df['Team'] = df['Team'].str.upper()
            df['Year'] = df['Year'].astype(int)
            
            # Remove excluded columns
            cols_to_drop = [col for col in self.exclude_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.logger.info(f"Dropped columns from {filename}: {cols_to_drop}")
            
            self.logger.info(f"Loaded {filename}: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return None
    
    def create_team_year_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a team-year key for joining
        
        Args:
            df: DataFrame with Team and Year columns
            
        Returns:
            DataFrame with team_year_key column
        """
        df = df.copy()
        df['team_year_key'] = df['Team'].astype(str) + '_' + df['Year'].astype(str)
        return df
    
    def combine_datasets(self) -> Optional[pd.DataFrame]:
        """
        Combine all datasets using the base table as foundation
        
        Returns:
            Combined DataFrame or None if failed
        """
        # Load base table
        combined_df = self.load_base_table()
        if combined_df is None:
            return None
        
        # Create team-year key for base table
        combined_df = self.create_team_year_key(combined_df)
        base_keys = set(combined_df['team_year_key'])
        
        self.logger.info(f"Base table has {len(base_keys)} unique team-year combinations")
        
        # Join each dataset
        successful_joins = []
        failed_joins = []
        
        for filename in self.files_to_combine:
            self.logger.info(f"Processing {filename}")
            
            # Load dataset
            df = self.load_dataset(filename)
            if df is None:
                failed_joins.append(filename)
                continue
            
            # Create team-year key
            df = self.create_team_year_key(df)
            
            # Filter to only include team-year combinations in base table
            df_filtered = df[df['team_year_key'].isin(base_keys)]
            
            if len(df_filtered) == 0:
                self.logger.warning(f"No matching team-year combinations for {filename}")
                failed_joins.append(filename)
                continue
            
            # Drop duplicate team_year_key, Team, Year columns for joining
            join_cols = [col for col in df_filtered.columns 
                        if col not in ['team_year_key', 'Team', 'Year']]
            df_to_join = df_filtered[['team_year_key'] + join_cols]
            
            # Check for column name conflicts
            existing_cols = set(combined_df.columns)
            new_cols = set(join_cols)
            conflicts = existing_cols.intersection(new_cols)
            
            if conflicts:
                self.logger.warning(f"Column conflicts in {filename}: {conflicts}")
                # Add suffix to conflicting columns
                suffix = f"_{filename.split('.')[0].split('_')[-1]}"
                rename_dict = {col: col + suffix for col in conflicts}
                df_to_join = df_to_join.rename(columns=rename_dict)
                self.logger.info(f"Renamed columns: {rename_dict}")
            
            # Perform left join
            before_cols = len(combined_df.columns)
            combined_df = combined_df.merge(
                df_to_join, 
                on='team_year_key', 
                how='left'
            )
            after_cols = len(combined_df.columns)
            
            added_cols = after_cols - before_cols
            overlap = len(df_filtered)
            
            self.logger.info(f"Joined {filename}: +{added_cols} columns, {overlap} matching rows")
            successful_joins.append(filename)
        
        # Remove team_year_key helper column
        combined_df = combined_df.drop(columns=['team_year_key'])
        
        # Sort by Team and Year
        combined_df = combined_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        self.logger.info(f"Successfully joined {len(successful_joins)} datasets")
        self.logger.info(f"Failed to join {len(failed_joins)} datasets: {failed_joins}")
        self.logger.info(f"Final combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def save_combined_dataset(self, df: pd.DataFrame) -> bool:
        """
        Save the combined dataset with metadata
        
        Args:
            df: Combined DataFrame
            
        Returns:
            True if saved successfully, False otherwise
        """
        if df.empty:
            self.logger.warning("No data to save")
            return False
        
        try:
            # Save main combined file
            output_file = self.output_dir / "combined_final_dataset.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved combined dataset to {output_file}")
            
            # Calculate summary statistics
            total_rows = len(df)
            teams_count = df['Team'].nunique()
            year_range = f"{df['Year'].min()}-{df['Year'].max()}"
            total_columns = len(df.columns)
            
            # Count data completeness by major categories
            completeness_stats = {}
            
            # Salary cap columns
            salary_cols = [col for col in df.columns if 'Cap' in col or 'Pct' in col]
            if salary_cols:
                salary_complete = df[salary_cols].notna().all(axis=1).sum()
                completeness_stats['Salary_Cap_Complete'] = salary_complete
            
            # Position columns  
            position_cols = [col for col in df.columns if any(pos in col for pos in ['QB_', 'RB_', 'WR_', 'TE_', 'OL_', 'DL_', 'LB_', 'CB_', 'S_'])]
            if position_cols:
                position_complete = df[position_cols].notna().all(axis=1).sum()
                completeness_stats['Position_Metrics_Complete'] = position_complete
            
            # Winning percentage
            if 'Win_Pct' in df.columns:
                win_pct_complete = df['Win_Pct'].notna().sum()
                completeness_stats['Win_Pct_Complete'] = win_pct_complete
            
            # Create metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Rows': total_rows,
                'Total_Columns': total_columns,
                'Teams_Count': teams_count,
                'Year_Range': year_range,
                'Base_File': self.base_file,
                'Files_Combined': len(self.files_to_combine),
                'Description': 'Combined dataset from all final data sources with team-year as key'
            }
            
            # Add completeness stats
            metadata.update(completeness_stats)
            
            # Save metadata
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "combined_final_dataset_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving combined dataset: {e}")
            return False
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the combined dataset
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'unique_teams': df['Team'].nunique(),
            'unique_years': df['Year'].nunique(),
            'year_range': f"{df['Year'].min()}-{df['Year'].max()}",
            'data_completeness': {}
        }
        
        # Calculate completeness by column category
        categories = {
            'salary_cap': [col for col in df.columns if 'Cap' in col or 'Total' in col],
            'positional_spending': [col for col in df.columns if col.endswith('_Pct') and any(pos in col for pos in ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'SEC'])],
            'turnover_roster': [col for col in df.columns if 'Retention_Rate' in col or 'Departure_Rate' in col],
            'age_experience': [col for col in df.columns if 'Age' in col or 'Experience' in col or 'Exp' in col],
            'av_metrics': [col for col in df.columns if 'AV' in col],
            'performance': [col for col in df.columns if col in ['Win_Pct', 'SoS', 'W', 'L', 'T']],
            'penalties': [col for col in df.columns if 'Pen' in col or 'Int_Passing' in col]
        }
        
        for category, cols in categories.items():
            if cols:
                available_cols = [col for col in cols if col in df.columns]
                if available_cols:
                    complete_rows = df[available_cols].notna().all(axis=1).sum()
                    stats['data_completeness'][category] = {
                        'columns': len(available_cols),
                        'complete_rows': complete_rows,
                        'completeness_pct': round(complete_rows / len(df) * 100, 1)
                    }
        
        return stats


def main():
    """Main function to run dataset combination"""
    parser = argparse.ArgumentParser(
        description='Combine all final datasets into single team-year table'
    )
    parser.add_argument(
        '--final-dir',
        type=str,
        default='data/final',
        help='Directory containing final datasets (default: data/final)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for combined dataset (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize combiner
    combiner = FinalDatasetCombiner(
        final_dir=args.final_dir,
        output_dir=args.output_dir
    )
    
    print("Combining all final datasets...")
    print(f"Base table: {combiner.base_file}")
    print(f"Files to combine: {len(combiner.files_to_combine)}")
    
    # Combine datasets
    combined_df = combiner.combine_datasets()
    
    if combined_df is None or combined_df.empty:
        print("Error: No data combined")
        sys.exit(1)
    
    # Save combined dataset
    if combiner.save_combined_dataset(combined_df):
        print(f"\nSuccessfully combined datasets!")
        print(f"Results saved to: {args.output_dir}")
        print("Generated files:")
        print("  - combined_final_dataset.csv (main dataset)")
        print("  - combined_final_dataset_metadata.csv (metadata)")
        
        # Display summary statistics
        stats = combiner.get_summary_stats(combined_df)
        print(f"\nSummary Statistics:")
        print(f"  - Total rows: {stats['total_rows']:,}")
        print(f"  - Total columns: {stats['total_columns']:,}")
        print(f"  - Unique teams: {stats['unique_teams']}")
        print(f"  - Year range: {stats['year_range']}")
        
        # Display completeness by category
        if 'data_completeness' in stats:
            print(f"\nData Completeness by Category:")
            for category, comp_stats in stats['data_completeness'].items():
                print(f"  - {category.replace('_', ' ').title()}: {comp_stats['complete_rows']:,} rows ({comp_stats['completeness_pct']}%) across {comp_stats['columns']} columns")
        
    else:
        print("Error: Failed to save combined dataset")
        sys.exit(1)


if __name__ == "__main__":
    main()