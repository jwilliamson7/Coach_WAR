#!/usr/bin/env python3
"""
Combine All Final Datasets into Single Team-Year Table

This script combines all non-metadata CSV files from data/final/ into a single dataset
using coaching data as the base. Only team-years with coaching data will be included
in the final output (inner join on coaching data).

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

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import standardize_team_abbreviation

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
        
        # All data files to combine (no base table concept)
        # Note: Head coach data comes from data/processed/, others from data/final/
        self.files_to_combine = [
            "salary_cap_totals_combined.csv",
            "positional_percentages_combined.csv",
            "roster_turnover_crosstab.csv", 
            "starters_turnover_crosstab.csv",
            "starters_games_missed_crosstab.csv",
            "age_experience_metrics_crosstab.csv",
            "av_metrics_crosstab.csv",
            "penalty_interception_metrics.csv",
            "sos_winning_percentage.csv",
            "draft_picks_final.csv"
        ]
        
        # Additional files from other directories
        # Note: coaching and coach_win_pct will be merged together first
        self.additional_files = {
            "../processed/Coaching/yearly_coach_performance.csv": "coaching"
        }

        # File to merge with coaching data before joining to main dataset
        self.coach_win_pct_file = "../processed/Coaching/coach_specific_win_pct.csv"
        
        # Columns to exclude from joins (date/metadata columns)
        self.exclude_columns = [
            'Analysis_Date', 'Extraction_Date', 'Creation_Date',
            'Last_Updated', 'Generated_Date', 'Processed_Date'
        ]

    def standardize_team_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize team abbreviations to uppercase for consistency

        NOTE: Source data is already in correct PFR format (lowercase),
        so we only need to uppercase for consistency. Do NOT apply
        standardize_team_abbreviation() again as it would break the
        already-correct codes.

        Args:
            df: DataFrame with Team column

        Returns:
            DataFrame with uppercase team names
        """
        if 'Team' in df.columns:
            # Convert to uppercase for consistency (data is already correct)
            df['Team'] = df['Team'].str.upper()
        return df
    
    def collect_all_team_years(self) -> Optional[pd.DataFrame]:
        """
        Collect team-year combinations from coaching data (base table)

        Returns:
            DataFrame with team-year combinations from coaching data
        """
        # Use coaching data as the base - only include team-years with coaching data
        coaching_path = self.final_dir / "../processed/Coaching/yearly_coach_performance.csv"

        if not coaching_path.exists():
            self.logger.error(f"Coaching data not found: {coaching_path}")
            return None

        try:
            df = pd.read_csv(coaching_path)

            if 'Team' not in df.columns or 'Year' not in df.columns:
                self.logger.error("Coaching data missing Team/Year columns")
                return None

            # Standardize team names
            df['Team'] = df['Team'].str.upper()
            df['Year'] = df['Year'].astype(int)
            df = self.standardize_team_names(df)

            # Extract unique team-years
            team_years_df = df[['Team', 'Year']].drop_duplicates()
            team_years_df = team_years_df.sort_values(['Team', 'Year'], ignore_index=True)

            self.logger.info(f"Using coaching data as base: {len(team_years_df)} unique team-year combinations")
            self.logger.info(f"Team-Year range: {team_years_df['Year'].min()}-{team_years_df['Year'].max()}")
            self.logger.info(f"Teams: {team_years_df['Team'].nunique()}")

            return team_years_df

        except Exception as e:
            self.logger.error(f"Error loading coaching data: {e}")
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
                # Special handling for draft files that use Draft_Year
                elif 'Draft_Year' in df.columns:
                    df = df.rename(columns={'Draft_Year': 'Year'})
                else:
                    self.logger.warning(f"No Year column found in {filename}")
                    return None
            
            # Standardize data types
            df['Team'] = df['Team'].str.upper()
            df['Year'] = df['Year'].astype(int)
            
            # Apply historical team mappings
            df = self.standardize_team_names(df)
            
            # Filter for 1970 onwards to match coaching data availability
            df = df[df['Year'] >= 1970]
            
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
    
    def load_additional_dataset(self, file_path: Path, file_label: str) -> Optional[pd.DataFrame]:
        """
        Load an additional dataset file from another directory
        
        Args:
            file_path: Full path to the file
            file_label: Label for logging purposes
            
        Returns:
            Prepared DataFrame or None if failed
        """
        if not file_path.exists():
            self.logger.warning(f"Additional file not found: {file_path} ({file_label})")
            return None
        
        try:
            df = pd.read_csv(file_path)
            
            if df.empty:
                self.logger.warning(f"Empty additional file: {file_path} ({file_label})")
                return None
            
            # Special handling for coaching data
            if file_label == "coaching":
                # Check for required columns - coaching data uses Team column
                if 'Team' not in df.columns or 'Year' not in df.columns:
                    self.logger.warning(f"Missing Team/Year columns in {file_path} ({file_label})")
                    return None

                # Standardize Team column before merging
                df['Team'] = df['Team'].str.upper()

                # First, merge with coach-specific Win_Pct
                coach_win_pct_path = self.final_dir / self.coach_win_pct_file
                if coach_win_pct_path.exists():
                    try:
                        win_pct_df = pd.read_csv(coach_win_pct_path)
                        # Keep only Coach_Win_Pct column (merge on Coach and Year only)
                        win_pct_df = win_pct_df[['Coach', 'Year', 'Coach_Win_Pct']].copy()

                        # Merge with coaching data on Coach and Year (ignore Team to avoid abbreviation issues)
                        df = df.merge(win_pct_df, on=['Coach', 'Year'], how='left')
                        self.logger.info(f"Merged coach-specific Win_Pct with coaching data (on Coach + Year)")

                        # Check coverage
                        coverage = df['Coach_Win_Pct'].notna().sum()
                        self.logger.info(f"Coach_Win_Pct coverage: {coverage}/{len(df)} ({100*coverage/len(df):.1f}%)")
                    except Exception as e:
                        self.logger.warning(f"Failed to merge coach-specific Win_Pct: {e}")
                else:
                    self.logger.warning(f"Coach-specific Win_Pct file not found: {coach_win_pct_path}")

                # Drop non-feature columns (keep only coaching metrics and Win_Pct)
                coaching_exclude = ['Coach', 'Role', 'Age']
                cols_to_drop = [col for col in coaching_exclude if col in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    self.logger.info(f"Dropped coaching metadata columns: {cols_to_drop}")

                # Add _Norm suffix to normalized coaching features
                normalized_patterns = ['PF (Points For)', 'Yds__', 'Y/P__', 'TO__', '1stD__', 'Cmp Passing__',
                                     'Att Passing__', 'Yds Passing__', 'TD Passing__', 'Int Passing__',
                                     'NY/A Passing__', '1stD Passing__', 'Att Rushing__', 'Yds Rushing__',
                                     'TD Rushing__', 'Y/A Rushing__', '1stD Rushing__', 'Pen__',
                                     'Yds Penalties__', '1stPy__', '#Dr__', 'Sc%__', 'TO%__',
                                     'Time Average Drive__', 'Plays Average Drive__', 'Yds Average Drive__',
                                     'Pts Average Drive__', '3DAtt__', '3D%__', '4DAtt__', '4D%__',
                                     'RZAtt__', 'RZPct__']

                rename_dict = {}
                for col in df.columns:
                    if any(pattern in col for pattern in normalized_patterns):
                        if not col.endswith('_Norm'):  # Don't double-suffix
                            rename_dict[col] = col + '_Norm'

                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    self.logger.info(f"Added _Norm suffix to {len(rename_dict)} normalized coaching features")

            else:
                # Check for required columns for other additional files
                if 'Team' not in df.columns or 'Year' not in df.columns:
                    self.logger.warning(f"Missing Team/Year columns in {file_path} ({file_label})")
                    return None
            
            # Standardize data types
            df['Team'] = df['Team'].str.upper()
            df['Year'] = df['Year'].astype(int)
            
            # Apply historical team mappings
            df = self.standardize_team_names(df)
            
            # Filter for 1970 onwards to match coaching data availability
            df = df[df['Year'] >= 1970]
            
            # Remove excluded columns
            cols_to_drop = [col for col in self.exclude_columns if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self.logger.info(f"Dropped columns from {file_label}: {cols_to_drop}")
            
            self.logger.info(f"Loaded {file_label}: {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading additional file {file_path} ({file_label}): {e}")
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
        Combine all datasets using full outer join approach
        
        Returns:
            Combined DataFrame or None if failed
        """
        # Get all unique team-year combinations
        combined_df = self.collect_all_team_years()
        if combined_df is None:
            return None
        
        # Create team-year key
        combined_df = self.create_team_year_key(combined_df)
        
        # Join each dataset
        successful_joins = []
        failed_joins = []
        
        # Process main files from final directory
        for filename in self.files_to_combine:
            self.logger.info(f"Processing {filename}")
            
            # Load dataset
            df = self.load_dataset(filename)
            if df is None:
                failed_joins.append(filename)
                continue
            
            # Create team-year key
            df = self.create_team_year_key(df)
            
            # Drop duplicate team_year_key, Team, Year columns for joining
            join_cols = [col for col in df.columns 
                        if col not in ['team_year_key', 'Team', 'Year']]
            df_to_join = df[['team_year_key'] + join_cols]
            
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
            
            # Perform left join to keep only base team-years (coaching data)
            before_cols = len(combined_df.columns)
            combined_df = combined_df.merge(
                df_to_join,
                on='team_year_key',
                how='left'
            )
            after_cols = len(combined_df.columns)
            
            added_cols = after_cols - before_cols
            overlap = len(df)
            
            self.logger.info(f"Joined {filename}: +{added_cols} columns, {overlap} total rows in dataset")
            successful_joins.append(filename)
        
        # Process additional files from other directories
        for file_path, file_label in self.additional_files.items():
            self.logger.info(f"Processing additional file: {file_label}")
            
            # Load additional dataset
            full_path = self.final_dir / file_path
            df = self.load_additional_dataset(full_path, file_label)
            if df is None:
                failed_joins.append(file_label)
                continue
            
            # Create team-year key
            df = self.create_team_year_key(df)
            
            # Drop duplicate team_year_key, Team, Year columns for joining
            join_cols = [col for col in df.columns 
                        if col not in ['team_year_key', 'Team', 'Year']]
            df_to_join = df[['team_year_key'] + join_cols]
            
            # Check for column name conflicts
            existing_cols = set(combined_df.columns)
            new_cols = set(join_cols)
            conflicts = existing_cols.intersection(new_cols)
            
            if conflicts:
                self.logger.warning(f"Column conflicts in {file_label}: {conflicts}")
                # Add suffix to conflicting columns
                suffix = f"_{file_label}"
                rename_dict = {col: col + suffix for col in conflicts}
                df_to_join = df_to_join.rename(columns=rename_dict)
                self.logger.info(f"Renamed columns: {rename_dict}")
            
            # Use left join for all additional files (keeps only coaching data team-years)
            join_type = 'left'
            self.logger.info(f"Using LEFT JOIN for {file_label} (only coaching team-years)")

            # Perform join
            before_cols = len(combined_df.columns)
            before_rows = len(combined_df)
            combined_df = combined_df.merge(
                df_to_join,
                on='team_year_key',
                how=join_type
            )
            after_cols = len(combined_df.columns)
            after_rows = len(combined_df)
            
            added_cols = after_cols - before_cols
            overlap = len(df)
            
            self.logger.info(f"Joined {file_label}: +{added_cols} columns, {overlap} source rows, {before_rows}→{after_rows} result rows")
            successful_joins.append(file_label)
        
        # Remove team_year_key helper column
        combined_df = combined_df.drop(columns=['team_year_key'])
        
        # Final filter: ensure Year is integer and >= 1970
        combined_df['Year'] = combined_df['Year'].astype(int)
        combined_df = combined_df[combined_df['Year'] >= 1970]

        # Replace team-level Win_Pct with coach-specific Win_Pct
        if 'Coach_Win_Pct' in combined_df.columns:
            if 'Win_Pct' in combined_df.columns:
                # Drop old team-level Win_Pct
                combined_df = combined_df.drop(columns=['Win_Pct'])
                self.logger.info("Dropped team-level Win_Pct (will use coach-specific instead)")

            # Rename Coach_Win_Pct to Win_Pct
            combined_df = combined_df.rename(columns={'Coach_Win_Pct': 'Win_Pct'})
            self.logger.info("Renamed Coach_Win_Pct to Win_Pct (now using coach-specific win percentages)")

        # Move Win_Pct to last column if it exists
        if 'Win_Pct' in combined_df.columns:
            win_pct_col = combined_df['Win_Pct']
            combined_df = combined_df.drop(columns=['Win_Pct'])
            combined_df['Win_Pct'] = win_pct_col
            self.logger.info("Moved Win_Pct to last column")
        
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
                'Join_Type': 'full_outer_join',
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
            'penalties': [col for col in df.columns if ('Team_Pen' in col or 'Opp_Pen' in col or 'Team_Int_Passing' in col or 'Opp_Int_Passing' in col or 'Team_Yds_Penalties' in col or 'Opp_Yds_Penalties' in col)],
            'coaching': [col for col in df.columns if ('_Norm' in col and ('__oc' in col or '__dc' in col or '__hc' in col)) or any(metric in col for metric in ['num_times_hc', 'num_yr_col', 'num_yr_nfl']) or col.startswith('num_')],
            'draft': [col for col in df.columns if 'Round_' in col and 'Picks' in col]
        }
        
        for category, cols in categories.items():
            if cols:
                available_cols = [col for col in cols if col in df.columns]
                if available_cols:
                    # For coaching data, count any row with at least one non-null value
                    if category == 'coaching':
                        # Count rows where at least one coaching column has data
                        non_null_counts = df[available_cols].notna().sum(axis=1)
                        complete_rows = (non_null_counts > 0).sum()
                    elif len(available_cols) > 50:  # Other large categories
                        # Count rows where at least 50% of columns are non-null
                        non_null_counts = df[available_cols].notna().sum(axis=1)
                        threshold = len(available_cols) * 0.5
                        complete_rows = (non_null_counts >= threshold).sum()
                    else:
                        # For smaller categories, require all columns to be non-null
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
    print(f"Join strategy: Using coaching data as base (only team-years with coaching data)")
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