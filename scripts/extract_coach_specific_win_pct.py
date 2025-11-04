#!/usr/bin/env python3
"""
Extract Coach-Specific Win Percentage from Coaching Results

This script extracts coach-specific W-L records from all_coaching_results.csv files
and calculates Win_Pct for each coach's stint. It only includes coaches that are
already in the current coaching performance data (respecting interim coach filtering).

Usage:
    python scripts/extract_coach_specific_win_pct.py
    python scripts/extract_coach_specific_win_pct.py --output-dir data/processed/Coaching
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional
import glob

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import standardize_team_abbreviation

class CoachSpecificWinPctExtractor:
    """Extracts coach-specific win percentages from coaching results files"""

    def __init__(self, coaches_dir: str = "data/raw/Coaches",
                 coaching_data_path: str = "data/processed/Coaching/yearly_coach_performance.csv",
                 output_dir: str = "data/processed/Coaching"):
        """
        Initialize the extractor

        Args:
            coaches_dir: Directory containing coach subdirectories
            coaching_data_path: Path to current coaching performance data
            output_dir: Directory to save extracted Win_Pct data
        """
        self.coaches_dir = Path(coaches_dir)
        self.coaching_data_path = Path(coaching_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_current_coaching_data(self) -> pd.DataFrame:
        """
        Load the current coaching performance data to use as filter

        Returns:
            DataFrame with current coaching data
        """
        self.logger.info(f"Loading current coaching data from {self.coaching_data_path}")

        if not self.coaching_data_path.exists():
            raise FileNotFoundError(f"Coaching data not found: {self.coaching_data_path}")

        df = pd.read_csv(self.coaching_data_path)

        # Standardize team abbreviations to uppercase
        df['Team'] = df['Team'].str.upper()

        self.logger.info(f"Loaded {len(df)} coaching records")
        self.logger.info(f"  Unique coaches: {df['Coach'].nunique()}")
        self.logger.info(f"  Year range: {df['Year'].min()}-{df['Year'].max()}")

        return df

    def load_all_coaching_results(self) -> pd.DataFrame:
        """
        Load all coaching results from individual coach files

        Returns:
            DataFrame with all coaching results
        """
        self.logger.info(f"Loading coaching results from {self.coaches_dir}")

        all_results = []
        coach_files = list(self.coaches_dir.glob('*/all_coaching_results.csv'))

        self.logger.info(f"Found {len(coach_files)} coach result files")

        for file_path in coach_files:
            coach_name = file_path.parent.name
            try:
                df = pd.read_csv(file_path)
                df['Coach'] = coach_name

                # Keep only relevant columns
                required_cols = ['Coach', 'Year', 'Tm', 'G', 'W', 'L', 'T']
                if not all(col in df.columns for col in required_cols):
                    self.logger.warning(f"Missing columns in {coach_name}, skipping")
                    continue

                df = df[required_cols + ['W-L%']].copy()
                all_results.append(df)

            except Exception as e:
                self.logger.warning(f"Error reading {coach_name}: {e}")
                continue

        if not all_results:
            raise ValueError("No coaching results loaded")

        combined = pd.concat(all_results, ignore_index=True)
        self.logger.info(f"Loaded {len(combined)} total coaching records")

        return combined

    def standardize_team_abbreviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize team abbreviations to PFR format with year-based logic

        Args:
            df: DataFrame with 'Tm' and 'Year' columns

        Returns:
            DataFrame with standardized 'Team' column
        """
        df = df.copy()

        # Apply centralized standardize_team_abbreviation function
        df['Team'] = df.apply(
            lambda row: standardize_team_abbreviation(row['Tm'], row['Year']),
            axis=1
        )

        # Convert to uppercase for consistency with coaching data
        df['Team'] = df['Team'].str.upper()

        return df

    def calculate_coach_win_pct(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate coach-specific win percentage

        Args:
            df: DataFrame with W, L, T, G columns

        Returns:
            DataFrame with Coach_Win_Pct column added
        """
        df = df.copy()

        # Calculate Win_Pct: (W + 0.5*T) / G
        df['Coach_Win_Pct'] = (df['W'] + 0.5 * df['T']) / df['G']

        # Round to 4 decimal places for consistency
        df['Coach_Win_Pct'] = df['Coach_Win_Pct'].round(4)

        return df

    def merge_with_current_data(self, results: pd.DataFrame,
                                current: pd.DataFrame) -> pd.DataFrame:
        """
        Merge coaching results with current data to filter out interim coaches

        Args:
            results: DataFrame with coaching results
            current: DataFrame with current coaching data

        Returns:
            DataFrame with merged data (only coaches in current data)
        """
        self.logger.info("Merging coaching results with current data...")

        # Merge on Coach, Team, Year to keep only coaches in current data
        merged = current[['Coach', 'Team', 'Year']].merge(
            results[['Coach', 'Team', 'Year', 'G', 'W', 'L', 'T', 'Coach_Win_Pct']],
            on=['Coach', 'Team', 'Year'],
            how='left',
            indicator=True
        )

        # Report matching statistics
        matched = (merged['_merge'] == 'both').sum()
        not_matched = (merged['_merge'] == 'left_only').sum()

        self.logger.info(f"Merge results:")
        self.logger.info(f"  Matched: {matched} records ({100*matched/len(merged):.1f}%)")
        self.logger.info(f"  Not matched: {not_matched} records ({100*not_matched/len(merged):.1f}%)")

        if not_matched > 0:
            self.logger.warning(f"\n{not_matched} coaching records could not be matched to results files")
            self.logger.warning("These will have null Coach_Win_Pct values")

            # Show sample of unmatched
            unmatched_sample = merged[merged['_merge'] == 'left_only'][['Coach', 'Team', 'Year']].head(10)
            self.logger.warning("Sample unmatched records:")
            for _, row in unmatched_sample.iterrows():
                self.logger.warning(f"  {row['Coach']} - {row['Team']} {int(row['Year'])}")

        # Drop the merge indicator
        merged = merged.drop('_merge', axis=1)

        return merged

    def validate_results(self, df: pd.DataFrame) -> None:
        """
        Validate the extracted Win_Pct data

        Args:
            df: DataFrame with coach-specific Win_Pct
        """
        self.logger.info("\nValidating results...")

        # Check for missing Win_Pct
        missing = df['Coach_Win_Pct'].isnull().sum()
        if missing > 0:
            self.logger.warning(f"Missing Coach_Win_Pct: {missing} records ({100*missing/len(df):.1f}%)")

        # Check Win_Pct range
        valid_pct = df['Coach_Win_Pct'].dropna()
        if len(valid_pct) > 0:
            min_pct = valid_pct.min()
            max_pct = valid_pct.max()

            if min_pct < 0 or max_pct > 1:
                self.logger.error(f"Invalid Win_Pct range: {min_pct:.4f} to {max_pct:.4f}")
            else:
                self.logger.info(f"Win_Pct range: {min_pct:.4f} to {max_pct:.4f}")

        # Check for duplicates
        dupes = df.duplicated(['Coach', 'Team', 'Year']).sum()
        if dupes > 0:
            self.logger.error(f"Duplicate Coach-Team-Year combinations: {dupes}")
        else:
            self.logger.info("No duplicate Coach-Team-Year combinations")

        # Check mid-season changes
        team_year_counts = df.groupby(['Team', 'Year']).size()
        mid_season = (team_year_counts > 1).sum()
        self.logger.info(f"Team-years with mid-season changes: {mid_season} ({100*mid_season/len(team_year_counts):.1f}%)")

        # Show examples of mid-season changes
        if mid_season > 0:
            self.logger.info("\nSample mid-season changes:")
            mid_season_teams = team_year_counts[team_year_counts > 1].head(5)
            for (team, year), count in mid_season_teams.items():
                coaches = df[(df['Team'] == team) & (df['Year'] == year)]
                self.logger.info(f"  {team} {int(year)}: {count} coaches")
                for _, row in coaches.iterrows():
                    if pd.notna(row['Coach_Win_Pct']):
                        self.logger.info(f"    - {row['Coach']}: {row['Coach_Win_Pct']:.3f} ({int(row['G'])} games)")

    def save_results(self, df: pd.DataFrame) -> str:
        """
        Save extracted Win_Pct data

        Args:
            df: DataFrame with coach-specific Win_Pct

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / "coach_specific_win_pct.csv"

        # Select columns to save
        output_cols = ['Coach', 'Team', 'Year', 'G', 'W', 'L', 'T', 'Coach_Win_Pct']
        df_output = df[output_cols].copy()

        df_output.to_csv(output_path, index=False)
        self.logger.info(f"\nSaved {len(df_output)} records to: {output_path}")

        return str(output_path)

    def run(self) -> str:
        """
        Run the full extraction process

        Returns:
            Path to output file
        """
        self.logger.info("="*80)
        self.logger.info("COACH-SPECIFIC WIN PERCENTAGE EXTRACTION")
        self.logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)

        # Load current coaching data (serves as filter)
        current = self.load_current_coaching_data()

        # Load all coaching results
        results = self.load_all_coaching_results()

        # Standardize team abbreviations
        results = self.standardize_team_abbreviations(results)

        # Calculate coach-specific Win_Pct
        results = self.calculate_coach_win_pct(results)

        # Merge with current data to filter out interim coaches
        merged = self.merge_with_current_data(results, current)

        # Validate results
        self.validate_results(merged)

        # Save results
        output_path = self.save_results(merged)

        self.logger.info("\n" + "="*80)
        self.logger.info(f"EXTRACTION COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*80)

        return output_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Extract coach-specific win percentages from coaching results files"
    )
    parser.add_argument(
        '--coaches-dir',
        default='data/raw/Coaches',
        help='Directory containing coach subdirectories (default: data/raw/Coaches)'
    )
    parser.add_argument(
        '--coaching-data',
        default='data/processed/Coaching/yearly_coach_performance.csv',
        help='Path to current coaching performance data (default: data/processed/Coaching/yearly_coach_performance.csv)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed/Coaching',
        help='Directory to save output (default: data/processed/Coaching)'
    )

    args = parser.parse_args()

    # Create extractor and run
    extractor = CoachSpecificWinPctExtractor(
        coaches_dir=args.coaches_dir,
        coaching_data_path=args.coaching_data,
        output_dir=args.output_dir
    )

    output_path = extractor.run()
    print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    main()
