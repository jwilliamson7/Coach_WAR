#!/usr/bin/env python3
"""
Extract Penalty and Interception Metrics from Normalized League Data

This script extracts Int Passing, Pen (Penalties), and Yds Penalties metrics
from the normalized league data files for both team and opponent data.

Usage:
    python scripts/extract_penalty_interception_metrics.py
    python scripts/extract_penalty_interception_metrics.py --start-year 2015 --end-year 2023
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

class PenaltyInterceptionExtractor:
    """Extracts penalty and interception metrics from normalized league data"""
    
    def __init__(self, league_data_dir: str = "data/processed/League Data", 
                 output_dir: str = "data/final"):
        """
        Initialize the extractor
        
        Args:
            league_data_dir: Directory containing league data by year
            output_dir: Directory to save extracted metrics
        """
        self.league_data_dir = Path(league_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Metrics to extract
        self.metrics = ['Int Passing', 'Pen', 'Yds Penalties']
        
        # Column names for output
        self.output_columns = [
            'Team', 'Year',
            'Team_Int_Passing_Norm', 'Team_Pen_Norm', 'Team_Yds_Penalties_Norm',
            'Opp_Int_Passing_Norm', 'Opp_Pen_Norm', 'Opp_Yds_Penalties_Norm'
        ]
    
    def get_available_years(self) -> List[int]:
        """
        Get list of available years in the league data directory
        
        Returns:
            List of years with data available
        """
        years = []
        if self.league_data_dir.exists():
            for year_dir in self.league_data_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    years.append(int(year_dir.name))
        return sorted(years)
    
    def load_year_data(self, year: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load team and opponent normalized data for a specific year
        
        Args:
            year: Year to load data for
            
        Returns:
            Tuple of (team_data, opponent_data) DataFrames or (None, None) if not found
        """
        year_dir = self.league_data_dir / str(year)
        
        team_file = year_dir / "league_team_data_normalized.csv"
        opp_file = year_dir / "league_opponent_data_normalized.csv"
        
        if not team_file.exists() or not opp_file.exists():
            self.logger.warning(f"Missing data files for year {year}")
            return None, None
        
        try:
            team_data = pd.read_csv(team_file)
            opp_data = pd.read_csv(opp_file)
            
            # Check for required columns
            required_cols = ['Team Abbreviation'] + self.metrics
            
            for col in required_cols:
                if col not in team_data.columns:
                    self.logger.error(f"Missing column '{col}' in team data for {year}")
                    return None, None
                if col not in opp_data.columns:
                    self.logger.error(f"Missing column '{col}' in opponent data for {year}")
                    return None, None
            
            return team_data, opp_data
            
        except Exception as e:
            self.logger.error(f"Error loading data for year {year}: {e}")
            return None, None
    
    def extract_year_metrics(self, year: int) -> Optional[pd.DataFrame]:
        """
        Extract metrics for a specific year
        
        Args:
            year: Year to extract metrics for
            
        Returns:
            DataFrame with extracted metrics or None if failed
        """
        # Load data
        team_data, opp_data = self.load_year_data(year)
        if team_data is None or opp_data is None:
            return None
        
        # Create result dataframe
        results = []
        
        # Process each team
        for idx, team_row in team_data.iterrows():
            team_abbr = team_row['Team Abbreviation']
            
            # Find corresponding opponent data
            opp_row = opp_data[opp_data['Team Abbreviation'] == team_abbr]
            
            if opp_row.empty:
                self.logger.warning(f"No opponent data found for {team_abbr} in {year}")
                continue
            
            opp_row = opp_row.iloc[0]
            
            # Extract metrics
            result = {
                'Team': team_abbr.upper(),
                'Year': year,
                'Team_Int_Passing_Norm': round(team_row['Int Passing'], 4),
                'Team_Pen_Norm': round(team_row['Pen'], 4),
                'Team_Yds_Penalties_Norm': round(team_row['Yds Penalties'], 4),
                'Opp_Int_Passing_Norm': round(opp_row['Int Passing'], 4),
                'Opp_Pen_Norm': round(opp_row['Pen'], 4),
                'Opp_Yds_Penalties_Norm': round(opp_row['Yds Penalties'], 4)
            }
            
            results.append(result)
        
        if not results:
            self.logger.warning(f"No data extracted for year {year}")
            return None
        
        df = pd.DataFrame(results)
        self.logger.info(f"Extracted metrics for {len(df)} teams in {year}")
        
        return df
    
    def extract_all_years(self, start_year: Optional[int] = None, 
                         end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Extract metrics for all available years or specified range
        
        Args:
            start_year: Starting year (inclusive), None for earliest available
            end_year: Ending year (inclusive), None for latest available
            
        Returns:
            Combined DataFrame with all extracted metrics
        """
        available_years = self.get_available_years()
        
        if not available_years:
            self.logger.error("No league data found")
            return pd.DataFrame()
        
        # Determine year range
        if start_year is None:
            start_year = min(available_years)
        if end_year is None:
            end_year = max(available_years)
        
        # Filter years to process
        years_to_process = [y for y in available_years if start_year <= y <= end_year]
        
        self.logger.info(f"Processing years {start_year} to {end_year}")
        self.logger.info(f"Found {len(years_to_process)} years to process")
        
        # Extract data for each year
        all_data = []
        for year in years_to_process:
            self.logger.info(f"Processing year {year}")
            year_data = self.extract_year_metrics(year)
            
            if year_data is not None:
                all_data.append(year_data)
        
        if not all_data:
            self.logger.warning("No data extracted")
            return pd.DataFrame()
        
        # Combine all years
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by team and year
        combined_df = combined_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        # Add metadata column
        combined_df['Extraction_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return combined_df
    
    def save_metrics(self, df: pd.DataFrame) -> bool:
        """
        Save extracted metrics to CSV
        
        Args:
            df: DataFrame with extracted metrics
            
        Returns:
            True if saved successfully, False otherwise
        """
        if df.empty:
            self.logger.warning("No data to save")
            return False
        
        try:
            # Save main data file
            output_file = self.output_dir / "penalty_interception_metrics.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved metrics to {output_file}")
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Rows': len(df),
                'Teams': df['Team'].nunique(),
                'Years': df['Year'].nunique(),
                'Year_Range': f"{df['Year'].min()}-{df['Year'].max()}",
                'Metrics_Extracted': ', '.join(self.metrics),
                'Source_Files': 'league_team_data_normalized.csv, league_opponent_data_normalized.csv',
                'Description': 'Normalized penalty and interception metrics for teams and opponents'
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "penalty_interception_metrics_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {e}")
            return False
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the extracted metrics
        
        Args:
            df: DataFrame with extracted metrics
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        stats = {
            'total_rows': len(df),
            'unique_teams': df['Team'].nunique(),
            'unique_years': df['Year'].nunique(),
            'year_range': f"{df['Year'].min()}-{df['Year'].max()}",
        }
        
        # Calculate means for each metric
        for col in df.columns:
            if col.endswith('_Norm'):
                stats[f'{col}_mean'] = round(df[col].mean(), 4)
                stats[f'{col}_std'] = round(df[col].std(), 4)
        
        return stats


def main():
    """Main function to run penalty and interception metrics extraction"""
    parser = argparse.ArgumentParser(
        description='Extract penalty and interception metrics from normalized league data'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting year for extraction (inclusive)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Ending year for extraction (inclusive)'
    )
    parser.add_argument(
        '--league-data-dir',
        type=str,
        default='data/processed/League Data',
        help='Directory containing league data (default: data/processed/League Data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for extracted metrics (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = PenaltyInterceptionExtractor(
        league_data_dir=args.league_data_dir,
        output_dir=args.output_dir
    )
    
    # Check available years
    available_years = extractor.get_available_years()
    if not available_years:
        print("Error: No league data found")
        sys.exit(1)
    
    print(f"Found league data for years: {min(available_years)}-{max(available_years)}")
    
    # Extract metrics
    print(f"Extracting penalty and interception metrics...")
    metrics_df = extractor.extract_all_years(
        start_year=args.start_year,
        end_year=args.end_year
    )
    
    if metrics_df.empty:
        print("Error: No metrics extracted")
        sys.exit(1)
    
    # Save metrics
    if extractor.save_metrics(metrics_df):
        print(f"\nSuccessfully extracted metrics for {len(metrics_df)} team-years")
        print(f"Results saved to: {args.output_dir}")
        print("Generated files:")
        print("  - penalty_interception_metrics.csv (main dataset)")
        print("  - penalty_interception_metrics_metadata.csv (metadata)")
        
        # Display summary statistics
        stats = extractor.get_summary_stats(metrics_df)
        print(f"\nSummary Statistics:")
        print(f"  - Total rows: {stats['total_rows']}")
        print(f"  - Unique teams: {stats['unique_teams']}")
        print(f"  - Year range: {stats['year_range']}")
        print(f"\nMetric Averages (normalized values should be ~0):")
        for metric in ['Team_Int_Passing_Norm', 'Team_Pen_Norm', 'Team_Yds_Penalties_Norm']:
            if f'{metric}_mean' in stats:
                print(f"  - {metric}: {stats[f'{metric}_mean']:.4f} (std: {stats[f'{metric}_std']:.4f})")
    else:
        print("Error: Failed to save metrics")
        sys.exit(1)


if __name__ == "__main__":
    main()