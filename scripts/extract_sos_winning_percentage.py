#!/usr/bin/env python3
"""
Extract Strength of Schedule (SoS) and Calculate Winning Percentage from Team Records

This script extracts the SoS metric and calculates winning percentage from team record files.
Winning percentage is calculated as: (W + 0.5*T) / (W + T + L)

Usage:
    python scripts/extract_sos_winning_percentage.py
    python scripts/extract_sos_winning_percentage.py --team crd
    python scripts/extract_sos_winning_percentage.py --all-teams
    python scripts/extract_sos_winning_percentage.py --all-teams --min-year 2000 --max-year 2024
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
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS

# Create PFR team abbreviations list from the corrected mappings
PFR_TEAM_ABBREVIATIONS = list(set(SPOTRAC_TO_PFR_MAPPINGS.values()))

class SoSWinPctExtractor:
    """Extracts SoS and calculates winning percentage from team records"""
    
    def __init__(self, teams_dir: str = "data/raw/Teams", 
                 output_dir: str = "data/final"):
        """
        Initialize the extractor
        
        Args:
            teams_dir: Directory containing team data subdirectories
            output_dir: Directory to save extracted metrics
        """
        self.teams_dir = Path(teams_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_available_teams(self) -> List[str]:
        """
        Get list of available teams in the teams directory
        
        Returns:
            List of team abbreviations with data available
        """
        teams = []
        if self.teams_dir.exists():
            for team_dir in self.teams_dir.iterdir():
                if team_dir.is_dir():
                    team_abbr = team_dir.name.lower()
                    # Check if team_record.csv exists
                    record_file = team_dir / "team_record.csv"
                    if record_file.exists():
                        teams.append(team_abbr)
        return sorted(teams)
    
    def calculate_winning_percentage(self, wins: float, losses: float, ties: float) -> float:
        """
        Calculate winning percentage using the formula: (W + 0.5*T) / (W + T + L)
        
        Args:
            wins: Number of wins
            losses: Number of losses
            ties: Number of ties
            
        Returns:
            Winning percentage as a decimal
        """
        total_games = wins + losses + ties
        if total_games == 0:
            return 0.0
        
        # Calculate winning percentage
        win_pct = (wins + 0.5 * ties) / total_games
        return round(win_pct, 4)
    
    def load_team_record(self, team: str) -> Optional[pd.DataFrame]:
        """
        Load team record file for a specific team
        
        Args:
            team: Team abbreviation
            
        Returns:
            DataFrame with team record data or None if not found
        """
        team_upper = team.upper()
        record_file = self.teams_dir / team_upper / "team_record.csv"
        
        if not record_file.exists():
            self.logger.warning(f"Team record file not found: {record_file}")
            return None
        
        try:
            df = pd.read_csv(record_file)
            
            # Check for required columns
            required_cols = ['Year', 'W', 'L']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Missing required column '{col}' in {record_file}")
                    return None
            
            # Add team abbreviation
            df['Team'] = team.upper()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {record_file}: {e}")
            return None
    
    def extract_team_metrics(self, team: str, min_year: Optional[int] = None, 
                            max_year: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Extract SoS and calculate winning percentage for a team
        
        Args:
            team: Team abbreviation
            min_year: Minimum year to include (inclusive)
            max_year: Maximum year to include (inclusive)
            
        Returns:
            DataFrame with extracted metrics or None if failed
        """
        # Load team record
        df = self.load_team_record(team)
        if df is None:
            return None
        
        # Filter by year range if specified
        if min_year is not None:
            df = df[df['Year'] >= min_year]
        if max_year is not None:
            df = df[df['Year'] <= max_year]
        
        if df.empty:
            self.logger.warning(f"No data for {team} in specified year range")
            return None
        
        # Extract metrics
        results = []
        for idx, row in df.iterrows():
            # Get basic values
            year = row['Year']
            wins = row['W'] if pd.notna(row['W']) else 0
            losses = row['L'] if pd.notna(row['L']) else 0
            ties = row.get('T', 0)  # Ties might not exist in all years
            if pd.isna(ties):
                ties = 0
            
            # Get SoS (might be missing for some years)
            sos = row.get('SoS', np.nan)
            if pd.notna(sos):
                try:
                    sos = float(sos)
                except:
                    sos = np.nan
            
            # Calculate winning percentage
            win_pct = self.calculate_winning_percentage(wins, losses, ties)
            
            # Create result row with only required columns in specified order
            result = {
                'Team': team.upper(),
                'Year': int(year),
                'SoS': round(sos, 2) if pd.notna(sos) else np.nan,
                'Win_Pct': win_pct,
                'Extraction_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results.append(result)
        
        if not results:
            return None
        
        result_df = pd.DataFrame(results)
        self.logger.info(f"Extracted metrics for {team}: {len(result_df)} years")
        
        return result_df
    
    def extract_all_teams(self, teams: Optional[List[str]] = None,
                         min_year: Optional[int] = None, 
                         max_year: Optional[int] = None) -> pd.DataFrame:
        """
        Extract metrics for all specified teams
        
        Args:
            teams: List of team abbreviations, None for all available
            min_year: Minimum year to include (inclusive)
            max_year: Maximum year to include (inclusive)
            
        Returns:
            Combined DataFrame with all extracted metrics
        """
        if teams is None:
            teams = self.get_available_teams()
        
        self.logger.info(f"Processing {len(teams)} teams")
        if min_year or max_year:
            self.logger.info(f"Year range: {min_year or 'earliest'} to {max_year or 'latest'}")
        
        # Extract data for each team
        all_data = []
        successful_teams = []
        
        for team in teams:
            self.logger.info(f"Processing team: {team}")
            team_data = self.extract_team_metrics(team, min_year, max_year)
            
            if team_data is not None:
                all_data.append(team_data)
                successful_teams.append(team)
            else:
                self.logger.warning(f"No data extracted for {team}")
        
        if not all_data:
            self.logger.warning("No data extracted for any team")
            return pd.DataFrame()
        
        # Combine all teams
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Sort by team and year
        combined_df = combined_df.sort_values(['Team', 'Year'], ignore_index=True)
        
        self.logger.info(f"Successfully extracted data for {len(successful_teams)} teams")
        
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
            output_file = self.output_dir / "sos_winning_percentage.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved metrics to {output_file}")
            
            # Calculate summary statistics
            teams_processed = df['Team'].nunique()
            year_range = f"{df['Year'].min()}-{df['Year'].max()}"
            total_rows = len(df)
            
            # Calculate averages
            avg_win_pct = df['Win_Pct'].mean()
            avg_sos = df['SoS'].mean()
            sos_coverage = df['SoS'].notna().sum() / len(df) * 100
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Rows': total_rows,
                'Teams_Processed': teams_processed,
                'Year_Range': year_range,
                'Avg_Win_Pct': round(avg_win_pct, 4),
                'Avg_SoS': round(avg_sos, 2) if pd.notna(avg_sos) else 'N/A',
                'SoS_Coverage_Pct': round(sos_coverage, 1),
                'Description': 'Strength of Schedule and Winning Percentage extracted from team records'
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "sos_winning_percentage_metadata.csv"
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
            'avg_win_pct': df['Win_Pct'].mean(),
            'sos_coverage': df['SoS'].notna().sum() / len(df) * 100
        }
        
        if 'SoS' in df.columns:
            sos_data = df['SoS'].dropna()
            if len(sos_data) > 0:
                stats['avg_sos'] = sos_data.mean()
                stats['min_sos'] = sos_data.min()
                stats['max_sos'] = sos_data.max()
        
        return stats


def main():
    """Main function to run SoS and winning percentage extraction"""
    parser = argparse.ArgumentParser(
        description='Extract Strength of Schedule and calculate winning percentage from team records'
    )
    parser.add_argument(
        '--team',
        type=str,
        help='Team abbreviation (e.g., "crd")'
    )
    parser.add_argument(
        '--all-teams',
        action='store_true',
        help='Process all available teams'
    )
    parser.add_argument(
        '--min-year',
        type=int,
        help='Minimum year to include (inclusive)'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        help='Maximum year to include (inclusive)'
    )
    parser.add_argument(
        '--teams-dir',
        type=str,
        default='data/raw/Teams',
        help='Directory containing team data (default: data/raw/Teams)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for extracted metrics (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SoSWinPctExtractor(
        teams_dir=args.teams_dir,
        output_dir=args.output_dir
    )
    
    # Determine teams to process
    if args.all_teams:
        teams = None  # Will use all available teams
        print("Processing all available teams...")
    elif args.team:
        teams = [args.team.lower()]
        print(f"Processing team: {args.team}")
    else:
        # Default to all teams if no specific option given
        teams = None
        print("No team specified, processing all available teams...")
    
    # Get available teams for reporting
    available_teams = extractor.get_available_teams()
    if not available_teams:
        print("Error: No team data found")
        sys.exit(1)
    
    print(f"Found {len(available_teams)} teams with data")
    
    # Extract metrics
    print(f"Extracting SoS and calculating winning percentage...")
    metrics_df = extractor.extract_all_teams(
        teams=teams,
        min_year=args.min_year,
        max_year=args.max_year
    )
    
    if metrics_df.empty:
        print("Error: No metrics extracted")
        sys.exit(1)
    
    # Save metrics
    if extractor.save_metrics(metrics_df):
        print(f"\nSuccessfully extracted metrics for {len(metrics_df)} team-years")
        print(f"Results saved to: {args.output_dir}")
        print("Generated files:")
        print("  - sos_winning_percentage.csv (main dataset)")
        print("  - sos_winning_percentage_metadata.csv (metadata)")
        
        # Display summary statistics
        stats = extractor.get_summary_stats(metrics_df)
        print(f"\nSummary Statistics:")
        print(f"  - Total rows: {stats['total_rows']}")
        print(f"  - Unique teams: {stats['unique_teams']}")
        print(f"  - Year range: {stats['year_range']}")
        print(f"  - Average winning percentage: {stats['avg_win_pct']:.3f}")
        
        if 'avg_sos' in stats:
            print(f"  - SoS coverage: {stats['sos_coverage']:.1f}%")
            print(f"  - Average SoS: {stats['avg_sos']:.2f}")
            print(f"  - SoS range: {stats['min_sos']:.2f} to {stats['max_sos']:.2f}")
    else:
        print("Error: Failed to save metrics")
        sys.exit(1)


if __name__ == "__main__":
    main()