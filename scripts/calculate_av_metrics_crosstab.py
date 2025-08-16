#!/usr/bin/env python3
"""
AV (Approximate Value) Metrics Crosstab Analysis Script

This script calculates AV (Approximate Value) metrics for both starters and full roster data,
comparing the performance value between starting players and the full team roster.

Usage:
    python scripts/calculate_av_metrics_crosstab.py --team crd --year 2011
    python scripts/calculate_av_metrics_crosstab.py --all-teams --year all --minyear 2015 --maxyear 2024
    python scripts/calculate_av_metrics_crosstab.py --all-teams --year 2024
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
import glob

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS

# Create PFR team abbreviations list from the corrected mappings
PFR_TEAM_ABBREVIATIONS = list(set(SPOTRAC_TO_PFR_MAPPINGS.values()))

class AVMetricsCrosstabAnalyzer:
    """Analyzes AV metrics for starters vs full roster in crosstab format"""
    
    def __init__(self, starters_dir: str = "data/raw/Starters", 
                 roster_dir: str = "data/raw/Rosters", 
                 output_dir: str = "data/final"):
        """
        Initialize the analyzer
        
        Args:
            starters_dir: Directory containing starters CSV files
            roster_dir: Directory containing roster CSV files
            output_dir: Directory to save analysis results
        """
        self.starters_dir = Path(starters_dir)
        self.roster_dir = Path(roster_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Position groups for analysis
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB', 'HB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['LT', 'LG', 'C', 'RG', 'RT', 'G', 'T', 'OL', 'OG', 'OT'],
            'DL': ['LDE', 'RDE', 'DE', 'NT', 'DT', 'LDT', 'RDT', 'DL'],
            'LB': ['LOLB', 'LILB', 'RILB', 'ROLB', 'LB', 'ILB', 'OLB', 'MLB'],
            'CB': ['LCB', 'RCB', 'CB', 'DB'],  # DB often means cornerback in roster
            'S': ['SS', 'FS', 'S', 'SAF']
        }
        
        # Base metrics we'll calculate
        self.base_metrics = [
            'Avg_Starter_AV',
            'StdDev_Starter_AV',
            'Avg_Roster_AV',
            'StdDev_Roster_AV'
        ]
        
        # Position group metrics for both starters and roster
        self.position_metrics = []
        for pos_group in self.position_groups.keys():
            self.position_metrics.append(f'Avg_Starter_AV_{pos_group}')
            self.position_metrics.append(f'Avg_Roster_AV_{pos_group}')
        
        # Combined metrics list
        self.metrics = self.base_metrics + self.position_metrics
    
    def normalize_player_name(self, name: str) -> str:
        """
        Normalize player name for matching between starters and roster
        
        Args:
            name: Player name to normalize
            
        Returns:
            Normalized player name
        """
        if pd.isna(name):
            return ""
        
        # Convert to string and strip whitespace
        name = str(name).strip()
        
        # Remove any asterisks or plus signs that indicate Pro Bowl/All-Pro
        name = name.replace('*', '').replace('+', '')
        
        # Strip again after removing symbols
        name = name.strip()
        
        return name
    
    def map_position_to_group(self, position: str) -> Optional[str]:
        """
        Map a position to its position group
        
        Args:
            position: Position abbreviation from starters data
            
        Returns:
            Position group name or None if not found
        """
        if pd.isna(position):
            return None
        
        position = str(position).strip().upper()
        
        for group, positions in self.position_groups.items():
            if position in positions:
                return group
        
        return None
    
    def load_starters_file(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Load starters file for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with starters data or None if file doesn't exist
        """
        team_upper = team.upper()
        starters_file = self.starters_dir / team_upper / f"{team}_{year}_starters.csv"
        
        if not starters_file.exists():
            self.logger.debug(f"Starters file not found: {starters_file}")
            return None
        
        try:
            df = pd.read_csv(starters_file)
            
            # Ensure required columns exist
            required_columns = ['Player', 'Pos']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns {missing_columns} in {starters_file}")
                return None
            
            # Normalize player names for matching
            df['Player_Normalized'] = df['Player'].apply(self.normalize_player_name)
            
            # Remove rows with missing player names
            df = df[df['Player_Normalized'] != '']
            
            # Add metadata
            df['Team'] = team.upper()
            df['Year'] = year
            
            self.logger.debug(f"Loaded {len(df)} starters from {starters_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {starters_file}: {e}")
            return None
    
    def load_roster_file(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Load roster file for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with roster data or None if file doesn't exist
        """
        team_upper = team.upper()
        roster_file = self.roster_dir / team_upper / f"{team}_{year}_roster.csv"
        
        if not roster_file.exists():
            self.logger.debug(f"Roster file not found: {roster_file}")
            return None
        
        try:
            df = pd.read_csv(roster_file)
            
            # Ensure required columns exist
            required_columns = ['Player', 'AV', 'Pos']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns {missing_columns} in {roster_file}")
                return None
            
            # Convert AV to numeric
            df['AV'] = pd.to_numeric(df['AV'], errors='coerce')
            
            # Normalize player names for matching
            df['Player_Normalized'] = df['Player'].apply(self.normalize_player_name)
            
            # Remove rows with missing player names or AV
            df = df[(df['Player_Normalized'] != '') & pd.notna(df['AV'])]
            
            # Add metadata
            df['Team'] = team.upper()
            df['Year'] = year
            
            self.logger.debug(f"Loaded {len(df)} roster players from {roster_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {roster_file}: {e}")
            return None
    
    def match_starters_to_roster(self, starters_df: pd.DataFrame, 
                                roster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match starters to roster to get AV values
        
        Args:
            starters_df: Starters DataFrame
            roster_df: Roster DataFrame
            
        Returns:
            DataFrame with matched starters and their AV values
        """
        # Merge starters with roster based on normalized player names
        matched_df = starters_df.merge(
            roster_df[['Player_Normalized', 'AV']], 
            on='Player_Normalized', 
            how='left'
        )
        
        # Count successful matches
        matched_count = matched_df['AV'].notna().sum()
        total_starters = len(matched_df)
        
        self.logger.debug(f"Matched {matched_count} of {total_starters} starters to roster AV values")
        
        return matched_df
    
    def calculate_av_metrics(self, starters_df: pd.DataFrame, 
                            roster_df: pd.DataFrame, 
                            team: str, year: int) -> Dict[str, float]:
        """
        Calculate AV metrics for starters vs roster with position group breakdown
        
        Args:
            starters_df: Starters DataFrame
            roster_df: Roster DataFrame
            team: Team abbreviation
            year: Season year
            
        Returns:
            Dictionary with calculated metrics
        """
        metrics = {}
        
        # Initialize all metrics to 0.0
        for metric in self.metrics:
            metrics[metric] = 0.0
        
        # Match starters to roster to get AV values
        if not starters_df.empty and not roster_df.empty:
            matched_starters = self.match_starters_to_roster(starters_df, roster_df)
            
            # Check for unmatched starters and warn
            unmatched_starters = matched_starters[matched_starters['AV'].isna()]
            if not unmatched_starters.empty:
                unmatched_names = unmatched_starters['Player'].tolist()
                print(f"WARNING: {team} {year} - Could not match {len(unmatched_names)} starters to roster AV: {', '.join(unmatched_names)}")
            
            # Get successfully matched starters with AV values
            successfully_matched = matched_starters.dropna(subset=['AV'])
            starter_av_values = successfully_matched['AV']
            
            # Calculate overall starter metrics
            if len(starter_av_values) > 0:
                metrics['Avg_Starter_AV'] = round(starter_av_values.mean(), 2)
                metrics['StdDev_Starter_AV'] = round(starter_av_values.std(), 2) if len(starter_av_values) > 1 else 0.0
            
            # Calculate position group metrics for starters
            successfully_matched['Position_Group'] = successfully_matched['Pos'].apply(self.map_position_to_group)
            
            for pos_group in self.position_groups.keys():
                group_starters = successfully_matched[successfully_matched['Position_Group'] == pos_group]
                if not group_starters.empty:
                    group_av_values = group_starters['AV']
                    if len(group_av_values) > 0:
                        metrics[f'Avg_Starter_AV_{pos_group}'] = round(group_av_values.mean(), 2)
        
        # Calculate roster metrics
        if not roster_df.empty:
            roster_av_values = roster_df['AV'].dropna()
            
            if len(roster_av_values) > 0:
                metrics['Avg_Roster_AV'] = round(roster_av_values.mean(), 2)
                metrics['StdDev_Roster_AV'] = round(roster_av_values.std(), 2) if len(roster_av_values) > 1 else 0.0
            
            # Calculate position group metrics for roster
            roster_df['Position_Group'] = roster_df['Pos'].apply(self.map_position_to_group)
            
            for pos_group in self.position_groups.keys():
                group_roster = roster_df[roster_df['Position_Group'] == pos_group]
                if not group_roster.empty:
                    group_av_values = group_roster['AV'].dropna()
                    if len(group_av_values) > 0:
                        metrics[f'Avg_Roster_AV_{pos_group}'] = round(group_av_values.mean(), 2)
        
        return metrics
    
    def create_crosstab_row(self, team: str, year: int, 
                           metrics_data: Dict[str, float]) -> Dict[str, any]:
        """
        Create a single row for the crosstab format
        
        Args:
            team: Team abbreviation
            year: Season year
            metrics_data: Calculated metrics
            
        Returns:
            Dictionary representing one row of crosstab data
        """
        row = {
            'Team': team.upper(),
            'Year': year,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add all metrics
        for metric in self.metrics:
            row[metric] = metrics_data.get(metric, 0.0)
        
        return row
    
    def calculate_team_av_crosstab(self, team: str, years: List[int]) -> List[Dict]:
        """
        Calculate AV metrics for a team across multiple years in crosstab format
        
        Args:
            team: Team abbreviation
            years: List of years to analyze
            
        Returns:
            List of dictionaries, each representing a team-year row
        """
        crosstab_rows = []
        
        for year in sorted(years):
            # Load both starters and roster data for this year
            starters_df = self.load_starters_file(team, year)
            roster_df = self.load_roster_file(team, year)
            
            if starters_df is None and roster_df is None:
                self.logger.warning(f"Missing both starters and roster data for {team} {year}")
                continue
            
            # Use empty DataFrame if one is missing
            if starters_df is None:
                starters_df = pd.DataFrame(columns=['Player', 'Player_Normalized', 'Pos'])
            if roster_df is None:
                roster_df = pd.DataFrame(columns=['Player', 'Player_Normalized', 'AV', 'Pos'])
            
            # Calculate metrics
            metrics_data = self.calculate_av_metrics(starters_df, roster_df, team, year)
            
            # Create crosstab row
            crosstab_row = self.create_crosstab_row(team, year, metrics_data)
            crosstab_rows.append(crosstab_row)
            
            self.logger.info(f"Calculated AV metrics for {team} {year}: "
                           f"Starters avg AV {metrics_data['Avg_Starter_AV']}, "
                           f"Roster avg AV {metrics_data['Avg_Roster_AV']}")
        
        return crosstab_rows
    
    def get_column_order(self) -> List[str]:
        """
        Get the proper column order for the crosstab output
        
        Returns:
            List of column names in desired order
        """
        columns = ['Team', 'Year']
        columns.extend(self.metrics)
        columns.append('Analysis_Date')
        return columns
    
    def save_crosstab_data(self, crosstab_data: List[Dict], teams: List[str]) -> bool:
        """
        Save crosstab AV analysis results
        
        Args:
            crosstab_data: List of crosstab row dictionaries
            teams: List of teams processed
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not crosstab_data:
                self.logger.warning("No crosstab data to save")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(crosstab_data)
            
            # Reorder columns for better organization
            column_order = self.get_column_order()
            # Only include columns that actually exist in the data
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]
            
            # Sort by team and year
            df = df.sort_values(['Team', 'Year'], ignore_index=True)
            
            # Save main crosstab file
            output_file = self.output_dir / "av_metrics_crosstab.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved AV metrics crosstab data to {output_file}")
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Teams_Processed': len(teams),
                'Team_List': ', '.join(sorted(teams)),
                'Total_Rows': len(df),
                'Year_Range': f"{df['Year'].min()}-{df['Year'].max()}" if not df.empty else "N/A",
                'Metrics_Calculated': ', '.join(self.metrics),
                'Total_Columns': len(df.columns),
                'Description': 'AV (Approximate Value) metrics comparing starters vs full roster with position group breakdown'
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "av_metrics_crosstab_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved AV metrics crosstab metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving crosstab data: {e}")
            return False
    
    def analyze_av_metrics_crosstab(self, teams: List[str], years: List[int]) -> Dict[str, int]:
        """
        Analyze AV metrics for specified teams and years in crosstab format
        
        Args:
            teams: List of team abbreviations
            years: List of years to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {'success': 0, 'failed': 0, 'no_data': 0}
        all_crosstab_data = []
        successful_teams = []
        
        for team in teams:
            self.logger.info(f"Analyzing AV metrics for {team}")
            
            try:
                # Calculate AV crosstab
                team_crosstab_data = self.calculate_team_av_crosstab(team, years)
                
                if not team_crosstab_data:
                    self.logger.warning(f"No AV metrics data calculated for {team}")
                    results['no_data'] += 1
                    continue
                
                all_crosstab_data.extend(team_crosstab_data)
                successful_teams.append(team)
                results['success'] += 1
                
                self.logger.info(f"Successfully analyzed {team}: "
                               f"{len(team_crosstab_data)} years of data")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {team}: {e}")
                results['failed'] += 1
        
        # Save all crosstab data
        if all_crosstab_data:
            if self.save_crosstab_data(all_crosstab_data, successful_teams):
                self.logger.info(f"Saved AV metrics crosstab data for {len(successful_teams)} teams, "
                               f"{len(all_crosstab_data)} total team-years")
            else:
                results['failed'] += len(successful_teams)
                results['success'] = 0
        
        return results


def main():
    """Main function to run AV metrics crosstab analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze NFL AV (Approximate Value) metrics comparing starters vs full roster in crosstab format'
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
        '--year',
        type=str,
        help='Specific year to analyze or "all" for all available years'
    )
    parser.add_argument(
        '--minyear',
        type=int,
        default=2010,
        help='Minimum year when using --year all (default: 2010)'
    )
    parser.add_argument(
        '--maxyear',
        type=int,
        default=2024,
        help='Maximum year when using --year all (default: 2024)'
    )
    parser.add_argument(
        '--starters-dir',
        type=str,
        default='data/raw/Starters',
        help='Directory containing starters CSV files (default: data/raw/Starters)'
    )
    parser.add_argument(
        '--roster-dir',
        type=str,
        default='data/raw/Rosters',
        help='Directory containing roster CSV files (default: data/raw/Rosters)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for crosstab analysis (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AVMetricsCrosstabAnalyzer(
        starters_dir=args.starters_dir,
        roster_dir=args.roster_dir,
        output_dir=args.output_dir
    )
    
    # Determine teams to analyze
    if args.all_teams:
        teams = PFR_TEAM_ABBREVIATIONS
    elif args.team:
        if args.team.lower() not in PFR_TEAM_ABBREVIATIONS:
            print(f"Error: Unknown team abbreviation '{args.team}'")
            print(f"Available teams: {', '.join(sorted(PFR_TEAM_ABBREVIATIONS))}")
            sys.exit(1)
        teams = [args.team.lower()]
    else:
        print("Error: Must specify either --team or --all-teams")
        print(f"Available teams: {', '.join(sorted(PFR_TEAM_ABBREVIATIONS))}")
        sys.exit(1)
    
    # Determine years to analyze
    if args.year and args.year.lower() != 'all':
        try:
            year = int(args.year)
            if year < 2010 or year > 2030:
                raise ValueError("Year out of reasonable range")
            years = [year]
        except ValueError:
            print(f"Error: Invalid year '{args.year}'. Must be a valid year or 'all'")
            sys.exit(1)
    else:
        years = list(range(args.minyear, args.maxyear + 1))
    
    # Run analysis
    print(f"Starting AV metrics crosstab analysis for {len(teams)} team(s)")
    print(f"Teams: {', '.join(teams)}")
    print(f"Years: {years[0]}-{years[-1]}" if len(years) > 1 else f"Year: {years[0]}")
    print(f"Starters directory: {args.starters_dir}")
    print(f"Roster directory: {args.roster_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = analyzer.analyze_av_metrics_crosstab(teams=teams, years=years)
    
    # Print results
    print("\nAV metrics crosstab analysis completed!")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"No data: {results['no_data']}")
    print(f"Total teams processed: {sum(results.values())}")
    
    if results['success'] > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Generated files:")
        print("  - av_metrics_crosstab.csv (main crosstab dataset)")
        print("  - av_metrics_crosstab_metadata.csv (processing metadata)")
        
        # Show column structure  
        base_metrics = analyzer.base_metrics
        starter_pos_metrics = [m for m in analyzer.position_metrics if 'Starter' in m]
        roster_pos_metrics = [m for m in analyzer.position_metrics if 'Roster' in m]
        print(f"\nCrosstab structure:")
        print(f"  - Base columns: Team, Year")
        print(f"  - Overall metrics: {', '.join(base_metrics)}")
        print(f"  - Starter position metrics: {', '.join(starter_pos_metrics[:3])}...")
        print(f"  - Roster position metrics: {', '.join(roster_pos_metrics[:3])}...")
        print(f"  - Total columns: {len(analyzer.metrics) + 3}")  # +3 for Team, Year, Analysis_Date


if __name__ == "__main__":
    main()