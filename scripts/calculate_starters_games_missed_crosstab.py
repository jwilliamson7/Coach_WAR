#!/usr/bin/env python3
"""
Starters Games Missed Crosstab Analysis Script

This script calculates the percentage of games each starter missed based on their Games Started (GS)
field and the total games in that season. Results are aggregated by position and output in crosstab format.

Usage:
    python scripts/calculate_starters_games_missed_crosstab.py --team crd --year 2011
    python scripts/calculate_starters_games_missed_crosstab.py --all-teams --year all --minyear 2015 --maxyear 2024
    python scripts/calculate_starters_games_missed_crosstab.py --all-teams --year 2024
"""

import pandas as pd
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
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS, get_games_in_season

# Create PFR team abbreviations list from the corrected mappings
PFR_TEAM_ABBREVIATIONS = list(set(SPOTRAC_TO_PFR_MAPPINGS.values()))

class StartersGamesMissedCrosstabAnalyzer:
    """Analyzes starters games missed percentages in crosstab format"""
    
    def __init__(self, starters_dir: str = "data/raw/Starters", output_dir: str = "data/final"):
        """
        Initialize the analyzer
        
        Args:
            starters_dir: Directory containing starters CSV files
            output_dir: Directory to save games missed analysis
        """
        self.starters_dir = Path(starters_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Position groupings for starters analysis (combining O-line as requested)
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['LT', 'LG', 'C', 'RG', 'RT', 'G', 'T', 'OL', 'OG', 'OT'],  # Combined O-line
            'DL': ['LDE', 'RDE', 'NT', 'DT', 'DE', 'DL'],  # Combined D-line
            'LB': ['LOLB', 'LILB', 'RILB', 'ROLB', 'MLB', 'LB', 'ILB', 'OLB'],  # All linebackers
            'CB': ['LCB', 'RCB', 'CB', 'DB'],  # Cornerbacks
            'S': ['SS', 'FS', 'S', 'SAF']  # Safeties
        }
        
        # Metrics we'll calculate for each position
        self.metrics = [
            'Avg_Games_Missed_Pct',
            'Max_Games_Missed_Pct',
            'Min_Games_Missed_Pct',
            'Players_Count'
        ]
    
    def normalize_position(self, position: str) -> str:
        """
        Normalize starter position to position group
        
        Args:
            position: Raw position string
            
        Returns:
            Normalized position group
        """
        if pd.isna(position) or not position:
            return 'UNKNOWN'
        
        position = str(position).upper().strip()
        
        # Handle multiple positions (take first)
        if '/' in position:
            position = position.split('/')[0]
        if '-' in position:
            position = position.split('-')[0]
        
        # Map to position groups
        for group, positions in self.position_groups.items():
            if position in positions:
                return group
        
        return 'OTHER'
    
    def load_starters_file(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Load starters file for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with starters data or None if file doesn't exist
        """
        # Look for starters file pattern: {TEAM}/{team}_{year}_starters.csv
        team_upper = team.upper()
        starters_file = self.starters_dir / team_upper / f"{team}_{year}_starters.csv"
        
        if not starters_file.exists():
            self.logger.debug(f"Starters file not found: {starters_file}")
            return None
        
        try:
            df = pd.read_csv(starters_file)
            
            # Ensure required columns exist
            required_columns = ['Player', 'Pos', 'GS']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Missing required columns {missing_columns} in {starters_file}")
                return None
            
            # Clean player names
            df['Player_Clean'] = df['Player'].str.strip().str.upper()
            
            # Handle position column
            df['Position_Group'] = df['Pos'].apply(self.normalize_position)
            
            # Convert GS to numeric, handling any non-numeric values
            df['GS'] = pd.to_numeric(df['GS'], errors='coerce')
            
            # Remove rows with missing or invalid GS values
            df = df.dropna(subset=['GS'])
            
            # Add metadata
            df['Team'] = team.upper()
            df['Year'] = year
            
            self.logger.debug(f"Loaded {len(df)} starters from {starters_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {starters_file}: {e}")
            return None
    
    def calculate_games_missed_by_position(self, starters_df: pd.DataFrame, 
                                         team: str, year: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate games missed percentages by position for a team and year
        
        Args:
            starters_df: Starters DataFrame
            team: Team abbreviation
            year: Season year
            
        Returns:
            Dictionary with position as key and metrics as nested dict
        """
        # Get total games in season
        total_games = get_games_in_season(year)
        
        # Calculate games missed percentage for each player
        starters_df['Games_Missed'] = total_games - starters_df['GS']
        starters_df['Games_Missed_Pct'] = (starters_df['Games_Missed'] / total_games * 100).round(2)
        
        position_results = {}
        
        # Get all position groups that appear in the data
        positions = set(starters_df['Position_Group'].dropna())
        
        for position in positions:
            # Get players for this position
            position_players = starters_df[starters_df['Position_Group'] == position]
            
            if len(position_players) > 0:
                games_missed_pcts = position_players['Games_Missed_Pct']
                
                position_results[position] = {
                    'Avg_Games_Missed_Pct': round(games_missed_pcts.mean(), 2),
                    'Max_Games_Missed_Pct': round(games_missed_pcts.max(), 2),
                    'Min_Games_Missed_Pct': round(games_missed_pcts.min(), 2),
                    'Players_Count': len(position_players),
                    'Total_Games_In_Season': total_games
                }
                
                self.logger.debug(f"{team} {year} {position}: {len(position_players)} players, "
                                f"avg {position_results[position]['Avg_Games_Missed_Pct']}% missed")
        
        return position_results
    
    def create_crosstab_row(self, team: str, year: int, 
                           position_data: Dict[str, Dict[str, float]]) -> Dict[str, any]:
        """
        Create a single row for the crosstab format
        
        Args:
            team: Team abbreviation
            year: Season year
            position_data: Position games missed data
            
        Returns:
            Dictionary representing one row of crosstab data
        """
        row = {
            'Team': team.upper(),
            'Year': year,
            'Total_Games_In_Season': get_games_in_season(year),
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Position order for starters: QB, RB, WR, TE, OL, DL, LB, CB, S
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        
        # Add columns for each position and metric combination
        for position in position_order:
            for metric in self.metrics:
                column_name = f"{position}_{metric}"
                
                if position in position_data:
                    row[column_name] = position_data[position][metric]
                else:
                    # Fill missing positions with appropriate defaults
                    if metric == 'Players_Count':
                        row[column_name] = 0
                    else:  # Percentage metrics
                        row[column_name] = 0.0
        
        return row
    
    def calculate_team_games_missed_crosstab(self, team: str, years: List[int]) -> List[Dict]:
        """
        Calculate games missed for a team across multiple years in crosstab format
        
        Args:
            team: Team abbreviation
            years: List of years to analyze
            
        Returns:
            List of dictionaries, each representing a team-year row
        """
        crosstab_rows = []
        
        for year in sorted(years):
            # Load starters for this year
            starters_df = self.load_starters_file(team, year)
            
            if starters_df is None or starters_df.empty:
                self.logger.warning(f"Missing starters data for {team} {year}")
                continue
            
            # Calculate games missed by position
            position_data = self.calculate_games_missed_by_position(starters_df, team, year)
            
            # Create crosstab row
            crosstab_row = self.create_crosstab_row(team, year, position_data)
            crosstab_rows.append(crosstab_row)
            
            self.logger.info(f"Calculated games missed for {team} {year}: "
                           f"{len(position_data)} position groups")
        
        return crosstab_rows
    
    def get_column_order(self) -> List[str]:
        """
        Get the proper column order for the crosstab output
        
        Returns:
            List of column names in desired order
        """
        columns = ['Team', 'Year', 'Total_Games_In_Season']
        
        # Position order for starters: QB, RB, WR, TE, OL, DL, LB, CB, S
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        
        # Add position-metric columns
        for position in position_order:
            for metric in self.metrics:
                columns.append(f"{position}_{metric}")
        
        columns.append('Analysis_Date')
        return columns
    
    def save_crosstab_data(self, crosstab_data: List[Dict], teams: List[str]) -> bool:
        """
        Save crosstab games missed analysis results
        
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
            output_file = self.output_dir / "starters_games_missed_crosstab.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved starters games missed crosstab data to {output_file}")
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Teams_Processed': len(teams),
                'Team_List': ', '.join(sorted(teams)),
                'Total_Rows': len(df),
                'Year_Range': f"{df['Year'].min()}-{df['Year'].max()}" if not df.empty else "N/A",
                'Position_Groups': ', '.join(sorted(self.position_groups.keys())),
                'Metrics_Per_Position': len(self.metrics),
                'Total_Columns': len(df.columns)
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "starters_games_missed_crosstab_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved starters games missed crosstab metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving crosstab data: {e}")
            return False
    
    def analyze_starters_games_missed_crosstab(self, teams: List[str], years: List[int]) -> Dict[str, int]:
        """
        Analyze starters games missed for specified teams and years in crosstab format
        
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
            self.logger.info(f"Analyzing starters games missed for {team}")
            
            try:
                # Calculate games missed crosstab
                team_crosstab_data = self.calculate_team_games_missed_crosstab(team, years)
                
                if not team_crosstab_data:
                    self.logger.warning(f"No starters games missed data calculated for {team}")
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
                self.logger.info(f"Saved starters games missed crosstab data for {len(successful_teams)} teams, "
                               f"{len(all_crosstab_data)} total team-years")
            else:
                results['failed'] += len(successful_teams)
                results['success'] = 0
        
        return results


def main():
    """Main function to run starters games missed crosstab analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze NFL starters games missed percentages by position in crosstab format'
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
        '--output-dir',
        type=str,
        default='data/final',
        help='Output directory for crosstab analysis (default: data/final)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = StartersGamesMissedCrosstabAnalyzer(
        starters_dir=args.starters_dir,
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
    print(f"Starting starters games missed crosstab analysis for {len(teams)} team(s)")
    print(f"Teams: {', '.join(teams)}")
    print(f"Years: {years[0]}-{years[-1]}" if len(years) > 1 else f"Year: {years[0]}")
    print(f"Starters directory: {args.starters_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = analyzer.analyze_starters_games_missed_crosstab(teams=teams, years=years)
    
    # Print results
    print("\nStarters games missed crosstab analysis completed!")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"No data: {results['no_data']}")
    print(f"Total teams processed: {sum(results.values())}")
    
    if results['success'] > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Generated files:")
        print("  - starters_games_missed_crosstab.csv (main crosstab dataset)")
        print("  - starters_games_missed_crosstab_metadata.csv (processing metadata)")
        
        # Show column structure  
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        metrics = analyzer.metrics
        print(f"\nCrosstab structure:")
        print(f"  - Base columns: Team, Year, Total_Games_In_Season")
        print(f"  - Position groups: {', '.join(position_order)}")
        print(f"  - Metrics per position: {', '.join(metrics)}")
        print(f"  - Total position columns: {len(position_order) * len(metrics)}")


if __name__ == "__main__":
    main()