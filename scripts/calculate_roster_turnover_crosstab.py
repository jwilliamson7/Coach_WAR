#!/usr/bin/env python3
"""
Roster Turnover Crosstab Analysis Script

This script calculates roster turnover percentages by position for NFL teams and outputs
them in a crosstab format where each row is one team-year and columns contain all 
position-specific metrics.

Usage:
    python scripts/calculate_roster_turnover_crosstab.py --team den --year 2024
    python scripts/calculate_roster_turnover_crosstab.py --all-teams --year all --minyear 2015 --maxyear 2024
    python scripts/calculate_roster_turnover_crosstab.py --all-teams --year 2024
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
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS

# Create PFR team abbreviations list from the corrected mappings
PFR_TEAM_ABBREVIATIONS = list(set(SPOTRAC_TO_PFR_MAPPINGS.values()))

class RosterTurnoverCrosstabAnalyzer:
    """Analyzes roster turnover in crosstab format"""
    
    def __init__(self, roster_dir: str = "data/raw/Rosters", output_dir: str = "data/final"):
        """
        Initialize the analyzer
        
        Args:
            roster_dir: Directory containing roster CSV files
            output_dir: Directory to save turnover analysis
        """
        self.roster_dir = Path(roster_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Position groupings for analysis (ordered to match salary data: QB, RB, WR, TE, OL, DL, LB, CB, S)
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['C', 'G', 'T', 'OL', 'OG', 'OT'],
            'DL': ['DE', 'DT', 'NT', 'DL'],
            'LB': ['LB', 'ILB', 'OLB', 'MLB'],
            'CB': ['CB', 'DB'],
            'S': ['S', 'SS', 'FS', 'SAF']
        }
        
        # All metrics we'll calculate for each position
        self.metrics = [
            'Players_Retained',
            'Players_Departed',
            'Players_New',
            'Retention_Rate_Pct',
            'Departure_Rate_Pct',
            'New_Player_Rate_Pct',
            'Net_Change'
        ]
    
    def normalize_position(self, position: str) -> str:
        """
        Normalize position to position group
        
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
    
    def load_roster_file(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Load roster file for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with roster data or None if file doesn't exist
        """
        # Look for roster file pattern: {TEAM}/{team}_{year}_roster.csv
        team_upper = team.upper()
        roster_file = self.roster_dir / team_upper / f"{team}_{year}_roster.csv"
        
        if not roster_file.exists():
            self.logger.debug(f"Roster file not found: {roster_file}")
            return None
        
        try:
            df = pd.read_csv(roster_file)
            
            # Ensure required columns exist
            if 'Player' not in df.columns:
                self.logger.warning(f"No 'Player' column in {roster_file}")
                return None
            
            # Clean player names
            df['Player_Clean'] = df['Player'].str.strip().str.upper()
            
            # Handle position column (might be named differently)
            pos_column = None
            for col in ['Pos', 'Position', 'Positions']:
                if col in df.columns:
                    pos_column = col
                    break
            
            if pos_column:
                df['Position_Group'] = df[pos_column].apply(self.normalize_position)
            else:
                self.logger.warning(f"No position column found in {roster_file}")
                df['Position_Group'] = 'UNKNOWN'
            
            # Add metadata
            df['Team'] = team.upper()
            df['Year'] = year
            
            self.logger.debug(f"Loaded {len(df)} players from {roster_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {roster_file}: {e}")
            return None
    
    def calculate_position_turnover(self, roster_year1: pd.DataFrame, roster_year2: pd.DataFrame, 
                                  team: str, year1: int, year2: int) -> Dict[str, Dict[str, float]]:
        """
        Calculate turnover by position between two consecutive years
        
        Args:
            roster_year1: Roster DataFrame for year 1
            roster_year2: Roster DataFrame for year 2
            team: Team abbreviation
            year1: First year
            year2: Second year (year1 + 1)
            
        Returns:
            Dictionary with position as key and metrics as nested dict
        """
        turnover_results = {}
        
        # Get all position groups that appear in either year
        positions_year1 = set(roster_year1['Position_Group'].dropna())
        positions_year2 = set(roster_year2['Position_Group'].dropna())
        all_positions = positions_year1.union(positions_year2)
        
        for position in all_positions:
            # Get players for this position in each year
            players_year1 = set(roster_year1[roster_year1['Position_Group'] == position]['Player_Clean'].dropna())
            players_year2 = set(roster_year2[roster_year2['Position_Group'] == position]['Player_Clean'].dropna())
            
            # Calculate turnover metrics
            retained_players = players_year1.intersection(players_year2)
            departed_players = players_year1 - players_year2
            new_players = players_year2 - players_year1
            
            # Calculate percentages
            total_year1 = len(players_year1)
            total_year2 = len(players_year2)
            
            if total_year1 > 0:
                retention_rate = len(retained_players) / total_year1 * 100
                departure_rate = len(departed_players) / total_year1 * 100
            else:
                retention_rate = 0
                departure_rate = 0
            
            if total_year2 > 0:
                new_player_rate = len(new_players) / total_year2 * 100
            else:
                new_player_rate = 0
            
            turnover_results[position] = {
                'Players_Year1': total_year1,
                'Players_Year2': total_year2,
                'Players_Retained': len(retained_players),
                'Players_Departed': len(departed_players),
                'Players_New': len(new_players),
                'Retention_Rate_Pct': round(retention_rate, 2),
                'Departure_Rate_Pct': round(departure_rate, 2),
                'New_Player_Rate_Pct': round(new_player_rate, 2),
                'Net_Change': total_year2 - total_year1
            }
        
        return turnover_results
    
    def create_crosstab_row(self, team: str, year1: int, year2: int, 
                           position_data: Dict[str, Dict[str, float]]) -> Dict[str, any]:
        """
        Create a single row for the crosstab format
        
        Args:
            team: Team abbreviation
            year1: First year
            year2: Second year (comparison year)
            position_data: Position turnover data
            
        Returns:
            Dictionary representing one row of crosstab data
        """
        row = {
            'Team': team.upper(),
            'Year_From': year1,
            'Year_To': year2,
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Position order to match salary data: QB, RB, WR, TE, OL, DL, LB, CB, S
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        
        # Add columns for each position and metric combination
        for position in position_order:
            for metric in self.metrics:
                column_name = f"{position}_{metric}"
                
                if position in position_data:
                    row[column_name] = position_data[position][metric]
                else:
                    # Fill missing positions with appropriate defaults
                    if metric in ['Players_Retained', 'Players_Departed', 'Players_New', 'Net_Change']:
                        row[column_name] = 0
                    else:  # Percentage metrics
                        row[column_name] = 0.0
        
        return row
    
    def calculate_team_turnover_crosstab(self, team: str, years: List[int]) -> List[Dict]:
        """
        Calculate turnover for a team across multiple years in crosstab format
        
        Args:
            team: Team abbreviation
            years: List of years to analyze
            
        Returns:
            List of dictionaries, each representing a team-year row
        """
        crosstab_rows = []
        
        # Sort years to ensure consecutive comparisons
        years = sorted(years)
        
        for i in range(len(years) - 1):
            year1 = years[i]
            year2 = years[i + 1]
            
            # Load rosters for both years
            roster1 = self.load_roster_file(team, year1)
            roster2 = self.load_roster_file(team, year2)
            
            if roster1 is None or roster2 is None:
                self.logger.warning(f"Missing roster data for {team} {year1}-{year2} comparison")
                continue
            
            # Calculate turnover
            position_data = self.calculate_position_turnover(roster1, roster2, team, year1, year2)
            
            # Create crosstab row
            crosstab_row = self.create_crosstab_row(team, year1, year2, position_data)
            crosstab_rows.append(crosstab_row)
            
            self.logger.info(f"Calculated turnover for {team} {year1}-{year2}: "
                           f"{len(position_data)} position groups")
        
        return crosstab_rows
    
    def get_column_order(self) -> List[str]:
        """
        Get the proper column order for the crosstab output (matching salary data order)
        
        Returns:
            List of column names in desired order
        """
        columns = ['Team', 'Year_From', 'Year_To']
        
        # Position order to match salary data: QB, RB, WR, TE, OL, DL, LB, CB, S
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        
        # Add position-metric columns in salary data order
        for position in position_order:
            for metric in self.metrics:
                columns.append(f"{position}_{metric}")
        
        columns.append('Analysis_Date')
        return columns
    
    def save_crosstab_data(self, crosstab_data: List[Dict], teams: List[str]) -> bool:
        """
        Save crosstab turnover analysis results
        
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
            df = df.sort_values(['Team', 'Year_From'], ignore_index=True)
            
            # Save main crosstab file
            output_file = self.output_dir / "roster_turnover_crosstab.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved crosstab turnover data to {output_file}")
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Teams_Processed': len(teams),
                'Team_List': ', '.join(sorted(teams)),
                'Total_Rows': len(df),
                'Year_Range': f"{df['Year_From'].min()}-{df['Year_To'].max()}" if not df.empty else "N/A",
                'Position_Groups': ', '.join(sorted(self.position_groups.keys())),
                'Metrics_Per_Position': len(self.metrics),
                'Total_Columns': len(df.columns)
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "roster_turnover_crosstab_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved crosstab metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving crosstab data: {e}")
            return False
    
    def analyze_roster_turnover_crosstab(self, teams: List[str], years: List[int]) -> Dict[str, int]:
        """
        Analyze roster turnover for specified teams and years in crosstab format
        
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
            self.logger.info(f"Analyzing roster turnover for {team}")
            
            try:
                # Calculate turnover crosstab
                team_crosstab_data = self.calculate_team_turnover_crosstab(team, years)
                
                if not team_crosstab_data:
                    self.logger.warning(f"No turnover data calculated for {team}")
                    results['no_data'] += 1
                    continue
                
                all_crosstab_data.extend(team_crosstab_data)
                successful_teams.append(team)
                results['success'] += 1
                
                self.logger.info(f"Successfully analyzed {team}: "
                               f"{len(team_crosstab_data)} year-to-year comparisons")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {team}: {e}")
                results['failed'] += 1
        
        # Save all crosstab data
        if all_crosstab_data:
            if self.save_crosstab_data(all_crosstab_data, successful_teams):
                self.logger.info(f"Saved crosstab data for {len(successful_teams)} teams, "
                               f"{len(all_crosstab_data)} total comparisons")
            else:
                results['failed'] += len(successful_teams)
                results['success'] = 0
        
        return results


def main():
    """Main function to run roster turnover crosstab analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze NFL roster turnover by position in crosstab format'
    )
    parser.add_argument(
        '--team',
        type=str,
        help='Team abbreviation (e.g., "den")'
    )
    parser.add_argument(
        '--all-teams',
        action='store_true',
        help='Process all available teams'
    )
    parser.add_argument(
        '--year',
        type=str,
        help='Specific end year to analyze (compares year-1 to year) or "all" for all available years'
    )
    parser.add_argument(
        '--minyear',
        type=int,
        default=2011,
        help='Minimum year when using --year all (default: 2011)'
    )
    parser.add_argument(
        '--maxyear',
        type=int,
        default=2024,
        help='Maximum year when using --year all (default: 2024)'
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
    analyzer = RosterTurnoverCrosstabAnalyzer(
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
            if year < 2011 or year > 2030:  # Need at least 2011 to compare to 2010
                raise ValueError("Year out of reasonable range (minimum 2011)")
            years = [year - 1, year]  # Compare previous year to specified year
        except ValueError:
            print(f"Error: Invalid year '{args.year}'. Must be a valid year (2011+) or 'all'")
            sys.exit(1)
    else:
        years = list(range(args.minyear, args.maxyear + 1))
    
    # Run analysis
    print(f"Starting roster turnover crosstab analysis for {len(teams)} team(s)")
    print(f"Teams: {', '.join(teams)}")
    print(f"Years: {years[0]}-{years[-1]}" if len(years) > 1 else f"Year: {years[0]}")
    print(f"Roster directory: {args.roster_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = analyzer.analyze_roster_turnover_crosstab(teams=teams, years=years)
    
    # Print results
    print("\nRoster turnover crosstab analysis completed!")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"No data: {results['no_data']}")
    print(f"Total teams processed: {sum(results.values())}")
    
    if results['success'] > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Generated files:")
        print("  - roster_turnover_crosstab.csv (main crosstab dataset)")
        print("  - roster_turnover_crosstab_metadata.csv (processing metadata)")
        
        # Show column structure  
        position_order = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'CB', 'S']
        metrics = analyzer.metrics
        print(f"\nCrosstab structure:")
        print(f"  - Base columns: Team, Year_From, Year_To")
        print(f"  - Position groups (salary data order): {', '.join(position_order)}")
        print(f"  - Metrics per position: {', '.join(metrics)}")
        print(f"  - Total position columns: {len(position_order) * len(metrics)}")


if __name__ == "__main__":
    main()