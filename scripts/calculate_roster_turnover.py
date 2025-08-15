#!/usr/bin/env python3
"""
Roster Turnover Analysis Script

This script calculates roster turnover percentages by position for NFL teams by comparing
player names between consecutive years. It uses roster data scraped from Pro Football Reference.

Usage:
    python scripts/calculate_roster_turnover.py --team den --year 2024  # Compares 2023->2024
    python scripts/calculate_roster_turnover.py --all-teams --year all --minyear 2015 --maxyear 2024
    python scripts/calculate_roster_turnover.py --team buf --year all
    python scripts/calculate_roster_turnover.py --all-teams --year 2024  # All teams for 2024
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

class RosterTurnoverAnalyzer:
    """Analyzes roster turnover by comparing consecutive years"""
    
    def __init__(self, roster_dir: str = "data/raw/Rosters", output_dir: str = "data/processed/RosterTurnover"):
        """
        Initialize the analyzer
        
        Args:
            roster_dir: Directory containing roster CSV files
            output_dir: Directory to save turnover analysis
        """
        self.roster_dir = Path(roster_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders for different types of output
        self.detailed_dir = self.output_dir / "detailed"
        self.summary_dir = self.output_dir / "summary"
        self.detailed_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Position groupings for analysis
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['C', 'G', 'T', 'OL', 'OG', 'OT'],
            'DL': ['DE', 'DT', 'NT', 'DL'],
            'LB': ['LB', 'ILB', 'OLB', 'MLB'],
            'DB': ['CB', 'S', 'SS', 'FS', 'DB', 'SAF'],
            'ST': ['K', 'P', 'LS']
        }
    
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
                                  team: str, year1: int, year2: int) -> pd.DataFrame:
        """
        Calculate turnover by position between two consecutive years
        
        Args:
            roster_year1: Roster DataFrame for year 1
            roster_year2: Roster DataFrame for year 2
            team: Team abbreviation
            year1: First year
            year2: Second year (year1 + 1)
            
        Returns:
            DataFrame with turnover statistics by position
        """
        turnover_results = []
        
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
            
            turnover_results.append({
                'Team': team.upper(),
                'Year_From': year1,
                'Year_To': year2,
                'Position_Group': position,
                'Players_Year1': total_year1,
                'Players_Year2': total_year2,
                'Players_Retained': len(retained_players),
                'Players_Departed': len(departed_players),
                'Players_New': len(new_players),
                'Retention_Rate_Pct': round(retention_rate, 2),
                'Departure_Rate_Pct': round(departure_rate, 2),
                'New_Player_Rate_Pct': round(new_player_rate, 2),
                'Net_Change': total_year2 - total_year1
            })
        
        return pd.DataFrame(turnover_results)
    
    def calculate_team_turnover(self, team: str, years: List[int]) -> pd.DataFrame:
        """
        Calculate turnover for a team across multiple years
        
        Args:
            team: Team abbreviation
            years: List of years to analyze
            
        Returns:
            DataFrame with all turnover comparisons for the team
        """
        all_turnover_data = []
        
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
            turnover_df = self.calculate_position_turnover(roster1, roster2, team, year1, year2)
            
            if not turnover_df.empty:
                all_turnover_data.append(turnover_df)
                self.logger.info(f"Calculated turnover for {team} {year1}-{year2}: "
                               f"{len(turnover_df)} position groups")
        
        if all_turnover_data:
            combined_df = pd.concat(all_turnover_data, ignore_index=True)
            combined_df['Analysis_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return combined_df
        else:
            return pd.DataFrame()
    
    def generate_team_summary(self, turnover_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for team turnover
        
        Args:
            turnover_df: DataFrame with detailed turnover data
            
        Returns:
            DataFrame with summary statistics
        """
        if turnover_df.empty:
            return pd.DataFrame()
        
        # Calculate averages by team and position
        summary = turnover_df.groupby(['Team', 'Position_Group']).agg({
            'Year_From': ['min', 'max', 'count'],
            'Players_Year1': 'mean',
            'Players_Year2': 'mean', 
            'Retention_Rate_Pct': 'mean',
            'Departure_Rate_Pct': 'mean',
            'New_Player_Rate_Pct': 'mean',
            'Net_Change': 'mean'
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.rename(columns={
            'Year_From_min': 'First_Year',
            'Year_From_max': 'Last_Year',
            'Year_From_count': 'Years_Analyzed',
            'Players_Year1_mean': 'Avg_Players_Year1',
            'Players_Year2_mean': 'Avg_Players_Year2',
            'Retention_Rate_Pct_mean': 'Avg_Retention_Rate_Pct',
            'Departure_Rate_Pct_mean': 'Avg_Departure_Rate_Pct',
            'New_Player_Rate_Pct_mean': 'Avg_New_Player_Rate_Pct',
            'Net_Change_mean': 'Avg_Net_Change'
        })
        
        summary = summary.reset_index()
        return summary
    
    def save_turnover_data(self, turnover_df: pd.DataFrame, summary_df: pd.DataFrame, 
                          team: str) -> bool:
        """
        Save turnover analysis results to separate subfolders
        
        Args:
            turnover_df: Detailed turnover DataFrame
            summary_df: Summary statistics DataFrame
            team: Team abbreviation
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not turnover_df.empty:
                # Save detailed turnover data to detailed subfolder
                detail_file = self.detailed_dir / f"{team}_roster_turnover_detailed.csv"
                turnover_df.to_csv(detail_file, index=False)
                self.logger.info(f"Saved detailed turnover data to {detail_file}")
            
            if not summary_df.empty:
                # Save summary data to summary subfolder
                summary_file = self.summary_dir / f"{team}_roster_turnover_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                self.logger.info(f"Saved turnover summary to {summary_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving turnover data for {team}: {e}")
            return False
    
    def analyze_roster_turnover(self, teams: List[str], years: List[int]) -> Dict[str, int]:
        """
        Analyze roster turnover for specified teams and years
        
        Args:
            teams: List of team abbreviations
            years: List of years to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {'success': 0, 'failed': 0, 'no_data': 0}
        
        for team in teams:
            self.logger.info(f"Analyzing roster turnover for {team}")
            
            try:
                # Calculate turnover
                turnover_df = self.calculate_team_turnover(team, years)
                
                if turnover_df.empty:
                    self.logger.warning(f"No turnover data calculated for {team}")
                    results['no_data'] += 1
                    continue
                
                # Generate summary
                summary_df = self.generate_team_summary(turnover_df)
                
                # Save results
                if self.save_turnover_data(turnover_df, summary_df, team):
                    results['success'] += 1
                    self.logger.info(f"Successfully analyzed {team}: "
                                   f"{len(turnover_df)} year-to-year comparisons")
                else:
                    results['failed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {team}: {e}")
                results['failed'] += 1
        
        return results


def main():
    """Main function to run roster turnover analysis"""
    parser = argparse.ArgumentParser(
        description='Analyze NFL roster turnover by position'
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
        default='data/processed/RosterTurnover',
        help='Output directory for turnover analysis (default: data/processed/RosterTurnover)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RosterTurnoverAnalyzer(
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
    print(f"Starting roster turnover analysis for {len(teams)} team(s)")
    print(f"Teams: {', '.join(teams)}")
    print(f"Years: {years[0]}-{years[-1]}" if len(years) > 1 else f"Year: {years[0]}")
    print(f"Roster directory: {args.roster_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = analyzer.analyze_roster_turnover(teams=teams, years=years)
    
    # Print results
    print("\nRoster turnover analysis completed!")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"No data: {results['no_data']}")
    print(f"Total teams processed: {sum(results.values())}")
    
    if results['success'] > 0:
        print(f"\nResults saved to: {args.output_dir}")
        print("Generated files per team:")
        print("  - detailed/{team}_roster_turnover_detailed.csv (year-to-year comparisons)")
        print("  - summary/{team}_roster_turnover_summary.csv (position averages)")


if __name__ == "__main__":
    main()