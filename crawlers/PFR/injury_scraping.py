#!/usr/bin/env python3
"""
Pro Football Reference Injury Data Scraper

This script scrapes injury data from Pro Football Reference, extracting total games lost
by injury type for each team and year. It includes rate limiting, command line arguments,
and checks for existing files before making web requests.

Usage:
    python injury_scraping.py --team den --year 2024
    python injury_scraping.py --team all --year 2024
    python injury_scraping.py --team den --year all
    python injury_scraping.py --team all --year all --start-year 2010 --end-year 2024
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import random
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent.parent))
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS

# Create PFR team abbreviations list from the mappings
PFR_TEAM_ABBREVIATIONS = list(set(SPOTRAC_TO_PFR_MAPPINGS.values()))

class InjuryDataScraper:
    """Scrapes injury data from Pro Football Reference"""
    
    def __init__(self, output_dir: str = "data/raw/Injuries"):
        """
        Initialize the scraper
        
        Args:
            output_dir: Directory to save scraped data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting parameters
        self.min_delay = 2.0
        self.max_delay = 5.0
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def rate_limit(self):
        """Apply rate limiting between requests"""
        delay = random.uniform(self.min_delay, self.max_delay)
        self.logger.debug(f"Rate limiting: sleeping for {delay:.2f} seconds")
        time.sleep(delay)
    
    def file_exists(self, team: str, year: int) -> bool:
        """Check if injury data file already exists"""
        filename = f"{team}_{year}_injuries.csv"
        filepath = self.output_dir / filename
        return filepath.exists()
    
    def scrape_team_year_injuries(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Scrape injury data for a specific team and year
        
        Args:
            team: Team abbreviation (e.g., 'den')
            year: Season year
            
        Returns:
            DataFrame with injury data or None if scraping failed
        """
        url = f"https://www.pro-football-reference.com/teams/{team}/{year}_injuries.htm"
        
        try:
            self.logger.info(f"Scraping {team} {year} injury data from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the injury table - look for 'team_injuries' table
            injury_table = None
            
            # First try to find uncommented table
            injury_table = soup.find('table', {'id': 'team_injuries'})
            
            # If not found, look in comments for injury-related tables
            if injury_table is None:
                comments = soup.find_all(string=lambda text: isinstance(text, Comment))
                for comment in comments:
                    if 'id="team_injuries' in comment:
                        comment_soup = BeautifulSoup(comment, 'html.parser')
                        injury_table = comment_soup.find('table', {'id': 'team_injuries'})
                        if injury_table:
                            break
            
            if injury_table is None:
                self.logger.warning(f"No injury table found for {team} {year}")
                return None
            
            # Extract table data
            injury_data = self._parse_injury_table(injury_table, team, year)
            
            if injury_data is not None and not injury_data.empty:
                self.logger.info(f"Successfully scraped {len(injury_data)} injury records for {team} {year}")
                return injury_data
            else:
                self.logger.warning(f"No injury data extracted for {team} {year}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {team} {year}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error scraping {team} {year}: {e}")
            return None
    
    def _parse_injury_table(self, table, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Parse injury table and extract injury data with games lost by status
        
        Args:
            table: BeautifulSoup table element
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with injury data
        """
        try:
            # Get table headers (dates/weeks)
            headers = []
            header_row = table.find('thead')
            if header_row:
                for th in header_row.find_all(['th', 'td']):
                    header_text = th.get_text(strip=True)
                    if header_text and header_text not in ['Player', 'Pos']:
                        headers.append(header_text)
            
            self.logger.info(f"Found {len(headers)} week headers for {team} {year}")
            
            # Get table rows (players)
            player_data = []
            tbody = table.find('tbody')
            if tbody:
                for tr in tbody.find_all('tr'):
                    # Skip header rows within tbody
                    if tr.find('th') and tr.find('th').get('scope') == 'col':
                        continue
                    
                    cells = tr.find_all(['td', 'th'])
                    if len(cells) < 3:  # Need at least player, pos, and some weeks
                        continue
                    
                    # Get player name (first cell)
                    player_name = cells[0].get_text(strip=True)
                    if not player_name:
                        continue
                    
                    # Get position (second cell)
                    position = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                    
                    # Get injury status for each week (remaining cells)
                    week_statuses = []
                    for i, cell in enumerate(cells[2:], 1):  # Skip player name and position
                        status = cell.get_text(strip=True)
                        week_statuses.append(status)
                    
                    if any(status for status in week_statuses):  # If any injury status exists
                        player_data.append({
                            'Player': player_name,
                            'Position': position,
                            'Week_Statuses': week_statuses
                        })
            
            if not player_data:
                self.logger.warning(f"No player injury data found for {team} {year}")
                return None
            
            self.logger.info(f"Found injury data for {len(player_data)} players")
            
            # Calculate injury summary
            injury_summary = self._calculate_injury_summary(player_data, team, year)
            
            return injury_summary
            
        except Exception as e:
            self.logger.error(f"Error parsing injury table for {team} {year}: {e}")
            return None
    
    def _calculate_injury_summary(self, player_data: List[Dict], team: str, year: int) -> pd.DataFrame:
        """
        Calculate injury summary from weekly status data
        
        Args:
            player_data: List of player dictionaries with weekly status
            team: Team abbreviation
            year: Season year
            
        Returns:
            DataFrame with transposed injury summary (one row per team/year)
        """
        try:
            # Count injury statuses
            status_counts = {
                'Questionable': 0,
                'Doubtful': 0, 
                'Out': 0,
                'IR': 0,
                'PUP': 0,
                'Total_Weeks_Missed': 0
            }
            
            total_players_with_injuries = 0
            
            for player in player_data:
                player_had_injury = False
                
                for status in player['Week_Statuses']:
                    if status:  # Non-empty status
                        player_had_injury = True
                        
                        # Count different types of injury statuses
                        if status.upper() in ['O', 'OUT']:
                            status_counts['Out'] += 1
                            status_counts['Total_Weeks_Missed'] += 1
                        elif status.upper() in ['Q', 'QUESTIONABLE']:
                            status_counts['Questionable'] += 1
                        elif status.upper() in ['D', 'DOUBTFUL']:
                            status_counts['Doubtful'] += 1
                        elif status.upper() in ['IR', 'INJURED RESERVE']:
                            status_counts['IR'] += 1
                            status_counts['Total_Weeks_Missed'] += 1
                        elif status.upper() in ['PUP', 'PHYSICALLY UNABLE']:
                            status_counts['PUP'] += 1
                            status_counts['Total_Weeks_Missed'] += 1
                
                if player_had_injury:
                    total_players_with_injuries += 1
            
            # Create transposed summary DataFrame (one row per team/year)
            summary_data = {
                'Team': team.upper(),
                'Year': year,
                'Questionable': status_counts['Questionable'],
                'Doubtful': status_counts['Doubtful'],
                'Out': status_counts['Out'],
                'IR': status_counts['IR'],
                'PUP': status_counts['PUP'],
                'Total_Weeks_Missed': status_counts['Total_Weeks_Missed'],
                'Total_Players_Injured': total_players_with_injuries,
                'Scraped_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return pd.DataFrame([summary_data])
                
        except Exception as e:
            self.logger.error(f"Error calculating injury summary for {team} {year}: {e}")
            return pd.DataFrame({
                'Team': [team.upper()],
                'Year': [year],
                'Questionable': [0],
                'Doubtful': [0],
                'Out': [0],
                'IR': [0],
                'PUP': [0],
                'Total_Weeks_Missed': [0],
                'Total_Players_Injured': [0],
                'Scraped_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
    
    def _normalize_injury_type(self, injury_type: str) -> str:
        """
        Normalize injury type names for consistency
        
        Args:
            injury_type: Raw injury type string
            
        Returns:
            Normalized injury type
        """
        injury_type = injury_type.lower().strip()
        
        # Common injury type mappings
        mappings = {
            'knee': ['knee', 'acl', 'mcl', 'lcl', 'pcl', 'meniscus'],
            'ankle': ['ankle', 'achilles'],
            'shoulder': ['shoulder', 'rotator cuff', 'collarbone', 'clavicle'],
            'back': ['back', 'spine', 'disc'],
            'hamstring': ['hamstring', 'ham'],
            'quad': ['quad', 'quadriceps'],
            'calf': ['calf'],
            'groin': ['groin'],
            'hip': ['hip'],
            'wrist': ['wrist'],
            'hand': ['hand', 'finger', 'thumb'],
            'elbow': ['elbow'],
            'concussion': ['concussion', 'head', 'brain'],
            'covid': ['covid', 'coronavirus'],
            'illness': ['illness', 'sick', 'flu'],
            'personal': ['personal', 'family'],
            'suspension': ['suspension', 'suspended']
        }
        
        for category, keywords in mappings.items():
            if any(keyword in injury_type for keyword in keywords):
                return category.upper()
        
        # If no match found, return original (cleaned)
        return injury_type.upper().replace(' ', '_')
    
    def save_data(self, data: pd.DataFrame, team: str, year: int) -> bool:
        """
        Save injury data to CSV file
        
        Args:
            data: DataFrame to save
            team: Team abbreviation
            year: Season year
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            filename = f"{team}_{year}_injuries.csv"
            filepath = self.output_dir / filename
            
            data.to_csv(filepath, index=False)
            self.logger.info(f"Saved injury data to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data for {team} {year}: {e}")
            return False
    
    def scrape_teams_years(self, teams: List[str], years: List[int], skip_existing: bool = True) -> Dict[str, int]:
        """
        Scrape injury data for multiple teams and years
        
        Args:
            teams: List of team abbreviations
            years: List of years
            skip_existing: Whether to skip existing files
            
        Returns:
            Dictionary with scraping results
        """
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        total_combinations = len(teams) * len(years)
        current = 0
        
        for team in teams:
            for year in years:
                current += 1
                self.logger.info(f"Processing {team} {year} ({current}/{total_combinations})")
                
                # Check if file exists and skip if requested
                if skip_existing and self.file_exists(team, year):
                    self.logger.info(f"File exists for {team} {year}, skipping")
                    results['skipped'] += 1
                    continue
                
                # Apply rate limiting before request
                if current > 1:  # Don't delay on first request
                    self.rate_limit()
                
                # Scrape data
                data = self.scrape_team_year_injuries(team, year)
                
                if data is not None and not data.empty:
                    if self.save_data(data, team, year):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['failed'] += 1
        
        return results


def main():
    """Main function to run the injury data scraper"""
    parser = argparse.ArgumentParser(
        description='Scrape NFL injury data from Pro Football Reference'
    )
    parser.add_argument(
        '--team', 
        type=str, 
        required=True,
        help='Team abbreviation (e.g., "den") or "all" for all teams'
    )
    parser.add_argument(
        '--year', 
        type=str, 
        required=True,
        help='Year (e.g., "2024") or "all" for all years in range'
    )
    parser.add_argument(
        '--start-year', 
        type=int, 
        default=2010,
        help='Start year when using --year all (default: 2010)'
    )
    parser.add_argument(
        '--end-year', 
        type=int, 
        default=2024,
        help='End year when using --year all (default: 2024)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw/Injuries',
        help='Output directory for scraped data (default: data/raw/Injuries)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = InjuryDataScraper(output_dir=args.output_dir)
    
    # Determine teams to scrape
    if args.team.lower() == 'all':
        teams = PFR_TEAM_ABBREVIATIONS
    else:
        if args.team.lower() not in PFR_TEAM_ABBREVIATIONS:
            print(f"Error: Unknown team abbreviation '{args.team}'")
            print(f"Available teams: {', '.join(sorted(PFR_TEAM_ABBREVIATIONS))}")
            sys.exit(1)
        teams = [args.team.lower()]
    
    # Determine years to scrape
    if args.year.lower() == 'all':
        years = list(range(args.start_year, args.end_year + 1))
    else:
        try:
            year = int(args.year)
            if year < 1920 or year > 2030:
                raise ValueError("Year out of reasonable range")
            years = [year]
        except ValueError:
            print(f"Error: Invalid year '{args.year}'. Must be a valid year or 'all'")
            sys.exit(1)
    
    # Run scraper
    print(f"Starting injury data scraping for {len(teams)} team(s) and {len(years)} year(s)")
    print(f"Teams: {', '.join(teams)}")
    print(f"Years: {years[0]}-{years[-1]}" if len(years) > 1 else f"Year: {years[0]}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    results = scraper.scrape_teams_years(
        teams=teams, 
        years=years, 
        skip_existing=not args.force
    )
    
    # Print results
    print("\nScraping completed!")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Total processed: {sum(results.values())}")


if __name__ == "__main__":
    main()