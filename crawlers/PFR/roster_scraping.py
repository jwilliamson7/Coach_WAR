#!/usr/bin/env python3
"""
Pro Football Reference Roster Scraper

Scrapes roster information from Pro Football Reference for specified teams and years.
Includes rate limiting and command line interface for flexible execution.

Usage:
    python roster_scraping.py --team den --year 2024
    python roster_scraping.py --start-year 2020 --end-year 2024
    python roster_scraping.py --teams den,buf,dal --year 2024
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import random
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RosterScraper:
    """Scrapes roster data from Pro Football Reference"""
    
    def __init__(self, base_url: str = "https://www.pro-football-reference.com",
                 output_dir: str = "data/raw/Rosters",
                 min_delay: float = 6.0, max_delay: float = 10.0):
        """
        Initialize the roster scraper
        
        Args:
            base_url: Base URL for Pro Football Reference
            output_dir: Directory to save roster data
            min_delay: Minimum delay between requests (seconds)
            max_delay: Maximum delay between requests (seconds)
        """
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Team abbreviations used by PFR (correct abbreviations)
        self.team_abbreviations = [
            'crd',  # Arizona Cardinals (not 'ari')
            'atl',  # Atlanta Falcons
            'rav',  # Baltimore Ravens (not 'bal')
            'buf',  # Buffalo Bills
            'car',  # Carolina Panthers
            'chi',  # Chicago Bears
            'cin',  # Cincinnati Bengals
            'cle',  # Cleveland Browns
            'clt',  # Indianapolis Colts (not 'ind')
            'dal',  # Dallas Cowboys
            'den',  # Denver Broncos
            'det',  # Detroit Lions
            'gnb',  # Green Bay Packers (not 'gb')
            'htx',  # Houston Texans (not 'hou')
            'jax',  # Jacksonville Jaguars
            'kan',  # Kansas City Chiefs (not 'kc')
            'mia',  # Miami Dolphins
            'min',  # Minnesota Vikings
            'nwe',  # New England Patriots
            'nor',  # New Orleans Saints (not 'no')
            'nyg',  # New York Giants
            'nyj',  # New York Jets
            'oti',  # Tennessee Titans (not 'ten')
            'phi',  # Philadelphia Eagles
            'pit',  # Pittsburgh Steelers
            'rai',  # Las Vegas Raiders (not 'lv' or 'oak')
            'ram',  # Los Angeles Rams
            'sdg',  # Los Angeles Chargers (not 'lac')
            'sfo',  # San Francisco 49ers (not 'sf')
            'sea',  # Seattle Seahawks
            'tam',  # Tampa Bay Buccaneers (not 'tb')
            'was'   # Washington (not 'wsh')
        ]
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        delay = random.uniform(self.min_delay, self.max_delay)
        logger.info(f"Rate limiting: waiting {delay:.1f} seconds...")
        time.sleep(delay)
    
    def _construct_url(self, team: str, year: int) -> str:
        """Construct roster URL for given team and year"""
        return f"{self.base_url}/teams/{team.lower()}/{year}_roster.htm"
    
    def _scrape_roster_page(self, team: str, year: int) -> Optional[pd.DataFrame]:
        """
        Scrape roster data for a specific team and year
        
        Args:
            team: Team abbreviation (e.g., 'den')
            year: Year to scrape
            
        Returns:
            DataFrame with roster data or None if failed
        """
        url = self._construct_url(team, year)
        
        try:
            logger.info(f"Scraping {team.upper()} {year} roster from {url}")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Also check for tables hidden in HTML comments (common on PFR)
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if 'roster' in comment.lower() or '<table' in comment:
                    # Parse the comment content as HTML
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    comment_tables = comment_soup.find_all('table')
                    if comment_tables:
                        logger.info(f"Found {len(comment_tables)} tables in HTML comments")
                        # Add these tables to the main soup for processing
                        for table in comment_tables:
                            soup.append(table)
            
            # Find the roster table - look for the complete roster, not starters
            roster_table = None
            
            # Try different possible table identifiers for complete roster
            possible_ids = ['roster', 'games_played_team', 'team_roster']
            
            for table_id in possible_ids:
                roster_table = soup.find('table', {'id': table_id})
                if roster_table:
                    logger.info(f"Found roster table with id='{table_id}'")
                    break
            
            # If not found by ID, look for all tables and find the largest one that looks like a roster
            if not roster_table:
                # First try stats_table class, then all tables
                tables = soup.find_all('table', {'class': 'stats_table'})
                if not tables:
                    tables = soup.find_all('table')
                
                logger.info(f"Found {len(tables)} tables on page")
                
                largest_table = None
                largest_row_count = 0
                
                for i, table in enumerate(tables):
                    # Check if this table has roster-like headers
                    headers = table.find('thead')
                    if headers:
                        header_text = headers.get_text().lower()
                        # Look for tables with typical roster headers
                        roster_keywords = ['player', 'no.', 'pos', 'age', 'wt', 'ht', 'college', 'exp', 'born']
                        if any(keyword in header_text for keyword in roster_keywords):
                            # Count rows to find the largest table (complete roster vs starters)
                            tbody = table.find('tbody')
                            if tbody:
                                row_count = len(tbody.find_all('tr'))
                                logger.info(f"Table #{i} has {row_count} rows with headers: {header_text[:100]}")
                                
                                # Look for full roster (typically 50+ players) vs starters (~24 players)
                                if row_count > largest_row_count:
                                    largest_table = table
                                    largest_row_count = row_count
                    else:
                        # Check tables without thead
                        tbody = table.find('tbody') or table
                        if tbody:
                            rows = tbody.find_all('tr')
                            if rows and len(rows) > 0:
                                # Check first row for header-like content
                                first_row = rows[0]
                                row_text = first_row.get_text().lower()
                                if any(keyword in row_text for keyword in ['player', 'no.', 'pos', 'age', 'wt', 'ht', 'college']):
                                    row_count = len(rows)
                                    logger.info(f"Table #{i} (no thead) has {row_count} rows with first row: {row_text[:100]}")
                                    if row_count > largest_row_count:
                                        largest_table = table
                                        largest_row_count = row_count
                
                if largest_table is not None:
                    roster_table = largest_table
                    logger.info(f"Selected largest roster table with {largest_row_count} rows")
            
            if not roster_table:
                logger.warning(f"No roster table found for {team.upper()} {year}")
                # Debug: show available table IDs and row counts
                tables = soup.find_all('table')
                table_info = []
                for i, table in enumerate(tables):
                    table_id = table.get('id', 'no-id')
                    table_class = table.get('class', 'no-class')
                    tbody = table.find('tbody')
                    row_count = len(tbody.find_all('tr')) if tbody else 0
                    table_info.append(f"Table #{i}: ID={table_id}, Class={table_class}, Rows={row_count}")
                logger.info(f"Available tables: {'; '.join(table_info)}")
                return None
            
            # Extract table headers
            headers = []
            header_row = roster_table.find('thead')
            if header_row:
                header_cells = header_row.find_all(['th', 'td'])
                headers = [cell.get_text(strip=True) for cell in header_cells]
            
            if not headers:
                logger.warning(f"No headers found for {team.upper()} {year} roster table")
                return None
            
            # Extract table rows
            rows = []
            tbody = roster_table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_data = []
                        for cell in cells:
                            # Get text content, preserving links for player names
                            text = cell.get_text(strip=True)
                            row_data.append(text)
                        
                        if len(row_data) == len(headers):
                            rows.append(row_data)
            
            if not rows:
                logger.warning(f"No data rows found for {team.upper()} {year}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Add metadata columns
            df['Team'] = team.upper()
            df['Year'] = year
            df['Scraped_Date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"Successfully scraped {len(df)} players for {team.upper()} {year}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {team.upper()} {year}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {team.upper()} {year}: {e}")
            return None
    
    def _save_roster_data(self, df: pd.DataFrame, team: str, year: int):
        """Save roster data to CSV file"""
        if df is None or df.empty:
            logger.warning(f"No data to save for {team.upper()} {year}")
            return
        
        # Create team directory
        team_dir = self.output_dir / team.upper()
        team_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        filename = f"{team.lower()}_{year}_roster.csv"
        filepath = team_dir / filename
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved roster data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving roster data for {team.upper()} {year}: {e}")
    
    def scrape_team_year(self, team: str, year: int) -> bool:
        """
        Scrape roster for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Year to scrape
            
        Returns:
            True if successful, False otherwise
        """
        if team.lower() not in self.team_abbreviations:
            logger.error(f"Invalid team abbreviation: {team}")
            return False
        
        # Check if file already exists BEFORE any network calls
        team_dir = self.output_dir / team.upper()
        filename = f"{team.lower()}_{year}_roster.csv"
        filepath = team_dir / filename
        
        if filepath.exists():
            logger.info(f"Roster file already exists for {team.upper()} {year}, skipping...")
            return True
        
        df = self._scrape_roster_page(team, year)
        if df is not None:
            self._save_roster_data(df, team, year)
            return True
        return False
    
    def scrape_teams_year(self, teams: List[str], year: int) -> Dict[str, bool]:
        """
        Scrape rosters for multiple teams in a single year
        
        Args:
            teams: List of team abbreviations
            year: Year to scrape
            
        Returns:
            Dictionary mapping team to success status
        """
        results = {}
        
        for i, team in enumerate(teams):
            logger.info(f"Processing team {i+1}/{len(teams)}: {team.upper()}")
            
            # Check if file already exists before rate limiting
            team_dir = self.output_dir / team.upper()
            filename = f"{team.lower()}_{year}_roster.csv"
            filepath = team_dir / filename
            
            if filepath.exists():
                logger.info(f"Roster file already exists for {team.upper()} {year}, skipping...")
                results[team] = True
                continue  # Skip rate limiting for existing files
            
            results[team] = self.scrape_team_year(team, year)
            
            # Rate limit between teams (except for last team) - only for actual scraping
            if i < len(teams) - 1:
                self._rate_limit()
        
        return results
    
    def scrape_team_years(self, team: str, start_year: int, end_year: int) -> Dict[int, bool]:
        """
        Scrape rosters for a single team across multiple years
        
        Args:
            team: Team abbreviation
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            
        Returns:
            Dictionary mapping year to success status
        """
        results = {}
        years = list(range(start_year, end_year + 1))
        
        for i, year in enumerate(years):
            logger.info(f"Processing year {i+1}/{len(years)}: {year}")
            
            # Check if file already exists before rate limiting
            team_dir = self.output_dir / team.upper()
            filename = f"{team.lower()}_{year}_roster.csv"
            filepath = team_dir / filename
            
            if filepath.exists():
                logger.info(f"Roster file already exists for {team.upper()} {year}, skipping...")
                results[year] = True
                continue  # Skip rate limiting for existing files
            
            results[year] = self.scrape_team_year(team, year)
            
            # Rate limit between years (except for last year) - only for actual scraping
            if i < len(years) - 1:
                self._rate_limit()
        
        return results
    
    def scrape_teams_years(self, teams: List[str], start_year: int, end_year: int) -> Dict[str, Dict[int, bool]]:
        """
        Scrape rosters for multiple teams across multiple years
        
        Args:
            teams: List of team abbreviations
            start_year: Starting year (inclusive)
            end_year: Ending year (inclusive)
            
        Returns:
            Nested dictionary mapping team -> year -> success status
        """
        results = {}
        last_team_scraped = False
        
        for i, team in enumerate(teams):
            logger.info(f"Processing team {i+1}/{len(teams)}: {team.upper()}")
            
            # Check if this team has any missing files
            team_needs_scraping = False
            team_dir = self.output_dir / team.upper()
            for year in range(start_year, end_year + 1):
                filename = f"{team.lower()}_{year}_roster.csv"
                filepath = team_dir / filename
                if not filepath.exists():
                    team_needs_scraping = True
                    break
            
            team_results = self.scrape_team_years(team, start_year, end_year)
            results[team] = team_results
            
            # Only rate limit between teams if the previous team did actual scraping
            # and this is not the last team
            if i < len(teams) - 1 and last_team_scraped:
                self._rate_limit()
            
            last_team_scraped = team_needs_scraping
        
        return results


def parse_team_list(team_string: str) -> List[str]:
    """Parse comma-separated team string into list"""
    if not team_string:
        return []
    return [team.strip().lower() for team in team_string.split(',')]


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Scrape NFL roster data from Pro Football Reference')
    
    # Team arguments
    parser.add_argument('--team', type=str, help='Single team abbreviation (e.g., "den")')
    parser.add_argument('--teams', type=str, help='Comma-separated team abbreviations (e.g., "den,buf,dal")')
    parser.add_argument('--all-teams', action='store_true', help='Scrape all teams')
    
    # Year arguments
    parser.add_argument('--year', type=int, help='Single year to scrape')
    parser.add_argument('--start-year', type=int, help='Starting year (inclusive)')
    parser.add_argument('--end-year', type=int, help='Ending year (inclusive)')
    
    # Rate limiting
    parser.add_argument('--min-delay', type=float, default=6.0, 
                       help='Minimum delay between requests in seconds (default: 6.0)')
    parser.add_argument('--max-delay', type=float, default=10.0,
                       help='Maximum delay between requests in seconds (default: 10.0)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='data/raw/Rosters',
                       help='Output directory for roster data')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.team, args.teams, args.all_teams]):
        logger.error("Must specify --team, --teams, or --all-teams")
        sys.exit(1)
    
    if not any([args.year, (args.start_year and args.end_year)]):
        logger.error("Must specify --year or both --start-year and --end-year")
        sys.exit(1)
    
    if args.year and (args.start_year or args.end_year):
        logger.error("Cannot specify both --year and --start-year/--end-year")
        sys.exit(1)
    
    # Initialize scraper
    scraper = RosterScraper(
        output_dir=args.output_dir,
        min_delay=args.min_delay,
        max_delay=args.max_delay
    )
    
    # Determine teams to scrape
    if args.all_teams:
        teams = scraper.team_abbreviations
    elif args.teams:
        teams = parse_team_list(args.teams)
    elif args.team:
        teams = [args.team.lower()]
    
    # Validate teams
    invalid_teams = [team for team in teams if team not in scraper.team_abbreviations]
    if invalid_teams:
        logger.error(f"Invalid team abbreviations: {invalid_teams}")
        logger.info(f"Valid teams: {', '.join(scraper.team_abbreviations)}")
        sys.exit(1)
    
    # Determine years to scrape
    if args.year:
        start_year = end_year = args.year
    else:
        start_year = args.start_year
        end_year = args.end_year
    
    # Execute scraping
    logger.info(f"Starting roster scraping for {len(teams)} teams, years {start_year}-{end_year}")
    
    if len(teams) == 1 and start_year == end_year:
        # Single team, single year
        success = scraper.scrape_team_year(teams[0], start_year)
        logger.info(f"Scraping completed. Success: {success}")
    
    elif len(teams) == 1:
        # Single team, multiple years
        results = scraper.scrape_team_years(teams[0], start_year, end_year)
        successes = sum(results.values())
        logger.info(f"Scraping completed. {successes}/{len(results)} years successful")
    
    elif start_year == end_year:
        # Multiple teams, single year
        results = scraper.scrape_teams_year(teams, start_year)
        successes = sum(results.values())
        logger.info(f"Scraping completed. {successes}/{len(results)} teams successful")
    
    else:
        # Multiple teams, multiple years
        results = scraper.scrape_teams_years(teams, start_year, end_year)
        total_attempts = sum(len(team_results) for team_results in results.values())
        total_successes = sum(
            sum(team_results.values()) for team_results in results.values()
        )
        logger.info(f"Scraping completed. {total_successes}/{total_attempts} total attempts successful")
    
    logger.info("Roster scraping finished!")


if __name__ == "__main__":
    main()