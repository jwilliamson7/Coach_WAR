#!/usr/bin/env python3
"""
Pro Football Reference Combined Roster & Starters Scraper

Efficiently scrapes both roster and starters information from Pro Football Reference 
with a single web request per team/year. Includes rate limiting and command line interface.

IMPORTANT: When using year ranges, the scraper counts DOWN from max-year to min-year.
If a 404 or 403 error is encountered, the scraper skips to the next team.

Usage (run from project root directory):
    python crawlers/PFR/combined_roster_starters_scraping.py --team den --year 2024
    python crawlers/PFR/combined_roster_starters_scraping.py --all-teams --min-year 1970 --max-year 2009
    python crawlers/PFR/combined_roster_starters_scraping.py --teams den,buf,dal --year 2024
"""

import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import random
import argparse
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CombinedRosterStartersScraper:
    """Efficiently scrapes both roster and starters data from Pro Football Reference"""
    
    def __init__(self, roster_output_dir: str = "data/raw/Rosters",
                 starters_output_dir: str = "data/raw/Starters",
                 base_url: str = "https://www.pro-football-reference.com",
                 min_delay: float = 6.0, max_delay: float = 10.0):
        """
        Initialize the combined scraper
        
        Args:
            roster_output_dir: Directory to save roster data
            starters_output_dir: Directory to save starters data
            base_url: Base URL for Pro Football Reference
            min_delay: Minimum delay between requests (seconds)
            max_delay: Maximum delay between requests (seconds)
        """
        self.base_url = base_url
        self.roster_output_dir = Path(roster_output_dir)
        self.starters_output_dir = Path(starters_output_dir)
        self.roster_output_dir.mkdir(parents=True, exist_ok=True)
        self.starters_output_dir.mkdir(parents=True, exist_ok=True)
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
        try:
            time.sleep(delay)
        except KeyboardInterrupt:
            logger.info("Rate limiting interrupted by user")
            raise
    
    def _construct_url(self, team: str, year: int) -> str:
        """Construct roster URL for given team and year"""
        return f"{self.base_url}/teams/{team.lower()}/{year}_roster.htm"
    
    def _validate_team(self, team: str) -> bool:
        """Validate team abbreviation"""
        return team.lower() in self.team_abbreviations
    
    def _file_exists(self, team: str, year: int) -> Tuple[bool, bool]:
        """Check if both roster and starters files already exist"""
        team_roster_dir = self.roster_output_dir / team.upper()
        team_starters_dir = self.starters_output_dir / team.upper()
        
        roster_file = team_roster_dir / f"{team.lower()}_{year}_roster.csv"
        starters_file = team_starters_dir / f"{team.lower()}_{year}_starters.csv"
        
        return roster_file.exists(), starters_file.exists()
    
    def _fetch_page(self, team: str, year: int) -> Optional[BeautifulSoup]:
        """Fetch and parse the roster page for a given team and year"""
        url = self._construct_url(team, year)
        
        try:
            logger.info(f"Fetching {team.upper()} {year} from {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the main HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for tables in HTML comments (common in PFR)
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if '<table' in comment:
                    # Parse the comment content as HTML
                    comment_soup = BeautifulSoup(comment, 'html.parser')
                    comment_tables = comment_soup.find_all('table')
                    if comment_tables:
                        logger.info(f"Found {len(comment_tables)} tables in HTML comments")
                        # Add these tables to the main soup for processing
                        for table in comment_tables:
                            soup.append(table)
            
            return soup
            
        except requests.exceptions.HTTPError as e:
            # Check for 404 or 403 errors specifically
            if hasattr(e, 'response') and e.response is not None and e.response.status_code in [404, 403]:
                logger.warning(f"Received {e.response.status_code} error for {team.upper()} {year}: {e}")
                return "skip_team"  # Special return value to signal skipping to next team
            else:
                logger.error(f"HTTP error fetching {team.upper()} {year}: {e}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {team.upper()} {year}: {e}")
            return None
    
    def _extract_roster_table(self, soup: BeautifulSoup, team: str, year: int) -> Optional[pd.DataFrame]:
        """Extract the roster table from the page"""
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
            tables = soup.find_all('table', {'class': 'stats_table'})
            if not tables:
                tables = soup.find_all('table')
            
            logger.info(f"Found {len(tables)} tables on page, searching for roster table")
            
            largest_table = None
            largest_row_count = 0
            
            for i, table in enumerate(tables):
                # Check table headers to identify roster table
                header_text = ""
                thead = table.find('thead')
                if thead:
                    header_text = thead.get_text().lower()
                else:
                    # Check first row if no thead
                    first_row = table.find('tr')
                    if first_row:
                        header_text = first_row.get_text().lower()
                
                # Look for roster-specific keywords
                roster_keywords = ['player', 'no.', 'pos', 'age', 'wt', 'ht', 'college', 'exp', 'born']
                if any(keyword in header_text for keyword in roster_keywords):
                    # Count rows to find the largest table (complete roster vs starters)
                    tbody = table.find('tbody')
                    if tbody:
                        row_count = len(tbody.find_all('tr'))
                        logger.info(f"Table #{i} has {row_count} rows with headers: {header_text[:100]}")
                        
                        if row_count > largest_row_count:
                            largest_row_count = row_count
                            largest_table = table
            
            roster_table = largest_table
        
        if not roster_table:
            logger.warning(f"No roster table found for {team.upper()} {year}")
            return None
        
        # Extract table data
        try:
            # Get headers
            headers = []
            thead = roster_table.find('thead')
            if thead:
                header_rows = thead.find_all('tr')
                if header_rows:
                    # Use the last header row (may have multiple header levels)
                    for th in header_rows[-1].find_all(['th', 'td']):
                        headers.append(th.get_text(strip=True))
            
            # Get data rows
            rows = []
            tbody = roster_table.find('tbody')
            if tbody:
                data_rows = tbody.find_all('tr')
            else:
                # Fallback: get all rows except header rows
                all_rows = roster_table.find_all('tr')
                data_rows = all_rows[1:] if len(all_rows) > 1 else []
            
            for row in data_rows:
                # Skip any section header rows
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if any(cell.strip() for cell in row_data):  # Skip empty rows
                        rows.append(row_data)
            
            if not rows:
                logger.warning(f"No data rows found in roster table for {team.upper()} {year}")
                return None
            
            # Ensure consistent column count
            max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
            
            # Pad headers and rows if needed
            while len(headers) < max_cols:
                headers.append(f'Column_{len(headers)}')
            
            for row in rows:
                while len(row) < max_cols:
                    row.append('')
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers[:max_cols])
            df.insert(0, 'Team', team.upper())
            df.insert(1, 'Year', year)
            
            logger.info(f"Successfully extracted roster data: {len(df)} players")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting roster table for {team.upper()} {year}: {e}")
            return None
    
    def _extract_starters_tables(self, soup: BeautifulSoup, team: str, year: int) -> Optional[pd.DataFrame]:
        """Extract starters tables from the page"""
        starters_tables = []
        
        # Try different possible table identifiers for starters
        possible_ids = ['starters', 'team_starters', 'offensive_starters', 'defensive_starters']
        
        for table_id in possible_ids:
            table = soup.find('table', {'id': table_id})
            if table:
                logger.info(f"Found starters table with id='{table_id}'")
                starters_tables.append(table)
        
        # If not found by ID, look for tables with starters-related headers
        if not starters_tables:
            tables = soup.find_all('table', {'class': 'stats_table'})
            if not tables:
                tables = soup.find_all('table')
            
            logger.info(f"Found {len(tables)} tables on page, searching for starters tables")
            
            for i, table in enumerate(tables):
                # Check if this table has starters-like headers
                header_text = ""
                thead = table.find('thead')
                if thead:
                    header_text = thead.get_text().lower()
                else:
                    # Check first row if no thead
                    first_row = table.find('tr')
                    if first_row:
                        header_text = first_row.get_text().lower()
                
                # Look for starters-specific keywords
                starters_keywords = ['pos', 'starter', 'games started', 'gs', 'start']
                if any(keyword in header_text for keyword in starters_keywords):
                    # Additional check: make sure it's not the main roster table
                    tbody = table.find('tbody') or table
                    row_count = len(tbody.find_all('tr')) if tbody else 0
                    
                    # Starters tables are typically smaller than full roster (< 30 players usually)
                    if row_count < 60:  # Arbitrary threshold to distinguish from full roster
                        starters_tables.append(table)
                        logger.info(f"Found potential starters table #{i} with {row_count} rows")
        
        # Also look for tables near "Starters" headings
        starters_divs = soup.find_all(['div', 'h2', 'h3'], string=lambda text: text and 'starter' in text.lower())
        for div in starters_divs:
            parent = div.parent
            if parent:
                nearby_tables = parent.find_all('table')
                for table in nearby_tables:
                    if table not in starters_tables:
                        starters_tables.append(table)
                        logger.info(f"Found starters table near div: {div.get_text()[:50]}")
        
        if not starters_tables:
            logger.warning(f"No starters tables found for {team.upper()} {year}")
            return None
        
        # Process all starters tables and combine them
        all_starters_data = []
        
        for table_idx, starters_table in enumerate(starters_tables):
            logger.info(f"Processing starters table {table_idx + 1}/{len(starters_tables)}")
            
            try:
                # Extract table headers
                headers = []
                header_row = starters_table.find('thead')
                if header_row:
                    # Get the last header row (may have multiple levels)
                    header_rows = header_row.find_all('tr')
                    if header_rows:
                        for th in header_rows[-1].find_all(['th', 'td']):
                            headers.append(th.get_text(strip=True))
                else:
                    # Try first row as headers
                    first_row = starters_table.find('tr')
                    if first_row:
                        for cell in first_row.find_all(['th', 'td']):
                            headers.append(cell.get_text(strip=True))
                
                # Extract data rows
                rows = []
                tbody = starters_table.find('tbody')
                if tbody:
                    data_rows = tbody.find_all('tr')
                else:
                    # Fallback: get all rows except the first (header)
                    all_rows = starters_table.find_all('tr')
                    data_rows = all_rows[1:] if len(all_rows) > 1 else []
                
                for row in data_rows:
                    # Skip section header rows
                    if row.get('class') and 'thead' in row.get('class'):
                        continue
                    
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        if any(cell.strip() for cell in row_data):  # Skip empty rows
                            # Filter out section header rows like "Offensive Starters", "Defensive Starters"
                            if len(row_data) > 0:
                                first_cell = row_data[0].lower()
                                # Skip rows that are section headers
                                if first_cell in ['offensive starters', 'defensive starters', 'special teams', 'starters', '']:
                                    continue
                                # Also skip rows where the first cell contains only section text and other cells are empty
                                if 'starter' in first_cell and len([cell for cell in row_data[1:] if cell.strip()]) == 0:
                                    continue
                            rows.append(row_data)
                
                if rows:
                    # Ensure consistent column count
                    max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
                    
                    while len(headers) < max_cols:
                        headers.append(f'Column_{len(headers)}')
                    
                    for row in rows:
                        while len(row) < max_cols:
                            row.append('')
                    
                    # Create DataFrame for this table
                    df_table = pd.DataFrame(rows, columns=headers[:max_cols])
                    
                    # Add table identifier (for multiple starters tables)
                    df_table['Table_Type'] = f"Starters_{table_idx + 1}"
                    
                    all_starters_data.append(df_table)
                    logger.info(f"Extracted {len(df_table)} starters from table {table_idx + 1}")
                    
            except Exception as e:
                logger.error(f"Error processing starters table {table_idx + 1} for {team.upper()} {year}: {e}")
                continue
        
        if not all_starters_data:
            logger.warning(f"No starters data extracted for {team.upper()} {year}")
            return None
        
        # Combine all starters tables
        combined_df = pd.concat(all_starters_data, ignore_index=True)
        combined_df.insert(0, 'Team', team.upper())
        combined_df.insert(1, 'Year', year)
        
        logger.info(f"Successfully combined starters data: {len(combined_df)} total entries")
        return combined_df
    
    def _save_data(self, roster_df: Optional[pd.DataFrame], starters_df: Optional[pd.DataFrame], 
                   team: str, year: int) -> Tuple[bool, bool]:
        """Save roster and starters data to CSV files"""
        roster_saved = False
        starters_saved = False
        
        # Save roster data
        if roster_df is not None:
            try:
                team_roster_dir = self.roster_output_dir / team.upper()
                team_roster_dir.mkdir(parents=True, exist_ok=True)
                roster_file = team_roster_dir / f"{team.lower()}_{year}_roster.csv"
                roster_df.to_csv(roster_file, index=False)
                logger.info(f"Saved roster data to {roster_file.absolute()}")
                roster_saved = True
            except Exception as e:
                logger.error(f"Error saving roster data for {team.upper()} {year}: {e}")
        
        # Save starters data
        if starters_df is not None:
            try:
                team_starters_dir = self.starters_output_dir / team.upper()
                team_starters_dir.mkdir(parents=True, exist_ok=True)
                starters_file = team_starters_dir / f"{team.lower()}_{year}_starters.csv"
                starters_df.to_csv(starters_file, index=False)
                logger.info(f"Saved starters data to {starters_file.absolute()}")
                starters_saved = True
            except Exception as e:
                logger.error(f"Error saving starters data for {team.upper()} {year}: {e}")
        
        return roster_saved, starters_saved
    
    def scrape_team_year(self, team: str, year: int, force: bool = False) -> Tuple[bool, bool, Any]:
        """
        Scrape both roster and starters data for a specific team and year
        
        Args:
            team: Team abbreviation
            year: Year to scrape
            force: If True, overwrite existing files
            
        Returns:
            Tuple of (roster_success, starters_success, made_request)
            made_request can be True, False, or "skip_team"
        """
        if not self._validate_team(team):
            logger.error(f"Invalid team abbreviation: {team}")
            return False, False, False
        
        # Check if files already exist
        roster_exists, starters_exists = self._file_exists(team, year)
        
        if not force and roster_exists and starters_exists:
            logger.info(f"Both roster and starters data for {team.upper()} {year} already exist. Skipping web request.")
            return True, True, False
        
        logger.info(f"\n=== Scraping {team.upper()} {year} ===")
        
        # Fetch the page once
        soup = self._fetch_page(team, year)
        if soup == "skip_team":
            return False, False, "skip_team"  # Signal to skip remaining years for this team
        if not soup:
            return False, False, True
        
        # Extract both roster and starters data from the same page
        roster_df = None
        starters_df = None
        
        if not roster_exists or force:
            roster_df = self._extract_roster_table(soup, team, year)
        else:
            logger.info(f"Roster data for {team.upper()} {year} already exists, skipping extraction")
        
        if not starters_exists or force:
            starters_df = self._extract_starters_tables(soup, team, year)
        else:
            logger.info(f"Starters data for {team.upper()} {year} already exists, skipping extraction")
        
        # Save the data
        roster_saved, starters_saved = self._save_data(roster_df, starters_df, team, year)
        
        # Report results
        if roster_df is not None and starters_df is not None:
            logger.info(f"Successfully scraped {team.upper()} {year}: {len(roster_df)} roster, {len(starters_df)} starters")
        elif roster_df is not None:
            logger.info(f"Successfully scraped {team.upper()} {year} roster: {len(roster_df)} players")
        elif starters_df is not None:
            logger.info(f"Successfully scraped {team.upper()} {year} starters: {len(starters_df)} entries")
        else:
            logger.warning(f"No data extracted for {team.upper()} {year}")
        
        return (roster_saved or roster_exists), (starters_saved or starters_exists), True
    
    def scrape_multiple(self, teams: List[str], years: List[int], force: bool = False) -> Dict[str, Any]:
        """
        Scrape multiple teams and years
        
        Args:
            teams: List of team abbreviations
            years: List of years to scrape
            force: If True, overwrite existing files
            
        Returns:
            Dictionary with scraping results and statistics
        """
        total_requests = len(teams) * len(years)
        completed_requests = 0
        web_requests_made = 0
        roster_successes = 0
        starters_successes = 0
        failed_requests = []
        skipped_requests = 0
        
        logger.info(f"Starting scraping for {len(teams)} teams, {len(years)} years ({total_requests} total requests)")
        logger.info(f"Roster output directory: {self.roster_output_dir.absolute()}")
        logger.info(f"Starters output directory: {self.starters_output_dir.absolute()}")
        
        try:
            for team_idx, team in enumerate(teams):
                logger.info(f"Processing team {team_idx + 1}/{len(teams)}: {team.upper()}")
                
                # Process years in reverse order (countdown from max to min)
                for year_idx, year in enumerate(reversed(years)):
                    try:
                        roster_success, starters_success, made_request = self.scrape_team_year(team, year, force)
                        
                        # Check if we got a signal to skip to next team
                        if made_request == "skip_team":
                            logger.info(f"Skipping remaining years for {team.upper()} due to 404/403 error")
                            # Mark remaining years as skipped
                            remaining_years = len(years) - year_idx - 1
                            completed_requests += remaining_years
                            skipped_requests += remaining_years
                            break  # Move to next team
                        
                        if roster_success:
                            roster_successes += 1
                        if starters_success:
                            starters_successes += 1
                        
                        if made_request and made_request != "skip_team":
                            web_requests_made += 1
                            if not roster_success and not starters_success:
                                failed_requests.append((team, year))
                        else:
                            skipped_requests += 1
                        
                        completed_requests += 1
                        
                        # Only rate limit if we made a web request and it's not the last request
                        if made_request and made_request != "skip_team" and completed_requests < total_requests:
                            self._rate_limit()
                            
                    except KeyboardInterrupt:
                        logger.info(f"\nScraping interrupted by user after {completed_requests}/{total_requests} requests")
                        raise  # Re-raise to break out of outer loop
                    except Exception as e:
                        logger.error(f"Unexpected error for {team.upper()} {year}: {e}")
                        failed_requests.append((team, year))
                        completed_requests += 1
                        continue
                        
        except KeyboardInterrupt:
            logger.info("Exiting due to user interruption...")
        
        # Report summary
        logger.info(f"\n=== Scraping Summary ===")
        logger.info(f"Total requests: {total_requests}")
        logger.info(f"Completed requests: {completed_requests}")
        logger.info(f"Web requests made: {web_requests_made}")
        logger.info(f"Skipped (files exist): {skipped_requests}")
        logger.info(f"Roster successes: {roster_successes}")
        logger.info(f"Starters successes: {starters_successes}")
        logger.info(f"Failed requests: {len(failed_requests)}")
        
        if failed_requests:
            logger.info("Failed requests:")
            for team, year in failed_requests[:10]:  # Show first 10
                logger.info(f"  {team.upper()} {year}")
            if len(failed_requests) > 10:
                logger.info(f"  ... and {len(failed_requests) - 10} more")
        
        return {
            'total_requests': total_requests,
            'completed_requests': completed_requests,
            'web_requests_made': web_requests_made,
            'skipped_requests': skipped_requests,
            'roster_successes': roster_successes,
            'starters_successes': starters_successes,
            'failed_requests': failed_requests
        }


def main():
    """Main function to run the combined scraper"""
    parser = argparse.ArgumentParser(description='Scrape NFL roster and starters data from Pro Football Reference')
    
    # Team arguments
    parser.add_argument('--team', type=str, help='Single team abbreviation (e.g., "den")')
    parser.add_argument('--teams', type=str, help='Comma-separated team abbreviations (e.g., "den,buf,dal")')
    parser.add_argument('--all-teams', action='store_true', help='Scrape all teams')
    
    # Year arguments
    parser.add_argument('--year', type=int, help='Single year to scrape')
    parser.add_argument('--min-year', type=int, help='Minimum year (inclusive) - will count down from max-year to this')
    parser.add_argument('--max-year', type=int, help='Maximum year (inclusive) - will start from this year and count down')
    
    # Output directories
    parser.add_argument('--roster-output-dir', type=str, default='data/raw/Rosters',
                       help='Output directory for roster data')
    parser.add_argument('--starters-output-dir', type=str, default='data/raw/Starters',
                       help='Output directory for starters data')
    
    # Rate limiting
    parser.add_argument('--min-delay', type=float, default=6.0,
                       help='Minimum delay between requests in seconds (default: 6.0)')
    parser.add_argument('--max-delay', type=float, default=10.0,
                       help='Maximum delay between requests in seconds (default: 10.0)')
    
    # Options
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing files')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = CombinedRosterStartersScraper(
        roster_output_dir=args.roster_output_dir,
        starters_output_dir=args.starters_output_dir,
        min_delay=args.min_delay,
        max_delay=args.max_delay
    )
    
    # Determine teams
    if args.all_teams:
        teams = scraper.team_abbreviations
    elif args.teams:
        teams = [t.strip().lower() for t in args.teams.split(',')]
        # Validate teams
        invalid_teams = [t for t in teams if not scraper._validate_team(t)]
        if invalid_teams:
            logger.error(f"Invalid team abbreviations: {invalid_teams}")
            logger.error(f"Valid teams: {', '.join(scraper.team_abbreviations)}")
            sys.exit(1)
    elif args.team:
        teams = [args.team.lower()]
        if not scraper._validate_team(teams[0]):
            logger.error(f"Invalid team abbreviation: {args.team}")
            logger.error(f"Valid teams: {', '.join(scraper.team_abbreviations)}")
            sys.exit(1)
    else:
        logger.error("Must specify --team, --teams, or --all-teams")
        sys.exit(1)
    
    # Determine years
    if args.year:
        years = [args.year]
    elif args.min_year and args.max_year:
        years = list(range(args.min_year, args.max_year + 1))  # Still create ascending range for processing
    else:
        logger.error("Must specify --year or both --min-year and --max-year")
        sys.exit(1)
    
    # Run the scraper
    try:
        results = scraper.scrape_multiple(teams, years, args.force)
        
        # Exit with error code if too many failures
        failure_rate = len(results['failed_requests']) / results['total_requests'] if results['total_requests'] > 0 else 0
        if failure_rate > 0.1:  # More than 10% failure rate
            logger.warning(f"High failure rate: {failure_rate:.1%}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()