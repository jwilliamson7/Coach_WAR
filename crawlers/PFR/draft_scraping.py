import requests
import time
import os
import pandas as pd
import argparse
import sys
from random import randint
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Optional, Dict


class DraftDataScraper:
    """Scrapes NFL draft data from pro-football-reference.com"""
    
    BASE_URL = 'https://www.pro-football-reference.com/years'
    
    def __init__(self, output_dir: str = "../../data/raw/Draft"):
        """Initialize the scraper with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _rate_limit(self, min_delay: int = 6, max_delay: int = 10):
        """Apply rate limiting between requests"""
        delay = randint(min_delay, max_delay)
        print(f"Waiting {delay} seconds before next request...")
        time.sleep(delay)
        
    def _fetch_draft_page(self, year: int) -> Optional[BeautifulSoup]:
        """Fetch and parse the draft page for a given year"""
        url = f"{self.BASE_URL}/{year}/draft.htm"
        
        try:
            print(f"Fetching draft data for year {year}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching draft data for year {year}: {e}")
            return None
            
    def _extract_draft_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """Extract the main draft table from the page"""
        try:
            # Look for the draft table - typically has id 'drafts'
            draft_table = soup.find('table', {'id': 'drafts'})
            
            if not draft_table:
                # Fallback: look for any table with draft data
                draft_table = soup.find('table')
                
            if not draft_table:
                print("No draft table found on page")
                return None
                
            # Extract headers from thead
            thead = draft_table.find('thead')
            headers = []
            
            if thead:
                # Get all header rows (some tables have multiple header rows)
                header_rows = thead.find_all('tr')
                if header_rows:
                    # Use the last header row (usually contains the actual column names)
                    header_row = header_rows[-1]
                    for th in header_row.find_all(['th', 'td']):
                        header_text = th.get_text(strip=True)
                        # Handle empty headers or special characters
                        if not header_text or header_text in ['', ' ']:
                            header_text = f'Column_{len(headers)}'
                        headers.append(header_text)
                        
            # If no headers found in thead, try first row
            if not headers:
                first_row = draft_table.find('tr')
                if first_row:
                    for cell in first_row.find_all(['th', 'td']):
                        header_text = cell.get_text(strip=True)
                        if not header_text:
                            header_text = f'Column_{len(headers)}'
                        headers.append(header_text)
                        
            # Extract data rows from tbody
            tbody = draft_table.find('tbody')
            rows = []
            
            if tbody:
                data_rows = tbody.find_all('tr')
            else:
                # Fallback: get all rows except the first one (assumed to be header)
                all_rows = draft_table.find_all('tr')
                data_rows = all_rows[1:] if len(all_rows) > 1 else []
                
            for row in data_rows:
                # Skip header rows within tbody (class="thead")
                if row.get('class') and 'thead' in row.get('class'):
                    continue
                    
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    for cell in cells:
                        # Get text and clean it
                        text = cell.get_text(strip=True)
                        # Handle empty cells
                        if not text:
                            text = ''
                        row_data.append(text)
                    
                    # Only add rows with actual data
                    if any(cell.strip() for cell in row_data):
                        rows.append(row_data)
                        
            if not rows:
                print("No data rows found in draft table")
                return None
                
            # Ensure we have the right number of columns
            max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
            
            # Pad headers if needed
            while len(headers) < max_cols:
                headers.append(f'Column_{len(headers)}')
                
            # Pad rows if needed
            for row in rows:
                while len(row) < max_cols:
                    row.append('')
                    
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers[:max_cols])
            
            # Year will be added by the calling function
            # df.insert(0, 'Draft_Year', year)
            
            return df
            
        except Exception as e:
            print(f"Error extracting draft table: {e}")
            return None
            
    def _clean_draft_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the draft data"""
        if df is None or df.empty:
            return df
            
        df_cleaned = df.copy()
        
        # Clean common column names
        column_mapping = {
            'Rnd': 'Round',
            'Pick': 'Pick_Number', 
            'Tm': 'Team',
            'Pos': 'Position',
            'To': 'Last_Season',
            'AP1': 'All_Pro_Selections',
            'PB': 'Pro_Bowl_Selections',
            'St': 'Games_Started',
            'wAV': 'Weighted_AV',
            'DrAV': 'Draft_Rookie_AV'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df_cleaned.columns:
                df_cleaned = df_cleaned.rename(columns={old_name: new_name})
                
        return df_cleaned
        
    def _save_draft_data(self, df: pd.DataFrame, year: int):
        """Save the draft data to CSV"""
        filename = f"draft_{year}.csv"
        filepath = self.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved draft data for {year} to {filepath}")
        except Exception as e:
            print(f"Error saving draft data for {year}: {e}")
            
    def scrape_draft_year(self, year: int) -> bool:
        """Scrape draft data for a specific year"""
        print(f"\n=== Scraping draft data for {year} ===")
        
        # Check if file already exists
        filename = f"draft_{year}.csv"
        filepath = self.output_dir / filename
        if filepath.exists():
            print(f"Draft data for {year} already exists. Skipping...")
            return True
            
        soup = self._fetch_draft_page(year)
        if not soup:
            return False
            
        df = self._extract_draft_table(soup)
        if df is None or df.empty:
            print(f"No draft data extracted for {year}")
            return False
            
        # Clean the data
        df_cleaned = self._clean_draft_data(df)
        
        # Add year column for reference
        df_cleaned.insert(0, 'Draft_Year', year)
        
        # Save the data
        self._save_draft_data(df_cleaned, year)
        
        print(f"Successfully scraped {len(df_cleaned)} draft picks for {year}")
        return True
        
    def scrape_draft_years(self, start_year: int = 2000, end_year: int = 2024):
        """Scrape draft data for a range of years"""
        print(f"Starting to scrape draft data from {start_year} to {end_year}")
        
        successful_years = []
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            try:
                success = self.scrape_draft_year(year)
                if success:
                    successful_years.append(year)
                else:
                    failed_years.append(year)
                    
                # Rate limit between requests (except for the last year)
                if year < end_year:
                    self._rate_limit()
                    
            except KeyboardInterrupt:
                print("\nScraping interrupted by user")
                break
            except Exception as e:
                print(f"Unexpected error for year {year}: {e}")
                failed_years.append(year)
                continue
                
        print(f"\n=== Draft Scraping Summary ===")
        print(f"Successfully scraped: {len(successful_years)} years")
        print(f"Failed: {len(failed_years)} years")
        
        if failed_years:
            print(f"Failed years: {failed_years}")
            
        return successful_years, failed_years


def main():
    """Main function to run the draft scraper"""
    parser = argparse.ArgumentParser(description='Scrape NFL draft data from Pro Football Reference')
    parser.add_argument('--year', type=int, help='Specific year to scrape (e.g., 2024)')
    parser.add_argument('--start-year', type=int, default=2000, help='Start year for range scraping (default: 2000)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for range scraping (default: 2024)')
    
    args = parser.parse_args()
    
    scraper = DraftDataScraper()
    
    if args.year:
        # Scrape specific year
        print(f"Scraping draft data for year {args.year}...")
        success = scraper.scrape_draft_year(args.year)
        if success:
            print(f"Successfully scraped draft data for {args.year}")
        else:
            print(f"Failed to scrape draft data for {args.year}")
            sys.exit(1)
    else:
        # Scrape range of years
        print(f"Scraping draft data for years {args.start_year}-{args.end_year}...")
        successful, failed = scraper.scrape_draft_years(args.start_year, args.end_year)
        
        if failed:
            print(f"\nRetrying failed years: {failed}")
            for year in failed:
                print(f"Retrying year {year}...")
                scraper.scrape_draft_year(year)
                scraper._rate_limit()


if __name__ == "__main__":
    main()