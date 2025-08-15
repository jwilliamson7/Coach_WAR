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


class SalaryCapScraper:
    """Scrapes NFL salary cap data from spotrac.com"""
    
    BASE_URL = 'https://www.spotrac.com/nfl/cap/_/year'
    
    def __init__(self, output_dir: str = "../../data/raw/Spotrac/total_view"):
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
        
    def _fetch_page(self, year: int) -> Optional[BeautifulSoup]:
        """Fetch and parse the salary cap page for a given year"""
        url = f"{self.BASE_URL}/{year}"
        
        try:
            print(f"Fetching data for year {year}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for year {year}: {e}")
            return None
            
    def _extract_salary_cap_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """Extract the main salary cap table from the page"""
        try:
            # Look for the main table - Spotrac typically uses tables with specific classes
            table = soup.find('table', class_='table') or soup.find('table')
            
            if not table:
                print("No table found on page")
                return None
                
            # Extract headers
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            else:
                # Fallback: get headers from first row
                first_row = table.find('tr')
                headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
                
            # Extract data rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                table_rows = tbody.find_all('tr')
            else:
                table_rows = table.find_all('tr')[1:]  # Skip header row
                
            for row in table_rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    for cell in cells:
                        # Clean cell text
                        text = cell.get_text(strip=True)
                        # Remove any currency symbols and commas for numeric data
                        text = text.replace('$', '').replace(',', '')
                        row_data.append(text)
                    rows.append(row_data)
                    
            if not rows:
                print("No data rows found in table")
                return None
                
            # Create DataFrame
            # Ensure we have the right number of columns
            max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
            
            # Pad headers if needed
            while len(headers) < max_cols:
                headers.append(f'Column_{len(headers)}')
                
            # Pad rows if needed
            for row in rows:
                while len(row) < max_cols:
                    row.append('')
                    
            df = pd.DataFrame(rows, columns=headers[:max_cols])
            return df
            
        except Exception as e:
            print(f"Error extracting table: {e}")
            return None
            
    def _save_data(self, df: pd.DataFrame, year: int):
        """Save the scraped data to CSV"""
        filename = f"salary_cap_{year}.csv"
        filepath = self.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False)
            print(f"Saved data for {year} to {filepath}")
        except Exception as e:
            print(f"Error saving data for {year}: {e}")
            
    def scrape_year(self, year: int) -> bool:
        """Scrape salary cap data for a specific year"""
        print(f"\n=== Scraping salary cap data for {year} ===")
        
        # Check if file already exists
        filename = f"salary_cap_{year}.csv"
        filepath = self.output_dir / filename
        if filepath.exists():
            print(f"Data for {year} already exists. Skipping...")
            return True
            
        soup = self._fetch_page(year)
        if not soup:
            return False
            
        df = self._extract_salary_cap_table(soup)
        if df is None or df.empty:
            print(f"No data extracted for {year}")
            return False
            
        self._save_data(df, year)
        return True
        
    def scrape_years(self, start_year: int = 2011, end_year: int = 2025):
        """Scrape salary cap data for a range of years"""
        print(f"Starting to scrape salary cap data from {start_year} to {end_year}")
        
        successful_years = []
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            try:
                success = self.scrape_year(year)
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
                
        print(f"\n=== Scraping Summary ===")
        print(f"Successfully scraped: {len(successful_years)} years")
        print(f"Failed: {len(failed_years)} years")
        
        if failed_years:
            print(f"Failed years: {failed_years}")
            
        return successful_years, failed_years


def main():
    """Main function to run the scraper"""
    parser = argparse.ArgumentParser(description='Scrape NFL salary cap data from Spotrac')
    parser.add_argument('--year', type=int, help='Specific year to scrape (e.g., 2024)')
    parser.add_argument('--start-year', type=int, default=2011, help='Start year for range scraping (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2025, help='End year for range scraping (default: 2025)')
    
    args = parser.parse_args()
    
    scraper = SalaryCapScraper()
    
    if args.year:
        # Scrape specific year
        print(f"Scraping data for year {args.year}...")
        success = scraper.scrape_year(args.year)
        if success:
            print(f"Successfully scraped data for {args.year}")
        else:
            print(f"Failed to scrape data for {args.year}")
            sys.exit(1)
    else:
        # Scrape range of years
        print(f"Scraping data for years {args.start_year}-{args.end_year}...")
        successful, failed = scraper.scrape_years(args.start_year, args.end_year)
        
        if failed:
            print(f"\nRetrying failed years: {failed}")
            for year in failed:
                print(f"Retrying year {year}...")
                scraper.scrape_year(year)
                scraper._rate_limit()


if __name__ == "__main__":
    main()