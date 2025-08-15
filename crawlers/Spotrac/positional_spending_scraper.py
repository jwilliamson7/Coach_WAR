#!/usr/bin/env python3
"""
Spotrac Positional Spending Scraper

Scrapes NFL positional spending data from spotrac.com for specified years.
Includes rate limiting and command line interface for flexible execution.

Usage:
    python positional_spending_scraper.py --year 2024
    python positional_spending_scraper.py --start-year 2020 --end-year 2024
"""

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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PositionalSpendingScraper:
    """Scrapes NFL positional spending data from spotrac.com"""
    
    BASE_URL = 'https://www.spotrac.com/nfl/position/_/year'
    
    def __init__(self, output_dir: str = "data/raw/Spotrac/positional_spending"):
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
        logger.info(f"Rate limiting: waiting {delay} seconds...")
        time.sleep(delay)
        
    def _fetch_page(self, year: int) -> Optional[BeautifulSoup]:
        """Fetch and parse the positional spending page for a given year"""
        url = f"{self.BASE_URL}/{year}/table/full/type/cap_total"
        
        try:
            logger.info(f"Fetching positional spending data for year {year}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for year {year}: {e}")
            return None
            
    def _extract_positional_table(self, soup: BeautifulSoup, year: int = 0) -> Optional[pd.DataFrame]:
        """Extract the positional spending table from the page"""
        try:
            # Debug: Show what tables are available
            all_tables = soup.find_all('table')
            logger.info(f"Found {len(all_tables)} tables on page")
            
            # Look for the main table - try different selectors
            table = None
            
            # Try finding table by class or other attributes
            possible_selectors = [
                'table.datatable',
                'table.responsive-datatable',
                'table.table',
                'table.sortable',
                'table[class*="data"]',
                'table[class*="sort"]',
                'table',
                '.datatable',
                '[class*="table"]'
            ]
            
            for selector in possible_selectors:
                tables = soup.select(selector)
                if tables:
                    # Look for table with positional data (should have team names and positions)
                    for i, t in enumerate(tables):
                        text_content = t.get_text().lower()
                        
                        # Look for keywords that indicate this is the positional spending table
                        if any(keyword in text_content for keyword in [
                            'quarterback', 'running back', 'wide receiver', 'tight end',
                            'arizona', 'atlanta', 'baltimore', 'buffalo', 'carolina',
                            'total cap', 'cap hit', 'position', 'qb', 'rb', 'wr', 'te', 'ol'
                        ]):
                            table = t
                            logger.info(f"Found positional table using selector '{selector}'")
                            break
                    if table:
                        break
            
            # If still not found, try the largest table
            if not table and all_tables:
                logger.info("No table found with keywords, trying largest table...")
                table = max(all_tables, key=lambda t: len(t.get_text()))
                logger.info(f"Using largest table with {len(table.get_text())} characters")
            
            if not table:
                logger.warning("Could not find positional spending table")
                # Debug: show page structure
                logger.info(f"Page title: {soup.title.get_text() if soup.title else 'No title'}")
                logger.info(f"Page contains: {len(soup.get_text())} characters")
                return None
            
            # Extract headers - try multiple approaches
            headers = []
            
            # Method 1: Look for thead - check all rows in thead
            header_section = table.find('thead')
            if header_section:
                # Check all rows in thead, there might be multiple header rows
                header_rows = header_section.find_all('tr')
                logger.info(f"Found {len(header_rows)} header rows in thead")
                
                all_headers = []
                for i, row in enumerate(header_rows):
                    row_headers = []
                    header_cells = row.find_all(['th', 'td'])
                    for cell in header_cells:
                        text = cell.get_text(strip=True)
                        if text:
                            colspan = int(cell.get('colspan', 1))
                            for _ in range(colspan):
                                row_headers.append(text)
                    logger.info(f"Header row {i}: {row_headers}")
                    all_headers.extend(row_headers)
                
                # Use the longest header row or combine all unique headers
                if all_headers:
                    headers = all_headers
                    logger.info(f"Combined headers from thead: {headers}")
            
            # Method 2: Try first row if no thead headers found
            if not headers:
                first_row = table.find('tr')
                if first_row:
                    header_cells = first_row.find_all(['th', 'td'])
                    headers = []
                    for cell in header_cells:
                        text = cell.get_text(strip=True)
                        if text:
                            # Handle colspan
                            colspan = int(cell.get('colspan', 1))
                            for _ in range(colspan):
                                headers.append(text)
                    logger.info(f"Found headers from first row: {headers}")
            
            # Method 3: Look for all th elements in the table
            if not headers:
                th_elements = table.find_all('th')
                headers = []
                for th in th_elements:
                    text = th.get_text(strip=True)
                    if text and text.lower() not in ['', ' ']:
                        colspan = int(th.get('colspan', 1))
                        for _ in range(colspan):
                            headers.append(text)
                logger.info(f"Found headers from all th elements: {headers}")
            
            if not headers:
                logger.warning("Could not extract table headers")
                return None
            
            logger.info(f"Final headers ({len(headers)}): {headers}")  # Show all headers
            
            # Extract data rows
            rows = []
            tbody = table.find('tbody')
            if tbody:
                data_rows = tbody.find_all('tr')
            else:
                # Skip header row if no tbody
                data_rows = table.find_all('tr')[1:]
            
            for row in data_rows:
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = []
                    for cell in cells:
                        # Clean up cell text
                        text = cell.get_text(strip=True)
                        # Remove currency symbols and clean up
                        text = text.replace('$', '').replace(',', '').replace('%', '')
                        
                        # Handle colspan for data cells too
                        colspan = int(cell.get('colspan', 1))
                        for _ in range(colspan):
                            row_data.append(text)
                    
                    # Add rows even if column count doesn't match exactly (pad with empty values)
                    if row_data:
                        # Pad with empty strings if row has fewer columns than headers
                        while len(row_data) < len(headers):
                            row_data.append('')
                        # Truncate if row has more columns than headers
                        if len(row_data) > len(headers):
                            row_data = row_data[:len(headers)]
                        rows.append(row_data)
                        
                    logger.debug(f"Row has {len(row_data)} columns: {row_data[:3]}...")  # Show first 3 cells
            
            if not rows:
                logger.warning("No data rows found in table")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Clean up column names
            df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
            
            logger.info(f"Successfully extracted {len(df)} rows with {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting positional spending table: {e}")
            return None
    
    def _clean_and_save_data(self, df: pd.DataFrame, year: int):
        """Clean the data and save to CSV"""
        if df is None or df.empty:
            logger.warning(f"No data to save for {year}")
            return
        
        try:
            # Basic data cleaning
            # Convert numeric columns (remove $ and , symbols)
            for col in df.columns:
                if col not in ['Team', 'Rank']:  # Keep text columns as-is
                    df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Add metadata
            df['Year'] = year
            df['Scraped_Date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save to CSV
            filename = f"positional_spending_{year}.csv"
            filepath = self.output_dir / filename
            
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved positional spending data for {year} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data for {year}: {e}")
    
    def scrape_year(self, year: int) -> bool:
        """Scrape positional spending data for a specific year"""
        try:
            # Check if file already exists BEFORE any network calls
            filename = f"positional_spending_{year}.csv"
            filepath = self.output_dir / filename
            
            if filepath.exists():
                logger.info(f"Positional spending data for {year} already exists. Skipping...")
                return True
                
            soup = self._fetch_page(year)
            if not soup:
                return False
                
            df = self._extract_positional_table(soup, year)
            if df is None or df.empty:
                logger.warning(f"No positional spending data extracted for {year}")
                return False
                
            self._clean_and_save_data(df, year)
            return True
            
        except Exception as e:
            logger.error(f"Unexpected error scraping {year}: {e}")
            return False
        
    def scrape_years(self, start_year: int = 2011, end_year: int = 2024):
        """Scrape positional spending data for a range of years"""
        logger.info(f"Starting to scrape positional spending data from {start_year} to {end_year}")
        
        successful_years = []
        failed_years = []
        last_year_scraped = False
        
        for year in range(start_year, end_year + 1):
            try:
                logger.info(f"Processing year {year}...")
                
                # Check if file already exists before doing anything
                filename = f"positional_spending_{year}.csv"
                filepath = self.output_dir / filename
                
                if filepath.exists():
                    logger.info(f"Positional spending data for {year} already exists. Skipping...")
                    successful_years.append(year)
                    # Don't rate limit for existing files
                    last_year_scraped = False
                    continue
                
                # Only rate limit if the previous year actually did scraping
                if last_year_scraped and year > start_year:
                    self._rate_limit()
                
                success = self.scrape_year(year)
                
                if success:
                    successful_years.append(year)
                    logger.info(f"✓ Successfully scraped {year}")
                    last_year_scraped = True
                else:
                    failed_years.append(year)
                    logger.warning(f"✗ Failed to scrape {year}")
                    last_year_scraped = False
                    
            except KeyboardInterrupt:
                logger.info("\nScraping interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error for year {year}: {e}")
                failed_years.append(year)
                last_year_scraped = False
                continue
                
        logger.info(f"\n=== Scraping Summary ===")
        logger.info(f"Successfully scraped: {len(successful_years)} years")
        logger.info(f"Failed: {len(failed_years)} years")
        
        if failed_years:
            logger.warning(f"Failed years: {failed_years}")
            
        return successful_years, failed_years


def main():
    """Main function to run the scraper"""
    parser = argparse.ArgumentParser(description='Scrape NFL positional spending data from Spotrac')
    parser.add_argument('--year', type=int, help='Specific year to scrape (e.g., 2024)')
    parser.add_argument('--start-year', type=int, default=2011, help='Start year for range scraping (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for range scraping (default: 2024)')
    parser.add_argument('--output-dir', type=str, default='data/raw/Spotrac/positional_spending',
                       help='Output directory for positional spending data')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = PositionalSpendingScraper(output_dir=args.output_dir)
    
    if args.year:
        # Scrape specific year
        logger.info(f"Scraping positional spending data for {args.year}")
        success = scraper.scrape_year(args.year)
        
        if success:
            logger.info(f"Successfully scraped positional spending data for {args.year}")
        else:
            logger.error(f"Failed to scrape positional spending data for {args.year}")
            sys.exit(1)
    else:
        # Scrape range of years
        logger.info(f"Scraping positional spending data for years {args.start_year}-{args.end_year}")
        successful_years, failed_years = scraper.scrape_years(args.start_year, args.end_year)
        
        if failed_years:
            logger.warning(f"Some years failed to scrape: {failed_years}")
            sys.exit(1)
    
    logger.info("Positional spending scraping completed!")


if __name__ == "__main__":
    main()