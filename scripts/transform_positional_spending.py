#!/usr/bin/env python3
"""
Positional Spending Data Transformer

Transforms Spotrac positional spending data by:
1. Consolidating duplicate teams by summing their spending figures
2. Mapping Spotrac team abbreviations to Pro Football Reference abbreviations

Usage:
    python transform_positional_spending.py
    python transform_positional_spending.py --start-year 2020 --end-year 2024
"""

import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Spotrac to Pro Football Reference team abbreviation mappings
SPOTRAC_TO_PFR_MAPPINGS = {
    # AFC East
    "BUF": "buf",  # Buffalo Bills
    "MIA": "mia",  # Miami Dolphins  
    "NE": "nwe",   # New England Patriots
    "NYJ": "nyj",  # New York Jets
    
    # AFC North
    "BAL": "rav",  # Baltimore Ravens (PFR uses 'rav')
    "CIN": "cin",  # Cincinnati Bengals
    "CLE": "cle",  # Cleveland Browns
    "PIT": "pit",  # Pittsburgh Steelers
    
    # AFC South
    "HOU": "htx",  # Houston Texans (PFR uses 'htx')
    "IND": "clt",  # Indianapolis Colts (PFR uses 'clt')
    "JAX": "jax",  # Jacksonville Jaguars
    "TEN": "oti",  # Tennessee Titans (PFR uses 'oti')
    
    # AFC West
    "DEN": "den",  # Denver Broncos
    "KC": "kan",   # Kansas City Chiefs (PFR uses 'kan')
    "LAC": "sdg",  # Los Angeles Chargers (PFR uses 'sdg')
    "LV": "rai",   # Las Vegas Raiders (PFR uses 'rai')
    "OAK": "rai",  # Oakland Raiders (historical, maps to 'rai')
    
    # NFC East
    "DAL": "dal",  # Dallas Cowboys
    "NYG": "nyg",  # New York Giants
    "PHI": "phi",  # Philadelphia Eagles
    "WAS": "was",  # Washington (PFR uses 'was')
    
    # NFC North
    "CHI": "chi",  # Chicago Bears
    "DET": "det",  # Detroit Lions
    "GB": "gnb",   # Green Bay Packers (PFR uses 'gnb')
    "MIN": "min",  # Minnesota Vikings
    
    # NFC South
    "ATL": "atl",  # Atlanta Falcons
    "CAR": "car",  # Carolina Panthers
    "NO": "nor",   # New Orleans Saints (PFR uses 'nor')
    "TB": "tam",   # Tampa Bay Buccaneers (PFR uses 'tam')
    
    # NFC West
    "ARI": "crd",  # Arizona Cardinals (PFR uses 'crd')
    "LAR": "ram",  # Los Angeles Rams
    "SF": "sfo",   # San Francisco 49ers (PFR uses 'sfo')
    "SEA": "sea",  # Seattle Seahawks
    
    # Historical team names (before relocations)
    "STL": "ram",  # St. Louis Rams (2011-2015) → Los Angeles Rams
    "SD": "sdg"    # San Diego Chargers (2011-2016) → Los Angeles Chargers
}


class PositionalSpendingTransformer:
    """Transforms Spotrac positional spending data for analysis"""
    
    def __init__(self, input_dir: str = "data/raw/Spotrac/positional_spending", 
                 output_dir: str = "data/processed/Spotrac/positional_spending"):
        """Initialize transformer with input and output directories"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define spending columns (exclude metadata columns)
        self.spending_columns = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'SEC', 'K', 'P', 'LS', 'Total']
        
    def _clean_spending_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert spending values to numeric"""
        df_cleaned = df.copy()
        
        for col in self.spending_columns:
            if col in df_cleaned.columns:
                # Remove 'M' suffix and convert to float (millions)
                df_cleaned[col] = df_cleaned[col].astype(str).str.replace('M', '').str.replace('$', '')
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                
        return df_cleaned
    
    def _consolidate_duplicate_teams(self, df: pd.DataFrame) -> pd.DataFrame:
        """Consolidate duplicate teams by summing their spending figures"""
        if df.empty:
            return df
            
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['Team'], keep=False)]
        if duplicates.empty:
            logger.info("No duplicate teams found")
            return df
        
        logger.info(f"Found {len(duplicates)} rows with duplicate team names")
        duplicate_teams = duplicates['Team'].unique()
        logger.info(f"Duplicate teams: {duplicate_teams}")
        
        # Group by team and sum spending columns
        spending_cols_present = [col for col in self.spending_columns if col in df.columns]
        
        # Separate metadata columns that should be kept from first occurrence
        metadata_cols = [col for col in df.columns if col not in spending_cols_present + ['Team']]
        
        # Group by team and aggregate
        grouped = df.groupby('Team').agg({
            **{col: 'sum' for col in spending_cols_present},  # Sum spending columns
            **{col: 'first' for col in metadata_cols}         # Keep first occurrence of metadata
        }).reset_index()
        
        logger.info(f"Consolidated {len(df)} rows into {len(grouped)} rows")
        return grouped
    
    def _map_team_abbreviations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Spotrac team abbreviations to Pro Football Reference abbreviations"""
        if df.empty:
            return df
            
        df_mapped = df.copy()
        
        # Create mapping function
        def map_team_abbrev(team):
            # Handle various formats (sometimes with spaces or other characters)
            team_clean = str(team).strip().upper()
            
            if team_clean in SPOTRAC_TO_PFR_MAPPINGS:
                return SPOTRAC_TO_PFR_MAPPINGS[team_clean]
            else:
                logger.warning(f"No PFR mapping found for team: '{team_clean}'")
                return team_clean.lower()  # Return lowercase original if no mapping found
        
        # Apply mapping
        df_mapped['PFR_Team'] = df_mapped['Team'].apply(map_team_abbrev)
        
        # Log mapping results
        mappings_used = df_mapped[['Team', 'PFR_Team']].drop_duplicates()
        logger.info(f"Team mappings applied:")
        for _, row in mappings_used.iterrows():
            logger.info(f"  {row['Team']} -> {row['PFR_Team']}")
        
        return df_mapped
    
    def _add_calculated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated columns for Offense, Defense, and Special Teams"""
        if df.empty:
            return df
            
        df_calc = df.copy()
        
        # Calculate Offense (QB + RB + WR + TE + OL)
        offense_cols = ['QB', 'RB', 'WR', 'TE', 'OL']
        offense_cols_present = [col for col in offense_cols if col in df_calc.columns]
        if offense_cols_present:
            df_calc['Off'] = df_calc[offense_cols_present].sum(axis=1)
            logger.info(f"Added 'Off' column summing: {offense_cols_present}")
        
        # Calculate Defense (DL + LB + SEC)
        defense_cols = ['DL', 'LB', 'SEC']
        defense_cols_present = [col for col in defense_cols if col in df_calc.columns]
        if defense_cols_present:
            df_calc['Def'] = df_calc[defense_cols_present].sum(axis=1)
            logger.info(f"Added 'Def' column summing: {defense_cols_present}")
        
        # Calculate Special Teams (K + P + LS)
        st_cols = ['K', 'P', 'LS']
        st_cols_present = [col for col in st_cols if col in df_calc.columns]
        if st_cols_present:
            df_calc['SPT'] = df_calc[st_cols_present].sum(axis=1)
            logger.info(f"Added 'SPT' column summing: {st_cols_present}")
        
        return df_calc
    
    def transform_year(self, year: int) -> bool:
        """Transform positional spending data for a specific year"""
        try:
            # Load raw data
            input_file = self.input_dir / f"positional_spending_{year}.csv"
            if not input_file.exists():
                logger.warning(f"Input file not found: {input_file}")
                return False
            
            logger.info(f"Processing positional spending data for {year}")
            df = pd.read_csv(input_file)
            
            if df.empty:
                logger.warning(f"No data found in {input_file}")
                return False
            
            logger.info(f"Loaded {len(df)} rows for {year}")
            
            # Step 1: Clean spending values
            df = self._clean_spending_values(df)
            
            # Step 2: Consolidate duplicate teams
            df = self._consolidate_duplicate_teams(df)
            
            # Step 3: Map team abbreviations
            df = self._map_team_abbreviations(df)
            
            # Step 4: Add calculated columns
            df = self._add_calculated_columns(df)
            
            # Step 5: Reorder columns for better readability
            base_cols = ['Team', 'PFR_Team']
            spending_cols = ['QB', 'RB', 'WR', 'TE', 'OL', 'Off', 'DL', 'LB', 'SEC', 'Def', 'K', 'P', 'LS', 'SPT', 'Total']
            spending_cols_present = [col for col in spending_cols if col in df.columns]
            metadata_cols = [col for col in df.columns if col not in base_cols + spending_cols_present]
            
            column_order = base_cols + spending_cols_present + metadata_cols
            df = df[column_order]
            
            # Step 6: Save processed data
            output_file = self.output_dir / f"positional_spending_{year}_processed.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"Saved processed data to {output_file}")
            logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            return False
    
    def transform_years(self, start_year: int = 2011, end_year: int = 2024) -> Dict[int, bool]:
        """Transform positional spending data for a range of years"""
        logger.info(f"Transforming positional spending data for years {start_year}-{end_year}")
        
        results = {}
        successful_years = []
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing year {year}...")
            success = self.transform_year(year)
            results[year] = success
            
            if success:
                successful_years.append(year)
                logger.info(f"✓ Successfully processed {year}")
            else:
                failed_years.append(year)
                logger.warning(f"✗ Failed to process {year}")
        
        logger.info(f"\n=== Transformation Summary ===")
        logger.info(f"Successfully processed: {len(successful_years)} years")
        logger.info(f"Failed: {len(failed_years)} years")
        
        if failed_years:
            logger.warning(f"Failed years: {failed_years}")
        
        return results
    
    def show_sample_data(self, year: int = 2024, num_rows: int = 5):
        """Show sample of processed data for verification"""
        output_file = self.output_dir / f"positional_spending_{year}_processed.csv"
        
        if not output_file.exists():
            logger.warning(f"Processed file not found: {output_file}")
            return
        
        df = pd.read_csv(output_file)
        
        logger.info(f"\nSample data for {year} (first {num_rows} rows):")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")
        
        # Show sample rows
        sample = df.head(num_rows)
        for i, row in sample.iterrows():
            logger.info(f"Row {i}: {row['Team']} -> {row['PFR_Team']}, Total: {row.get('Total', 'N/A')}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Transform Spotrac positional spending data')
    parser.add_argument('--year', type=int, help='Specific year to transform (e.g., 2024)')
    parser.add_argument('--start-year', type=int, default=2011, help='Start year for range transformation (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for range transformation (default: 2024)')
    parser.add_argument('--input-dir', type=str, default='data/raw/Spotrac/positional_spending',
                       help='Input directory for raw positional spending data')
    parser.add_argument('--output-dir', type=str, default='data/processed/Spotrac/positional_spending',
                       help='Output directory for processed positional spending data')
    parser.add_argument('--sample', action='store_true', help='Show sample of processed data')
    parser.add_argument('--sample-year', type=int, default=2024, help='Year for sample data (default: 2024)')
    
    args = parser.parse_args()
    
    # Initialize transformer
    transformer = PositionalSpendingTransformer(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    if args.year:
        # Transform specific year
        logger.info(f"Transforming positional spending data for {args.year}")
        success = transformer.transform_year(args.year)
        
        if success:
            logger.info(f"Successfully transformed data for {args.year}")
        else:
            logger.error(f"Failed to transform data for {args.year}")
            sys.exit(1)
    else:
        # Transform range of years
        logger.info(f"Transforming positional spending data for years {args.start_year}-{args.end_year}")
        results = transformer.transform_years(args.start_year, args.end_year)
        
        failed_years = [year for year, success in results.items() if not success]
        if failed_years:
            logger.warning(f"Some years failed to transform: {failed_years}")
    
    # Show sample data if requested
    if args.sample:
        transformer.show_sample_data(args.sample_year)
    
    logger.info("Positional spending transformation completed!")


if __name__ == "__main__":
    main()