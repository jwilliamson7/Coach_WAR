#!/usr/bin/env python3
"""
Positional Spending Percentage Calculator

Calculates each position's percentage of effective cap space (total cap - dead cap)
by combining positional spending data with total salary cap data.

Usage:
    python calculate_positional_percentages.py
    python calculate_positional_percentages.py --start-year 2020 --end-year 2024
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


class PositionalPercentageCalculator:
    """Calculates positional spending percentages relative to effective cap space"""
    
    def __init__(self, 
                 positional_dir: str = "data/processed/Spotrac/positional_spending",
                 total_cap_dir: str = "data/processed/Spotrac/total_view",
                 output_dir: str = "data/processed/Spotrac/positional_percentages"):
        """Initialize calculator with input and output directories"""
        self.positional_dir = Path(positional_dir)
        self.total_cap_dir = Path(total_cap_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define position columns for percentage calculations
        self.position_columns = ['QB', 'RB', 'WR', 'TE', 'OL', 'DL', 'LB', 'SEC', 'K', 'P', 'LS']
        self.group_columns = ['Off', 'Def', 'SPT']  # Calculated group totals
        
    def _load_positional_data(self, year: int) -> Optional[pd.DataFrame]:
        """Load processed positional spending data for a year"""
        positional_file = self.positional_dir / f"positional_spending_{year}_processed.csv"
        
        if not positional_file.exists():
            logger.warning(f"Positional spending file not found: {positional_file}")
            return None
        
        try:
            df = pd.read_csv(positional_file)
            logger.info(f"Loaded positional data for {year}: {len(df)} teams")
            return df
        except Exception as e:
            logger.error(f"Error loading positional data for {year}: {e}")
            return None
    
    def _load_total_cap_data(self, year: int) -> Optional[pd.DataFrame]:
        """Load total salary cap data for a year"""
        # Try different possible filenames for total cap data
        possible_files = [
            f"salary_cap_{year}_processed.csv",
            f"salary_cap_{year}.csv", 
            f"total_cap_{year}_processed.csv",
            f"total_cap_{year}.csv"
        ]
        
        total_cap_file = None
        for filename in possible_files:
            filepath = self.total_cap_dir / filename
            if filepath.exists():
                total_cap_file = filepath
                break
        
        if not total_cap_file:
            logger.warning(f"Total cap file not found for {year}. Tried: {possible_files}")
            return None
        
        try:
            df = pd.read_csv(total_cap_file)
            logger.info(f"Loaded total cap data for {year}: {len(df)} teams from {total_cap_file.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading total cap data for {year}: {e}")
            return None
    
    def _merge_datasets(self, positional_df: pd.DataFrame, total_cap_df: pd.DataFrame, year: int) -> Optional[pd.DataFrame]:
        """Merge positional spending and total cap datasets"""
        try:
            # Check what team columns are available in total cap data
            team_columns = [col for col in total_cap_df.columns if 'team' in col.lower()]
            logger.info(f"Available team columns in total cap data: {team_columns}")
            
            # Use PFR_Team for merging since that's standardized
            if 'PFR_Team' in total_cap_df.columns:
                merge_key = 'PFR_Team'
            elif 'Team' in total_cap_df.columns:
                merge_key = 'Team'
                # If total cap uses original team names, we need to map them
                logger.warning(f"Using 'Team' column for merge - may need team name mapping")
            else:
                logger.error(f"No suitable team column found in total cap data")
                return None
            
            # Merge on the PFR team abbreviations
            merged_df = pd.merge(
                positional_df, 
                total_cap_df, 
                left_on='PFR_Team', 
                right_on=merge_key, 
                how='inner',
                suffixes=('_pos', '_cap')
            )
            
            logger.info(f"Merged datasets: {len(positional_df)} positional teams + {len(total_cap_df)} cap teams = {len(merged_df)} merged teams")
            
            if len(merged_df) < 30:  # Expect ~32 teams
                logger.warning(f"Low merge count ({len(merged_df)} teams) - check team name consistency")
                # Show unmatched teams for debugging
                pos_teams = set(positional_df['PFR_Team'])
                cap_teams = set(total_cap_df[merge_key])
                unmatched_pos = pos_teams - cap_teams
                unmatched_cap = cap_teams - pos_teams
                if unmatched_pos:
                    logger.warning(f"Positional teams not in cap data: {unmatched_pos}")
                if unmatched_cap:
                    logger.warning(f"Cap teams not in positional data: {unmatched_cap}")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets for {year}: {e}")
            return None
    
    def _prepare_total_cap(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare total cap allocation data for percentage calculations"""
        try:
            df = merged_df.copy()
            
            # Look for total cap columns
            total_cap_cols = [col for col in df.columns if 'total' in col.lower() and 'cap' in col.lower()]
            
            logger.info(f"Found total cap columns: {total_cap_cols}")
            
            # Try to identify the correct total cap column
            total_cap_col = None
            
            # Look for common column names
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['total cap', 'total_cap', 'cap total', 'cap_total', 'total capallocations']:
                    total_cap_col = col
                    break
            
            # If not found, use first match
            if not total_cap_col and total_cap_cols:
                total_cap_col = total_cap_cols[0]
            
            if not total_cap_col:
                logger.error("Could not find total cap column")
                logger.info(f"Available columns: {list(df.columns)}")
                return df
            
            # Convert to numeric, handling currency formatting
            df[total_cap_col] = pd.to_numeric(
                df[total_cap_col].astype(str).str.replace('$', '').str.replace('M', '').str.replace(',', ''), 
                errors='coerce'
            )
            
            # Use total cap allocations (which includes dead cap) for percentage calculations
            df['Total_Cap_For_Calc'] = df[total_cap_col]
            logger.info(f"Using {total_cap_col} for percentage calculations (includes dead cap)")
            
            logger.info(f"Total cap range: ${df['Total_Cap_For_Calc'].min():.1f}M - ${df['Total_Cap_For_Calc'].max():.1f}M")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing total cap data: {e}")
            return merged_df
    
    def _calculate_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage of total cap for each position"""
        try:
            result_df = df.copy()
            
            # Convert total cap from dollars to millions for consistent units
            total_cap_millions = df['Total_Cap_For_Calc'] / 1_000_000
            
            # Calculate percentages for individual positions (spending is already in millions)
            for pos in self.position_columns:
                if pos in df.columns:
                    pct_col = f"{pos}_Pct"
                    result_df[pct_col] = (df[pos] / total_cap_millions * 100).round(2)
            
            # Calculate percentages for position groups
            for group in self.group_columns:
                if group in df.columns:
                    pct_col = f"{group}_Pct"
                    result_df[pct_col] = (df[group] / total_cap_millions * 100).round(2)
            
            # Calculate total spending percentage (should be close to 100% since positional data includes dead cap)
            if 'Total' in df.columns:
                result_df['Total_Pct'] = (df['Total'] / total_cap_millions * 100).round(2)
            
            logger.info("Calculated position percentages relative to total cap allocations")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating percentages: {e}")
            return df
    
    def _organize_output_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Organize columns for better readability"""
        try:
            # Base team information
            base_cols = ['Team', 'PFR_Team']
            
            # Cap information  
            cap_cols = [col for col in df.columns if any(term in col.lower() for term in ['cap', 'total_cap_for_calc'])]
            
            # Position spending (raw values)
            pos_spending_cols = [col for col in df.columns if col in self.position_columns + self.group_columns + ['Total']]
            
            # Position percentages (exclude Total_Pct from output)
            pos_pct_cols = [col for col in df.columns if col.endswith('_Pct') and col != 'Total_Pct']
            
            # Metadata
            metadata_cols = [col for col in df.columns if col in ['Year', 'Scraped_Date']]
            
            # Other columns
            other_cols = [col for col in df.columns if col not in base_cols + cap_cols + pos_spending_cols + pos_pct_cols + metadata_cols]
            
            # Arrange column order
            column_order = base_cols + cap_cols + pos_spending_cols + pos_pct_cols + metadata_cols + other_cols
            
            # Only include columns that exist and exclude Total_Pct
            final_columns = [col for col in column_order if col in df.columns and col != 'Total_Pct']
            
            return df[final_columns]
            
        except Exception as e:
            logger.error(f"Error organizing columns: {e}")
            return df
    
    def calculate_year(self, year: int) -> bool:
        """Calculate positional percentages for a specific year"""
        try:
            logger.info(f"Processing positional percentages for {year}")
            
            # Load data
            positional_df = self._load_positional_data(year)
            if positional_df is None:
                return False
            
            total_cap_df = self._load_total_cap_data(year)
            if total_cap_df is None:
                return False
            
            # Merge datasets
            merged_df = self._merge_datasets(positional_df, total_cap_df, year)
            if merged_df is None:
                return False
            
            # Prepare total cap data
            df_with_cap = self._prepare_total_cap(merged_df)
            
            # Calculate percentages
            df_with_percentages = self._calculate_percentages(df_with_cap)
            
            # Organize output
            final_df = self._organize_output_columns(df_with_percentages)
            
            # Save results
            output_file = self.output_dir / f"positional_percentages_{year}.csv"
            final_df.to_csv(output_file, index=False)
            
            logger.info(f"Saved positional percentages to {output_file}")
            logger.info(f"Final dataset: {len(final_df)} teams, {len(final_df.columns)} columns")
            
            # Show sample statistics
            if 'QB_Pct' in final_df.columns:
                logger.info(f"QB percentage range: {final_df['QB_Pct'].min():.1f}% - {final_df['QB_Pct'].max():.1f}%")
            if 'Total_Pct' in df_with_percentages.columns:
                avg_total_pct = df_with_percentages['Total_Pct'].mean()
                logger.info(f"Average total cap utilization: {avg_total_pct:.1f}%")
            if 'Off_Pct' in final_df.columns and 'Def_Pct' in final_df.columns:
                avg_off_pct = final_df['Off_Pct'].mean()
                avg_def_pct = final_df['Def_Pct'].mean()
                logger.info(f"Average offense allocation: {avg_off_pct:.1f}%, defense: {avg_def_pct:.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            return False
    
    def calculate_years(self, start_year: int = 2011, end_year: int = 2024) -> Dict[int, bool]:
        """Calculate positional percentages for a range of years"""
        logger.info(f"Calculating positional percentages for years {start_year}-{end_year}")
        
        results = {}
        successful_years = []
        failed_years = []
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Processing year {year}...")
            success = self.calculate_year(year)
            results[year] = success
            
            if success:
                successful_years.append(year)
                logger.info(f"✓ Successfully processed {year}")
            else:
                failed_years.append(year)
                logger.warning(f"✗ Failed to process {year}")
        
        logger.info(f"\n=== Calculation Summary ===")
        logger.info(f"Successfully processed: {len(successful_years)} years")
        logger.info(f"Failed: {len(failed_years)} years")
        
        if failed_years:
            logger.warning(f"Failed years: {failed_years}")
        
        return results
    
    def show_sample_data(self, year: int = 2024, num_teams: int = 5):
        """Show sample of calculated percentages for verification"""
        output_file = self.output_dir / f"positional_percentages_{year}.csv"
        
        if not output_file.exists():
            logger.warning(f"Processed file not found: {output_file}")
            return
        
        df = pd.read_csv(output_file)
        
        logger.info(f"\nSample data for {year} (first {num_teams} teams):")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Shape: {df.shape}")
        
        # Show key columns for sample teams
        display_cols = ['PFR_Team', 'Total_Cap_For_Calc', 'QB_Pct', 'Off_Pct', 'Def_Pct', 'SPT_Pct', 'Total_Pct']
        display_cols = [col for col in display_cols if col in df.columns]
        
        sample = df[display_cols].head(num_teams)
        for i, row in sample.iterrows():
            team_info = row['PFR_Team'] if 'PFR_Team' in row else f"Team {i}"
            logger.info(f"Team {i}: {team_info}")
            for col in display_cols[1:]:  # Skip PFR_Team column
                if col in row:
                    if col == 'Total_Cap_For_Calc':
                        logger.info(f"  Total_Cap: ${row[col]/1_000_000:.1f}M")
                    else:
                        logger.info(f"  {col}: {row[col]}%")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate positional spending percentages relative to effective cap')
    parser.add_argument('--year', type=int, help='Specific year to process (e.g., 2024)')
    parser.add_argument('--start-year', type=int, default=2011, help='Start year for range processing (default: 2011)')
    parser.add_argument('--end-year', type=int, default=2024, help='End year for range processing (default: 2024)')
    parser.add_argument('--positional-dir', type=str, default='data/processed/Spotrac/positional_spending',
                       help='Input directory for positional spending data')
    parser.add_argument('--total-cap-dir', type=str, default='data/processed/Spotrac/total_view', 
                       help='Input directory for total salary cap data')
    parser.add_argument('--output-dir', type=str, default='data/processed/Spotrac/positional_percentages',
                       help='Output directory for percentage calculations')
    parser.add_argument('--sample', action='store_true', help='Show sample of calculated data')
    parser.add_argument('--sample-year', type=int, default=2024, help='Year for sample data (default: 2024)')
    
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = PositionalPercentageCalculator(
        positional_dir=args.positional_dir,
        total_cap_dir=args.total_cap_dir,
        output_dir=args.output_dir
    )
    
    if args.year:
        # Process specific year
        logger.info(f"Calculating positional percentages for {args.year}")
        success = calculator.calculate_year(args.year)
        
        if success:
            logger.info(f"Successfully calculated percentages for {args.year}")
        else:
            logger.error(f"Failed to calculate percentages for {args.year}")
            sys.exit(1)
    else:
        # Process range of years
        logger.info(f"Calculating positional percentages for years {args.start_year}-{args.end_year}")
        results = calculator.calculate_years(args.start_year, args.end_year)
        
        failed_years = [year for year, success in results.items() if not success]
        if failed_years:
            logger.warning(f"Some years failed to process: {failed_years}")
    
    # Show sample data if requested
    if args.sample:
        calculator.show_sample_data(args.sample_year)
    
    logger.info("Positional percentage calculation completed!")


if __name__ == "__main__":
    main()