import pandas as pd
import sys
from pathlib import Path

# Spotrac to Pro Football Reference team abbreviation mappings
SPOTRAC_TO_PFR_MAPPINGS = {
    # AFC East
    "BUFBUF": "buf",  # Buffalo Bills
    "MIAMIA": "mia",  # Miami Dolphins  
    "NENE": "nwe",    # New England Patriots
    "NYJNYJ": "nyj",  # New York Jets
    
    # AFC North
    "BALBAL": "bal",  # Baltimore Ravens
    "CINCIN": "cin",  # Cincinnati Bengals
    "CLECLE": "cle",  # Cleveland Browns
    "PITPIT": "pit",  # Pittsburgh Steelers
    
    # AFC South
    "HOUHOU": "hou",  # Houston Texans
    "INDIND": "clt",  # Indianapolis Colts (PFR uses 'clt')
    "JAXJAX": "jax",  # Jacksonville Jaguars
    "TENTEN": "ten",  # Tennessee Titans
    
    # AFC West
    "DENDEN": "den",  # Denver Broncos
    "KCKC": "kan",    # Kansas City Chiefs (PFR uses 'kan')
    "LACLAC": "lac",  # Los Angeles Chargers
    "LVLV": "rai",    # Las Vegas Raiders (PFR uses 'rai')
    "OAKOAK": "rai",  # Oakland Raiders (historical, maps to 'rai')
    
    # NFC East
    "DALDAL": "dal",  # Dallas Cowboys
    "NYGNYG": "nyg",  # New York Giants
    "PHIPHI": "phi",  # Philadelphia Eagles
    "WASWAS": "was",  # Washington (PFR uses 'was')
    
    # NFC North
    "CHICHI": "chi",  # Chicago Bears
    "DETDET": "det",  # Detroit Lions
    "GBGB": "gnb",    # Green Bay Packers (PFR uses 'gnb')
    "MINMIN": "min",  # Minnesota Vikings
    
    # NFC South
    "ATLATL": "atl",  # Atlanta Falcons
    "CARCAR": "car",  # Carolina Panthers
    "NONO": "nor",    # New Orleans Saints (PFR uses 'nor')
    "TBTB": "tam",    # Tampa Bay Buccaneers (PFR uses 'tam')
    
    # NFC West
    "ARIARI": "crd",  # Arizona Cardinals (PFR uses 'crd')
    "LARLAR": "ram",  # Los Angeles Rams
    "SFSF": "sfo",    # San Francisco 49ers (PFR uses 'sfo')
    "SEASEA": "sea"   # Seattle Seahawks
}


class SpotracDataProcessor:
    """Processes Spotrac salary cap data and adds PFR team abbreviations"""
    
    def __init__(self, raw_dir: str = "../data/raw/Spotrac/total_view", 
                 output_dir: str = "../data/processed/Spotrac/total_view"):
        """Initialize processor with input and output directories"""
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_pfr_abbreviation(self, spotrac_team: str) -> str:
        """Convert Spotrac team abbreviation to PFR abbreviation"""
        return SPOTRAC_TO_PFR_MAPPINGS.get(spotrac_team, spotrac_team.lower())
    
    def _process_salary_cap_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single salary cap CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns exist
            if 'Team' not in df.columns:
                print(f"Warning: 'Team' column not found in {file_path.name}")
                return None
            
            # Add PFR team abbreviation column
            df['PFR_Team'] = df['Team'].apply(self._get_pfr_abbreviation)
            
            # Reorder columns to put PFR_Team right after Team
            cols = df.columns.tolist()
            team_idx = cols.index('Team')
            cols.insert(team_idx + 1, cols.pop(cols.index('PFR_Team')))
            df = df[cols]
            
            return df
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            return None
    
    def _extract_year_from_filename(self, filename: str) -> str:
        """Extract year from salary cap filename"""
        # Assumes format: salary_cap_YYYY.csv
        parts = filename.split('_')
        if len(parts) >= 3:
            year_part = parts[2].replace('.csv', '')
            return year_part
        return "unknown"
    
    def process_all_files(self) -> None:
        """Process all salary cap files in the raw directory"""
        
        # Find all salary cap CSV files
        salary_cap_files = list(self.raw_dir.glob("salary_cap_*.csv"))
        
        if not salary_cap_files:
            print("No salary cap files found in raw directory")
            return
        
        print(f"Found {len(salary_cap_files)} salary cap files to process")
        
        processed_count = 0
        failed_count = 0
        
        for file_path in sorted(salary_cap_files):
            year = self._extract_year_from_filename(file_path.name)
            print(f"Processing {file_path.name} (year: {year})...")
            
            # Process the file
            processed_df = self._process_salary_cap_file(file_path)
            
            if processed_df is not None:
                # Save processed file
                output_filename = f"salary_cap_{year}_processed.csv"
                output_path = self.output_dir / output_filename
                processed_df.to_csv(output_path, index=False)
                
                print(f"  * Saved to {output_filename}")
                processed_count += 1
            else:
                print(f"  x Failed to process {file_path.name}")
                failed_count += 1
        
        print(f"\nProcessing Summary:")
        print(f"Successfully processed: {processed_count} files")
        print(f"Failed to process: {failed_count} files")
    
    def process_single_year(self, year: int) -> bool:
        """Process salary cap data for a single year"""
        filename = f"salary_cap_{year}.csv"
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            print(f"File {filename} not found in raw directory")
            return False
        
        print(f"Processing {filename}...")
        
        # Process the file
        processed_df = self._process_salary_cap_file(file_path)
        
        if processed_df is not None:
            # Save processed file
            output_filename = f"salary_cap_{year}_processed.csv"
            output_path = self.output_dir / output_filename
            processed_df.to_csv(output_path, index=False)
            
            print(f"* Saved to {output_filename}")
            return True
        else:
            print(f"x Failed to process {filename}")
            return False
    
    def show_mapping_stats(self) -> None:
        """Show statistics about team abbreviation mappings"""
        print(f"\nTeam Abbreviation Mappings Available:")
        print(f"Total mappings: {len(SPOTRAC_TO_PFR_MAPPINGS)}")
        
        # Show a few examples
        print("\nExample mappings:")
        for i, (spotrac, pfr) in enumerate(list(SPOTRAC_TO_PFR_MAPPINGS.items())[:5]):
            print(f"  {spotrac} → {pfr}")
        print("  ...")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Spotrac salary cap data')
    parser.add_argument('--year', type=int, help='Process specific year only')
    parser.add_argument('--show-mappings', action='store_true', 
                       help='Show team abbreviation mapping statistics')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = SpotracDataProcessor()
    
    if args.show_mappings:
        processor.show_mapping_stats()
        return
    
    if args.year:
        # Process single year
        success = processor.process_single_year(args.year)
        if not success:
            sys.exit(1)
    else:
        # Process all files
        processor.process_all_files()
    
    print("\nSpotrac data processing completed!")


if __name__ == "__main__":
    main()