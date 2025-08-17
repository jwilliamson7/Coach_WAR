#!/usr/bin/env python3
"""
Run All Data Processing Scripts in Order

This script executes all data processing and calculation scripts in the correct order,
preparing all datasets for the final combination step.

Usage:
    python scripts/run_all_processing.py
    python scripts/run_all_processing.py --skip-existing
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

def run_command(command, description):
    """
    Run a command and track its execution
    
    Args:
        command: Command to run as list of strings
        description: Description of what the command does
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('-'*80)
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Print output if there is any
        if result.stdout:
            print(result.stdout)
        
        print(f"[COMPLETED]: {description}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[FAILED]: {description}")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"[FAILED]: Script not found")
        return False

def main():
    """Main function to run all processing scripts"""
    parser = argparse.ArgumentParser(
        description='Run all data processing scripts in order'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip scripts if output files already exist (where applicable)'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop execution if any script fails'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("COACHING WAR DATA PROCESSING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Stop on error: {args.stop_on_error}")
    
    # Define all processing scripts in order
    scripts = [
        # 1. Salary cap data processing (needs to be before positional percentages)
        {
            'command': ['python', 'scripts/process_spotrac_data.py'],
            'description': 'Process salary cap data with PFR team mappings'
        },
        
        # 2. Combine salary cap totals
        {
            'command': ['python', 'scripts/combine_salary_cap_totals.py'],
            'description': 'Combine salary cap total files'
        },
        
        # 3. Calculate positional spending percentages
        {
            'command': ['python', 'scripts/calculate_positional_percentages.py'],
            'description': 'Calculate position spending as percentage of total cap'
        },
        
        # 4. Combine positional percentages
        {
            'command': ['python', 'scripts/combine_positional_percentages.py'],
            'description': 'Combine yearly positional percentage files'
        },
        
        # 5. Calculate roster turnover crosstab (1970-2024)
        {
            'command': ['python', 'scripts/calculate_roster_turnover_crosstab.py', '--all-teams', '--year', 'all', '--minyear', '1970'],
            'description': 'Calculate roster turnover in crosstab format (1970-2024)'
        },
        
        # 6. Calculate starters turnover crosstab (1970-2024)
        {
            'command': ['python', 'scripts/calculate_starters_turnover_crosstab.py', '--all-teams', '--year', 'all', '--minyear', '1970'],
            'description': 'Calculate starters turnover in crosstab format (1970-2024)'
        },
        
        # 7. Calculate starters games missed (1970-2024)
        {
            'command': ['python', 'scripts/calculate_starters_games_missed_crosstab.py', '--all-teams', '--year', 'all', '--minyear', '1970', '--maxyear', '2024'],
            'description': 'Calculate games missed by starters (1970-2024)'
        },
        
        # 8. Calculate age and experience metrics (1970-2024)
        {
            'command': ['python', 'scripts/calculate_age_experience_metrics_crosstab.py', '--all-teams', '--year', 'all', '--minyear', '1970', '--maxyear', '2024'],
            'description': 'Calculate age and experience metrics by position (1970-2024)'
        },
        
        # 9. Calculate AV metrics (1970-2024)
        {
            'command': ['python', 'scripts/calculate_av_metrics_crosstab.py', '--all-teams', '--year', 'all', '--minyear', '1970', '--maxyear', '2024'],
            'description': 'Calculate Approximate Value metrics (1970-2024)'
        },
        
        # 10. Extract penalty and interception metrics
        {
            'command': ['python', 'scripts/extract_penalty_interception_metrics.py', '--start-year', '1970', '--end-year', '2024'],
            'description': 'Extract penalty and interception rates (1970-2024)'
        },
        
        # 11. Combine injury data (use combine not process)
        {
            'command': ['python', 'scripts/combine_injury_data.py'],
            'description': 'Combine injury data files (all available years)'
        },
        
        # 12. Transform draft data (process all years)
        {
            'command': ['python', 'scripts/transform_draft_data.py'],
            'description': 'Transform draft data for all available years'
        },
        
        # 13. Extract SoS and winning percentage
        {
            'command': ['python', 'scripts/extract_sos_winning_percentage.py', '--all-teams'],
            'description': 'Extract Strength of Schedule and winning percentage (all historical data)'
        },
        
        # 14. Create yearly coach performance data
        {
            'command': ['python', 'scripts/create_yearly_coach_performance_data.py'],
            'description': 'Generate yearly coaching performance metrics (1970-2024)'
        }
    ]
    
    # Track results
    successful = []
    failed = []
    skipped = []
    
    # Run each script
    for i, script_info in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] {script_info['description']}")
        
        # Check if we should skip (this would need custom logic per script)
        # For now, we'll run everything
        
        success = run_command(script_info['command'], script_info['description'])
        
        if success:
            successful.append(script_info['description'])
        else:
            failed.append(script_info['description'])
            if args.stop_on_error:
                print("\n[STOPPED] Stopping due to error (--stop-on-error flag set)")
                break
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING PIPELINE SUMMARY")
    print("="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {len(successful)}/{len(scripts)}")
    print(f"Failed: {len(failed)}/{len(scripts)}")
    
    if successful:
        print("\n[SUCCESSFUL] scripts:")
        for desc in successful:
            print(f"  - {desc}")
    
    if failed:
        print("\n[FAILED] scripts:")
        for desc in failed:
            print(f"  - {desc}")
    
    # Final step recommendation
    if len(failed) == 0:
        print("\n" + "="*80)
        print("[SUCCESS] All processing scripts completed successfully!")
        print("You can now run the final combination script:")
        print("  python scripts/combine_final_datasets.py")
        print("="*80)
        
        # Optionally run the combination script automatically
        response = input("\nDo you want to run the final combination script now? (y/n): ")
        if response.lower() == 'y':
            print("\nRunning final combination script...")
            success = run_command(
                ['python', 'scripts/combine_final_datasets.py'],
                'Combine all final datasets into master dataset'
            )
            if success:
                print("\n[SUCCESS] Master dataset created successfully!")
                print("Output: data/final/combined_final_dataset.csv")
            else:
                print("\n[FAILED] Failed to create master dataset")
                sys.exit(1)
    else:
        print("\n" + "="*80)
        print("[WARNING] Some scripts failed. Please fix errors before running combination script.")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()