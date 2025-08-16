#!/usr/bin/env python3
"""
Debug Sean Payton's experience calculation
"""

import pandas as pd
import sys
from pathlib import Path

def classify_coaching_role(role: str) -> str:
    """Classify coaching role into main categories"""
    if not role or not isinstance(role, str):
        return "None"
    
    role = role.strip()
    
    # Exclude interim and temporary roles
    if any(keyword in role.upper() for keyword in ['INTERIM', 'TEMP', 'ACTING']):
        return "None"
    
    # Head Coach (only if it's actual head coach, not assistant head coach)
    if "Head Coach" in role and "Assistant" not in role and "Asst." not in role:
        return "HC"
    
    # Coordinators
    if "Coordinator" in role:
        if "Offensive" in role or "Off" in role:
            return "OC"
        elif "Defensive" in role or "Def" in role:
            return "DC"
        elif "Special" in role:
            return "STC"
    
    # Position coaches (including specific positions)
    position_keywords = [
        "Quarterbacks", "Running Backs", "Wide Receivers", "Tight Ends", "Offensive Line",
        "Defensive Line", "Linebackers", "Defensive Backs", "Secondary", "Cornerbacks", "Safeties",
        "Special Teams", "Kickers", "Punters", "Long Snappers",
        "Assistant", "Coach"
    ]
    
    if any(keyword in role for keyword in position_keywords):
        return "Position"
    
    return "None"

def debug_sean_payton():
    # Load Sean Payton's data directly
    coach_dir = Path("data/raw/Coaches/Sean Payton")
    history_file = coach_dir / "all_coaching_history.csv"
    ranks_file = coach_dir / "all_coaching_ranks.csv"
    
    history_df = pd.read_csv(history_file)
    ranks_df = pd.read_csv(ranks_file) if ranks_file.exists() else pd.DataFrame()
    
    print("Sean Payton's Coaching History:")
    print("=" * 50)
    print(history_df.to_string(index=False))
    
    print("\n\nExperience Calculation for key years:")
    print("=" * 50)
    
    # Test experience calculation for a few key years
    test_years = [2006, 2011, 2016, 2023]
    
    for year in test_years:
        print(f"\nYear {year} (calculating prior experience up to {year}):")
        print("-" * 40)
        
        # Filter to years before current year and 1970+
        prior_years = history_df[
            (history_df['Year'] < year) & 
            (history_df['Year'] >= 1970)
        ]
        
        if prior_years.empty:
            print("No prior experience found")
            continue
            
        print(f"Prior years: {prior_years['Year'].min()}-{prior_years['Year'].max()} ({len(prior_years)} years)")
        
        # Initialize counters
        counters = {
            'num_yr_col_pos': 0,
            'num_yr_col_coor': 0,
            'num_yr_col_hc': 0,
            'num_yr_nfl_pos': 0,
            'num_yr_nfl_coor': 0,
            'num_yr_nfl_hc': 0,
            'num_times_hc': 0
        }
        
        hc_years = []
        prev_franchise = None
        last_hc_year = None
        
        print("\nYear-by-year breakdown:")
        for _, row in prior_years.iterrows():
            row_year = row['Year']
            level = row.get('Level', '')
            role = row.get('Role', '')
            classified_role = classify_coaching_role(role)
            
            print(f"  {row_year}: {level:15} {role:40} -> {classified_role}")
            
            # Count experience by level and role
            if level == "College" or "College" in str(level):
                if classified_role == "Position":
                    counters["num_yr_col_pos"] += 1
                elif classified_role in ["OC", "DC", "STC"]:
                    counters["num_yr_col_coor"] += 1
                elif classified_role == "HC":
                    counters["num_yr_col_hc"] += 1
                    
            elif level == "NFL":
                if classified_role == "Position":
                    counters["num_yr_nfl_pos"] += 1
                elif classified_role in ["OC", "DC", "STC"]:
                    counters["num_yr_nfl_coor"] += 1
                elif classified_role == "HC":
                    counters["num_yr_nfl_hc"] += 1
                    hc_years.append(row_year)
                    
                    # Check if this is a new HC hire
                    is_new_hire = (len(hc_years) == 1 or  # First HC job
                                 (last_hc_year is not None and row_year - last_hc_year > 1))  # Gap
                    
                    if is_new_hire:
                        counters["num_times_hc"] += 1
                        
                    last_hc_year = row_year
        
        print(f"\nCalculated experience for {year}:")
        for key, value in counters.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    debug_sean_payton()