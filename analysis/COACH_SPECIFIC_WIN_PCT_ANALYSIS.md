# Coach-Specific Win Percentage Analysis

## Investigation Summary

### Current Problem
The XGBoost coaching WAR analysis currently uses **team-level Win_Pct** from `team_record.csv` files, which represents the full season record. When there are mid-season coaching changes, multiple coaches share the same actual Win_Pct despite having different individual records.

**Example: ATL 1976**
- Team Record: 4-10 (0.286 Win_Pct)
- Marion Campbell: 1-4 in 5 games (0.200 Win_Pct)
- Pat Peppler: 3-6 in 9 games (0.333 Win_Pct)
- Both coaches currently evaluated against the same 0.286 team Win_Pct

### Data Availability

**Source**: `data/raw/Coaches/*/all_coaching_results.csv`

#### Coverage (1970-2024)
- **Total coach-season records**: 1,738
- **Unique coaches**: 308
- **Unique team-years**: 1,638
- **Year range**: 1970-2024
- **Team-years with mid-season coaching changes**: 99 (6.0%)
  - 98 team-years with 2 coaches
  - 1 team-year with 3 coaches
- **Individual coach records with partial seasons**: 257 (14.8% of records)

#### Data Structure
Each file contains:
- `Year`: Season year
- `Tm`: Team abbreviation (PFR format)
- `G`: Games coached
- `W`: Wins
- `L`: Losses
- `T`: Ties
- `W-L%`: Win percentage

#### Match Rate with Current Data
- **Matched records**: 1,662 / 1,700 (97.8%)
- **Unmatched records**: 38 (2.2%)

#### Unmatched Records Breakdown
1. **2025 season data** (~20 records): Future season, should be excluded
2. **Las Vegas Raiders** (LVR): Team mapping issue
   - Antonio Pierce 2024
   - Jon Gruden 2020-2021
   - Josh McDaniels 2022-2023
3. **Bill Arnsparger 1976 MIA**: Missing coaching results file

### Team Abbreviation Mappings Needed

The coaching results files use different abbreviations than the PFR standard:

| Coaching Results | PFR Standard | Team Name |
|-----------------|--------------|-----------|
| ARI | CRD | Arizona Cardinals |
| BAL | RAV | Baltimore Ravens |
| BOS | NWE | New England Patriots |
| HOU | HTX | Houston Texans |
| IND | CLT | Indianapolis Colts |
| LAC | SDG | Los Angeles/San Diego Chargers |
| LAR | RAM | Los Angeles/St. Louis Rams |
| **LVR** | **RAI** | **Las Vegas/Oakland Raiders** |
| OAK | RAI | Oakland Raiders |
| PHO | CRD | Phoenix Cardinals |
| STL | RAM | St. Louis Rams |
| TEN | OTI | Tennessee Titans/Oilers |
| Indianapolis Colts | CLT | (one malformed entry) |

**Note**: LVR was initially included in mappings but needs to be verified against actual data.

### Key Findings

#### ✅ Advantages of Coach-Specific Win_Pct
1. **Accurate attribution**: Each coach evaluated against their own record
2. **Mid-season changes**: Properly handles 99 team-years with coaching changes (6.0% of team-years)
3. **High coverage**: 97.8% match rate with current coaching data
4. **Already collected**: Data exists in raw coaching results files
5. **Verified accuracy**: Spot checks confirm W-L records match team totals

#### ⚠️ Issues to Address
1. **Team abbreviation mapping**: Need to standardize 13 team abbreviations
2. **2.2% missing data**: 38 records lack coaching results
   - Mostly 2025 season (should exclude)
   - Las Vegas Raiders mapping issue
   - 1 legitimate gap (Bill Arnsparger 1976)
3. **Case sensitivity**: Current data uses lowercase teams, results use uppercase
4. **Duplicate handling**: Need to ensure proper Coach-Team-Year uniqueness

### Recommendation

**YES - We should use coach-specific Win_Pct** for the following reasons:

1. **Accuracy**: Evaluating coaches against their actual records is fundamentally more correct than using team season records
2. **Data quality**: 97.8% coverage with fixable gaps
3. **Minimal effort**: Team mapping is straightforward, files already exist
4. **Better analysis**: Properly accounts for mid-season changes affecting 6% of team-years (99 cases)

### Implementation Plan

1. **Create extraction script**: Read all `all_coaching_results.csv` files
2. **Apply team mappings**: Standardize abbreviations to PFR format
3. **Calculate Coach_Win_Pct**: `(W + 0.5*T) / G` for each coach-season
4. **Merge with coaching performance**: Replace team-level Win_Pct
5. **Handle missing data**:
   - Exclude 2025 records
   - Fix LVR → RAI mapping
   - Investigate Bill Arnsparger 1976 gap
6. **Update pipeline**: Modify `combine_final_datasets.py` to use coach-specific data
7. **Validate**: Verify team totals match for seasons without mid-season changes

### Expected Impact

- **Current duplicates**: 46 duplicate team-years in XGBoost results (not all are mid-season changes)
- **After fix**: Each coach-season gets unique Win_Pct attribution
- **Analysis improvement**: More accurate WAR calculations for 6% of team-years (99 mid-season changes)
- **Dataset change**: Coaching_Impact_Analysis will have accurate coach-specific targets

### Files to Modify

1. **New script**: `scripts/extract_coach_specific_win_pct.py`
   - Read all coaching results files
   - Apply team mappings
   - Calculate coach-specific Win_Pct
   - Output to `data/processed/Coaching/coach_specific_win_pct.csv`

2. **Update**: `scripts/combine_final_datasets.py`
   - Use coach-specific Win_Pct instead of team-level
   - Merge on Coach-Team-Year instead of just Team-Year

3. **Update**: `analysis/xgboost_coaching_impact_analysis.py`
   - Verify input data has coach-specific Win_Pct
   - Update documentation

### Validation Checks

- [ ] Team totals match: Sum of coach records = team season record
- [ ] No duplicate Coach-Team-Year combinations
- [ ] All mid-season changes have separate records
- [ ] 97.8%+ coverage maintained
- [ ] Win_Pct calculation matches PFR formula: (W + 0.5*T) / G

## Conclusion

Using coach-specific Win_Pct from `all_coaching_results.csv` is **feasible and recommended**. The data exists, coverage is excellent (97.8%), and implementation is straightforward. This will improve the accuracy of coaching WAR calculations, especially for the 6% of team-years (99 cases) involving mid-season coaching changes.
