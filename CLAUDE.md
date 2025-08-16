# Coaching WAR (Wins Above Replacement) Project

## Project Overview
This project aims to develop a comprehensive coaching WAR metric for NFL coaches, evaluating their impact on team performance relative to a replacement-level coach. The analysis incorporates coaching experience, team context, and performance outcomes to quantify coaching value.

## Repository Structure

```
Coach_WAR/
├── README.md
├── CLAUDE.md                   # This file - project context for Claude
├── data/
│   ├── raw/                    # Raw scraped data
│   │   ├── Coaches/           # Individual coach data (history, ranks, results)
│   │   ├── Teams/             # Team statistics and records
│   │   ├── Rosters/           # Player roster data by team/year (2010-2024)
│   │   ├── Starters/          # Starting lineup data by team/year (2010-2024)
│   │   ├── Injuries/          # Team injury data by season
│   │   └── Spotrac/           # Salary cap and spending data
│   │       ├── total_view/    # Total salary cap data
│   │       └── positional_spending/ # Spending by position
│   ├── processed/             # Cleaned and processed data
│   │   ├── League Data/       # Yearly league-wide statistics (1920-2024)
│   │   ├── Spotrac/           # Processed salary cap data
│   │   ├── Injury/            # Combined injury data
│   │   └── RosterTurnover/    # Roster turnover analysis
│   │       ├── detailed/      # Year-to-year turnover comparisons
│   │       └── summary/       # Position turnover averages
│   └── final/                 # Final datasets ready for analysis
├── crawlers/                  # Web scraping scripts
│   ├── PFR/                   # Pro Football Reference scrapers
│   │   ├── coach_scraping.py  # Main coach data scraper
│   │   ├── team_data_scraping.py # Team statistics scraper
│   │   ├── roster_scraping.py # Player roster scraper
│   │   ├── starters_scraping.py # Starting lineup scraper
│   │   └── injury_scraping.py # Team injury data scraper
│   ├── Spotrac/               # Spotrac salary data scrapers
│   │   ├── total_view_scraper.py # Total salary cap scraper
│   │   └── positional_spending_scraper.py # Position spending scraper
│   └── utils/                 # Shared utilities
│       └── data_constants.py  # Constants and mappings (with NFL season functions)
└── scripts/                   # Data processing scripts
    ├── transform_team_data.py # Team data transformation
    ├── process_spotrac_data.py # Salary cap data processing (with percentage conversion)
    ├── calculate_positional_percentages.py # Position spending percentages
    ├── combine_positional_percentages.py # Combine positional percentage files
    ├── process_injury_data.py # Injury data combination
    ├── calculate_roster_turnover.py # Roster turnover analysis
    ├── combine_roster_turnover.py # Combine turnover files
    ├── calculate_roster_turnover_crosstab.py # Roster turnover in crosstab format
    ├── calculate_starters_turnover_crosstab.py # Starters turnover in crosstab format
    ├── calculate_starters_games_missed_crosstab.py # Starters games missed analysis
    ├── extract_sos_winning_percentage.py # Extract SoS and winning percentage metrics
    └── combine_final_datasets.py # Combine all final datasets into single comprehensive table
```

## Key Components

### Data Sources
- **Pro Football Reference (PFR)**: Primary source for coach, team, roster, starters, and injury data
- **Spotrac**: Salary cap and positional spending data for financial analysis
- **Coach Data**: Historical records, rankings, and results for individual coaches
- **Team Data**: Yearly statistics for offensive/defensive performance
- **Roster Data**: Player rosters by team and year (2010-2024) for turnover analysis
- **Starters Data**: Starting lineup data by team and year (2010-2024) with games started
- **Injury Data**: Weekly injury statuses and games missed by team
- **Salary Data**: Team salary cap allocation and positional spending breakdowns

### Current Data Status
- **Coaches**: Extensive collection of coach data with 3 main files per coach:
  - `all_coaching_history.csv`: Career timeline and positions
  - `all_coaching_ranks.csv`: Team performance rankings during tenure
  - `all_coaching_results.csv`: Win-loss records and outcomes
- **Teams**: Team statistics organized by franchise
- **Rosters**: Player roster data for all 32 teams (2010-2024) organized by team directories
- **Starters**: Starting lineup data with position, games started, and statistics for each starter
- **Injuries**: Weekly injury status data with games missed by injury type
- **Salary Cap**: Total salary cap and positional spending data with PFR team mappings (converted to percentages)
- **Turnover Analysis**: Position-by-position roster and starters turnover rates between consecutive seasons
- **Games Missed Analysis**: Percentage of games missed by starters, aggregated by position

### Analysis Features
The project tracks 222 comprehensive features across multiple categories:

1. **Salary Cap Management (18 features)**:
   - Total salary cap allocations and percentages
   - Positional spending percentages (QB, RB, WR, TE, OL, DL, LB, SEC, K, P, LS, Off, Def, SPT)
   - Cap space utilization and dead cap analysis

2. **Roster Management (126 features)**:
   - Roster turnover rates by position (retention, departure, new player rates)
   - Starters turnover analysis with simplified position groups
   - Games missed percentages by position for starters
   - Player count and net change metrics

3. **Player Performance (66 features)**:
   - Age and experience metrics by position
   - Approximate Value (AV) metrics for player contributions
   - Performance consistency and depth analysis

4. **Team Performance (12 features)**:
   - Strength of Schedule (SoS) and winning percentage
   - Penalty rates and interception metrics
   - Performance context and efficiency measures

### Key Scripts

#### Data Collection Scripts

##### `crawlers/PFR/coach_scraping.py`
- **Purpose**: Scrapes comprehensive coach data from Pro Football Reference
- **Features**: Rate limiting, error handling, progress tracking
- **Output**: Individual coach directories with 3 CSV files each

##### `crawlers/PFR/roster_scraping.py`
- **Purpose**: Scrapes player roster data for turnover analysis
- **Features**: Handles team abbreviation corrections, comprehensive table detection
- **Output**: Team roster files by year (2010-2024)

##### `crawlers/PFR/starters_scraping.py`
- **Purpose**: Scrapes starting lineup data from Pro Football Reference
- **Features**: Rate limiting, position detection, games started tracking, removes section headers
- **Output**: Team starters files by year (2010-2024) with position, player, and statistics

##### `crawlers/PFR/injury_scraping.py`
- **Purpose**: Scrapes weekly injury status data
- **Features**: Parses injury table, calculates games missed by status type
- **Output**: Transposed injury data with team/year metrics

##### `crawlers/Spotrac/total_view_scraper.py` & `positional_spending_scraper.py`
- **Purpose**: Scrapes salary cap and positional spending data
- **Features**: Rate limiting, team mapping, data validation
- **Output**: Salary cap totals and position-specific spending breakdowns

#### Data Processing Scripts

##### `scripts/process_spotrac_data.py`
- **Purpose**: Processes salary cap data with correct PFR team mappings and percentage conversion
- **Features**: Team abbreviation correction, duplicate handling, salary cap percentage calculation
- **Output**: Standardized salary cap data with PFR team codes and percentages of maximum cap

##### `scripts/calculate_positional_percentages.py`
- **Purpose**: Calculates position spending as percentage of total cap
- **Features**: Unit conversion, data merging, percentage calculations, excludes Total_Pct from output
- **Output**: Position spending percentages by team/year

##### `scripts/combine_positional_percentages.py`
- **Purpose**: Combines yearly positional percentage files into consolidated dataset
- **Features**: Excludes salary cap percentage columns, filters unwanted metadata columns
- **Output**: Combined positional percentages dataset with only position spending percentages

##### `scripts/calculate_roster_turnover.py`
- **Purpose**: Analyzes roster turnover by position between consecutive years
- **Features**: Position grouping, retention/departure rate calculations
- **Output**: Detailed and summary turnover statistics

##### `scripts/combine_roster_turnover.py`
- **Purpose**: Combines individual team turnover files into consolidated datasets
- **Features**: Data cleaning, duplicate removal, metadata tracking
- **Output**: Combined turnover datasets ready for analysis

##### `scripts/calculate_roster_turnover_crosstab.py`
- **Purpose**: Analyzes roster turnover in crosstab format (team-year rows, position metrics columns)
- **Features**: Position grouping, percentage calculations, comprehensive turnover metrics
- **Output**: Crosstab format with retention/departure/new player rates by position

##### `scripts/calculate_starters_turnover_crosstab.py`
- **Purpose**: Analyzes starters turnover in crosstab format with combined position groups
- **Features**: O-line position combining, streamlined percentage-only metrics
- **Output**: Crosstab format with starter turnover percentages by position (QB, RB, WR, TE, OL, DL, LB, CB, S)

##### `scripts/calculate_starters_games_missed_crosstab.py`
- **Purpose**: Calculates percentage of games missed by starters using Games Started field
- **Features**: Uses get_games_in_season() function, position aggregation with avg/max/min metrics
- **Output**: Crosstab format with games missed percentages by position and player counts

##### `scripts/transform_team_data.py`
- **Purpose**: Transforms team data into league-wide yearly datasets
- **Features**: Data normalization, type conversion, z-score standardization
- **Output**: Yearly league datasets with raw and normalized versions

##### `scripts/extract_sos_winning_percentage.py`
- **Purpose**: Extracts Strength of Schedule and calculates winning percentage from team records
- **Features**: Comprehensive winning percentage calculation including ties, SoS extraction from all historical data
- **Output**: Team-year level SoS and Win_Pct metrics with extraction timestamps
- **Note**: Outputs only essential columns (Team, Year, SoS, Win_Pct, Extraction_Date) for streamlined analysis

##### `scripts/combine_final_datasets.py`
- **Purpose**: Combines all processed datasets into a single comprehensive coaching analysis table
- **Features**: Left join strategy using team-year keys, column conflict resolution, metadata generation
- **Input**: All CSV files in data/final/ (excluding metadata files)
- **Output**: Master dataset with 448 rows × 222 columns covering 2011-2024
- **Key Functions**: Team-year standardization, coverage reporting, automatic suffix handling for conflicting columns

#### Utility Scripts

##### `crawlers/utils/data_constants.py`
- **Purpose**: Central configuration and constants with corrected PFR team abbreviations
- **Contains**: Team mappings, feature definitions, exclusion criteria, salary cap maximums, NFL season functions
- **Key Updates**: Fixed Baltimore Ravens (rav), Houston Texans (htx), LA Chargers (sdg), Tennessee Titans (oti)
- **New Functions**: `get_games_in_season(year)` - returns 16 games for ≤2022, 17 games for ≥2023

### Processed League Data Structure

The **League Data** directory contains comprehensive yearly statistics from 1920-2024, with each year containing:

- **`league_team_data.csv`**: Raw team performance statistics for all NFL teams
- **`league_team_data_normalized.csv`**: Z-score normalized team statistics for fair comparison
- **`league_opponent_data.csv`**: Raw opponent statistics faced by each team
- **`league_opponent_data_normalized.csv`**: Z-score normalized opponent statistics

Each dataset includes extensive offensive and defensive metrics such as:
- Scoring and yardage statistics (PF, Yds, offensive plays, Y/P)
- Turnover metrics (TO, FL+, INT)
- Efficiency measures (3rd/4th down conversions, red zone performance)
- Drive statistics (average drive time, plays, yards, points)
- Penalty data and first down conversions

### Data Processing Pipeline

1. **Data Collection**: Web scrapers collect raw coach, team, roster, injury, and salary data
2. **Team Abbreviation Standardization**: Ensure consistent PFR abbreviations across all data sources
3. **Data Transformation**: Process raw data into standardized yearly datasets
4. **Data Cleaning**: Scripts process and standardize data formats, handle duplicates
5. **Feature Engineering**: Extract and calculate coaching performance metrics including:
   - Roster turnover rates by position
   - Starters turnover rates by position (with combined position groups)
   - Games missed percentages by starters and position
   - Injury impact metrics by team/season
   - Salary cap allocation efficiency (as percentages of maximum cap)
   - Position spending percentages
6. **Data Combination**: Merge individual team files into consolidated datasets
7. **Final Dataset Generation**: Combine all processed datasets into comprehensive master table
8. **Normalization**: Apply statistical normalization (z-scores) for fair comparison across eras
9. **Analysis**: Calculate WAR metrics and coaching effectiveness incorporating all data dimensions

### Team Franchise Mappings
The project handles historical team relocations and name changes through comprehensive mappings in `data_constants.py`, with corrected PFR abbreviations:
- **Baltimore Ravens**: `rav` (not `bal`)
- **Houston Texans**: `htx` (not `hou`)
- **Los Angeles Chargers**: `sdg` (not `lac`)
- **Tennessee Titans**: `oti` (not `ten`)

### Current Analysis Parameters
- **Master Dataset Coverage**: 2011-2024 (14 seasons, 448 team-year combinations)
- **Historical Data**: 1920-2024 (105 seasons of league data)
- **Roster Data**: 2010-2024 (15 seasons) with turnover calculated 2011-2024
- **Starters Data**: 2010-2024 (15 seasons) with turnover calculated 2011-2024
- **Injury Data**: 2010-2024 with weekly status tracking
- **Salary Data**: 2011-2024 with positional breakdowns (converted to percentages of max cap)
- **Complete Coverage**: 100% data coverage across all 448 team-year combinations
- **Total Features**: 222 comprehensive coaching performance metrics
- **Team Coverage**: All 32 current NFL teams with consistent PFR abbreviations
- **Season Length**: 16 games (≤2022), 17 games (≥2023)
- **Data Integrity**: Fixed historical team mappings (STL→RAM, SD→SDG) for complete coverage

## Development Notes

### Code Quality
- All scripts include proper error handling and logging
- Rate limiting implemented for web scraping to respect server resources
- Modular design with shared utilities for consistency
- Type hints and documentation for maintainability

### Data Integrity
- Comprehensive exclusion criteria for invalid coaching roles
- Special handling for fired coaches vs. active coaches
- Data validation and consistency checks throughout pipeline

### Extension Points
- Additional data sources can be easily integrated
- Feature set is extensible through configuration files
- Analysis parameters can be adjusted for different time periods
- New coaching metrics can be added to the framework

## Usage

The project is designed as a complete pipeline from data collection to analysis. Key entry points:

### Data Collection
1. **Coach Data**: `python crawlers/PFR/coach_scraping.py --team all --year all`
2. **Roster Data**: `python crawlers/PFR/roster_scraping.py --team all --year all`
3. **Starters Data**: `python crawlers/PFR/starters_scraping.py --all-teams --start-year 2010 --end-year 2024`
4. **Injury Data**: `python crawlers/PFR/injury_scraping.py --all-teams --year all`
5. **Salary Data**: `python crawlers/Spotrac/total_view_scraper.py --all-teams --all-years`

### Data Processing
1. **Process Salary Data**: `python scripts/process_spotrac_data.py`
2. **Calculate Position Percentages**: `python scripts/calculate_positional_percentages.py`
3. **Combine Position Percentages**: `python scripts/combine_positional_percentages.py`
4. **Analyze Roster Turnover**: `python scripts/calculate_roster_turnover.py --all-teams --year all`
5. **Combine Turnover Data**: `python scripts/combine_roster_turnover.py`
6. **Process Injury Data**: `python scripts/process_injury_data.py`

### Advanced Analysis (Crosstab Format)
1. **Roster Turnover Crosstab**: `python scripts/calculate_roster_turnover_crosstab.py --all-teams --year all --minyear 2010`
2. **Starters Turnover Crosstab**: `python scripts/calculate_starters_turnover_crosstab.py --all-teams --year all --minyear 2010`
3. **Starters Games Missed Crosstab**: `python scripts/calculate_starters_games_missed_crosstab.py --all-teams --year all`
4. **Extract SoS and Winning Percentage**: `python scripts/extract_sos_winning_percentage.py --all-teams`

### Final Dataset Generation
**Combine All Datasets**: `python scripts/combine_final_datasets.py`
- Combines all processed data into a single comprehensive dataset
- Uses salary cap data as the base table (448 team-year combinations: 2011-2024)
- Left joins all other datasets on team-year key
- Handles column conflicts with appropriate suffixes
- Generates metadata and coverage statistics

### Analysis
Use processed data in `data/final/` for comprehensive coaching WAR calculations incorporating:
- Team performance metrics
- Roster turnover rates by position (detailed and crosstab formats)
- Starters turnover rates by position (streamlined crosstab format)
- Games missed percentages by starters and position
- Injury impact analysis
- Salary cap allocation efficiency (as percentages of maximum cap)
- Position spending percentages

### Key Output Files in `data/final/`

#### Individual Component Files
- `salary_cap_totals_combined.csv` - Salary cap totals and percentages (448 rows, 100% coverage)
- `positional_percentages_combined.csv` - Position spending percentages (448 rows, 100% coverage)
- `roster_turnover_crosstab.csv` - Roster turnover analysis in crosstab format (448 rows, 100% coverage)
- `starters_turnover_crosstab.csv` - Starters turnover analysis in streamlined crosstab format (448 rows, 100% coverage)
- `starters_games_missed_crosstab.csv` - Games missed analysis by position (448 rows, 100% coverage)
- `age_experience_metrics_crosstab.csv` - Age and experience metrics by position (448 rows, 100% coverage)
- `av_metrics_crosstab.csv` - Approximate Value metrics (448 rows, 100% coverage)
- `penalty_interception_metrics.csv` - Penalty and interception rates (448 rows, 100% coverage)
- `sos_winning_percentage.csv` - Strength of Schedule and winning percentage (448 rows, 100% coverage)

#### Master Dataset
- **`combined_final_dataset.csv`** - **Complete comprehensive dataset combining all metrics**
  - **448 rows** (32 teams × 14 years: 2011-2024)
  - **222 columns** of coaching performance metrics
  - **100% data coverage** across all categories
  - **Team-year key structure** for easy analysis and modeling
  - **Metadata file**: `combined_final_dataset_metadata.csv` with coverage statistics

#### Historical Data
- League data files with normalized team and opponent statistics (1920-2024)

This structure supports both research and production use cases for comprehensive coaching performance analysis with multiple data dimensions including roster management, injury impact, and financial efficiency.