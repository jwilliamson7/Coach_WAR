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
│   │   └── Teams/             # Team statistics and records
│   ├── processed/             # Cleaned and processed data
│   └── final/                 # Final datasets ready for analysis
├── crawlers/                  # Web scraping scripts
│   ├── PFR/                   # Pro Football Reference scrapers
│   │   ├── coach_scraping.py  # Main coach data scraper
│   │   └── team_data_scraping.py # Team statistics scraper
│   └── utils/                 # Shared utilities
│       └── data_constants.py  # Constants and mappings
└── scripts/                   # Data processing scripts
    └── transform_team_data.py # Team data transformation
```

## Key Components

### Data Sources
- **Pro Football Reference (PFR)**: Primary source for coach and team data
- **Coach Data**: Historical records, rankings, and results for individual coaches
- **Team Data**: Yearly statistics for offensive/defensive performance

### Current Data Status
- **Coaches**: Extensive collection of coach data with 3 main files per coach:
  - `all_coaching_history.csv`: Career timeline and positions
  - `all_coaching_ranks.csv`: Team performance rankings during tenure
  - `all_coaching_results.csv`: Win-loss records and outcomes
- **Teams**: Team statistics organized by franchise

### Analysis Features
The project tracks 154+ features across multiple categories:

1. **Core Coaching Experience (8 features)**:
   - Age, number of times as head coach
   - Years of experience in college/NFL positions, coordinator roles, head coaching

2. **Team Performance Statistics (132 features)**:
   - Offensive/defensive statistics with role-specific suffixes
   - Points, yards, turnovers, efficiency metrics
   - Split by coordinator (OC/DC) and head coach roles

3. **Hiring Team Context (14 features)**:
   - Previous team performance metrics
   - Historical context for hiring decisions

### Key Scripts

#### `crawlers/PFR/coach_scraping.py`
- **Purpose**: Scrapes comprehensive coach data from Pro Football Reference
- **Features**: Rate limiting, error handling, progress tracking
- **Output**: Individual coach directories with 3 CSV files each
- **Key Classes**: `CoachDataScraper`

#### `crawlers/utils/data_constants.py`
- **Purpose**: Central configuration and constants
- **Contains**: Team mappings, feature definitions, exclusion criteria
- **Key Functions**: `get_all_feature_names()`, `get_feature_dict()`

#### `scripts/transform_team_data.py`
- **Purpose**: Transforms team data into league-wide yearly datasets
- **Features**: Data normalization, type conversion, z-score standardization
- **Output**: Yearly league datasets with raw and normalized versions

### Data Processing Pipeline

1. **Data Collection**: Web scrapers collect raw coach and team data
2. **Data Cleaning**: Scripts process and standardize data formats
3. **Feature Engineering**: Extract and calculate coaching performance metrics
4. **Normalization**: Apply statistical normalization for fair comparison
5. **Analysis**: Calculate WAR metrics and coaching effectiveness

### Team Franchise Mappings
The project handles historical team relocations and name changes through comprehensive mappings in `data_constants.py`, ensuring continuity across franchise moves.

### Current Analysis Parameters
- **Cutoff Year**: 2022
- **Current Year**: 2025
- **Expected Features**: 154 total features
- **Hiring Context**: 1-2 year lookback for team performance

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

1. **Data Collection**: Run scrapers in `crawlers/` directory
2. **Data Processing**: Execute transformation scripts in `scripts/`
3. **Analysis**: Use processed data in `data/final/` for WAR calculations

This structure supports both research and production use cases for coaching performance analysis.