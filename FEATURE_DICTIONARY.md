# NFL Coaching WAR Feature Dictionary

This document provides a comprehensive index and definition of all 389 features used in the NFL coaching WAR analysis model.

## Feature Categories Overview

| Category | Count | Description |
|----------|-------|-------------|
| **Meta** | 3 | Basic identifiers |
| **Salary Cap** | 18 | Financial allocation and spending |
| **Roster Turnover** | 63 | Player retention and departure metrics (full roster) |
| **Starter Turnover** | 27 | Player retention and departure metrics (starters only) |
| **Injury/Availability** | 45 | Games missed and player availability |
| **Player Demographics** | 44 | Age and experience by position |
| **Player Performance** | 22 | Approximate Value (AV) metrics |
| **Penalty/Opponent** | 6 | Penalties and opponent metrics |
| **Draft Capital** | 29 | Historical draft picks by round |
| **Coaching Experience** | 7 | Coaching background and tenure |
| **Team Performance (OC)** | 33 | Offensive coordinator performance metrics |
| **Team Performance (DC)** | 33 | Defensive coordinator performance metrics |
| **Team Performance (HC)** | 33 | Head coach performance metrics |
| **Opponent Performance** | 33 | Performance against opponents |
| **Context** | 11 | Schedule strength and league context |

---

## Complete Feature Index

### Meta Features (1-3)
| # | Feature | Definition |
|---|---------|------------|
| 1 | Team | NFL team abbreviation (e.g., NWE, DAL) |
| 2 | Year | Season year (1970-2024) |
| 3 | Total_Games_In_Season | Number of games in season (16 for ≤2022, 17 for ≥2023) |

### Salary Cap Features (4-21)
| # | Feature | Definition |
|---|---------|------------|
| 4 | Total CapAllocations_Pct | Total salary cap allocated as percentage of maximum |
| 5 | Cap SpaceAll_Pct | Remaining salary cap space as percentage |
| 6 | Active53-Man_Pct | Cap allocated to 53-man roster as percentage |
| 7 | ReservesIR/PUP/NFI/SUSP_Pct | Cap allocated to reserve lists as percentage |
| 8 | DeadCap_Pct | Dead money as percentage of total cap |
| 9 | QB_Pct | Salary cap percentage allocated to quarterbacks |
| 10 | RB_Pct | Salary cap percentage allocated to running backs |
| 11 | WR_Pct | Salary cap percentage allocated to wide receivers |
| 12 | TE_Pct | Salary cap percentage allocated to tight ends |
| 13 | OL_Pct | Salary cap percentage allocated to offensive line |
| 14 | DL_Pct | Salary cap percentage allocated to defensive line |
| 15 | LB_Pct | Salary cap percentage allocated to linebackers |
| 16 | SEC_Pct | Salary cap percentage allocated to secondary (CB/S) |
| 17 | K_Pct | Salary cap percentage allocated to kickers |
| 18 | P_Pct | Salary cap percentage allocated to punters |
| 19 | LS_Pct | Salary cap percentage allocated to long snappers |
| 20 | Off_Pct | Salary cap percentage allocated to offensive players |
| 21 | Def_Pct | Salary cap percentage allocated to defensive players |
| 22 | SPT_Pct | Salary cap percentage allocated to special teams |

### Roster Turnover Features (23-94)
*Note: Each position has 8 roster-level turnover metrics (retained, departed, new, rates, net change)*

#### QB Turnover (23-28)
| # | Feature | Definition |
|---|---------|------------|
| 23 | QB_Players_Retained | Number of QB players retained on roster from previous season |
| 24 | QB_Players_Departed | Number of QB players who left roster from previous season |
| 25 | QB_Players_New | Number of new QB players added to roster |
| 26 | QB_Retention_Rate_Pct | Percentage of QB roster players retained |
| 27 | QB_Departure_Rate_Pct | Percentage of QB roster players departed |
| 28 | QB_New_Player_Rate_Pct | Percentage of roster that is new QB players |
| 29 | QB_Net_Change | Net change in QB roster players (new minus departed) |

#### RB Turnover (30-36)
| # | Feature | Definition |
|---|---------|------------|
| 30 | RB_Players_Retained | Number of RB players retained from previous season |
| 31 | RB_Players_Departed | Number of RB players who left from previous season |
| 32 | RB_Players_New | Number of new RB players added |
| 33 | RB_Retention_Rate_Pct | Percentage of RB players retained |
| 34 | RB_Departure_Rate_Pct | Percentage of RB players departed |
| 35 | RB_New_Player_Rate_Pct | Percentage of roster that is new RB players |
| 36 | RB_Net_Change | Net change in RB players (new minus departed) |

#### WR Turnover (37-43)
| # | Feature | Definition |
|---|---------|------------|
| 37 | WR_Players_Retained | Number of WR players retained from previous season |
| 38 | WR_Players_Departed | Number of WR players who left from previous season |
| 39 | WR_Players_New | Number of new WR players added |
| 40 | WR_Retention_Rate_Pct | Percentage of WR players retained |
| 41 | WR_Departure_Rate_Pct | Percentage of WR players departed |
| 42 | WR_New_Player_Rate_Pct | Percentage of roster that is new WR players |
| 43 | WR_Net_Change | Net change in WR players (new minus departed) |

#### TE Turnover (44-50)
| # | Feature | Definition |
|---|---------|------------|
| 44 | TE_Players_Retained | Number of TE players retained from previous season |
| 45 | TE_Players_Departed | Number of TE players who left from previous season |
| 46 | TE_Players_New | Number of new TE players added |
| 47 | TE_Retention_Rate_Pct | Percentage of TE players retained |
| 48 | TE_Departure_Rate_Pct | Percentage of TE players departed |
| 49 | TE_New_Player_Rate_Pct | Percentage of roster that is new TE players |
| 50 | TE_Net_Change | Net change in TE players (new minus departed) |

#### OL Turnover (51-57)
| # | Feature | Definition |
|---|---------|------------|
| 51 | OL_Players_Retained | Number of OL players retained from previous season |
| 52 | OL_Players_Departed | Number of OL players who left from previous season |
| 53 | OL_Players_New | Number of new OL players added |
| 54 | OL_Retention_Rate_Pct | Percentage of OL players retained |
| 55 | OL_Departure_Rate_Pct | Percentage of OL players departed |
| 56 | OL_New_Player_Rate_Pct | Percentage of roster that is new OL players |
| 57 | OL_Net_Change | Net change in OL players (new minus departed) |

#### DL Turnover (58-64)
| # | Feature | Definition |
|---|---------|------------|
| 58 | DL_Players_Retained | Number of DL players retained from previous season |
| 59 | DL_Players_Departed | Number of DL players who left from previous season |
| 60 | DL_Players_New | Number of new DL players added |
| 61 | DL_Retention_Rate_Pct | Percentage of DL players retained |
| 62 | DL_Departure_Rate_Pct | Percentage of DL players departed |
| 63 | DL_New_Player_Rate_Pct | Percentage of roster that is new DL players |
| 64 | DL_Net_Change | Net change in DL players (new minus departed) |

#### LB Turnover (65-71)
| # | Feature | Definition |
|---|---------|------------|
| 65 | LB_Players_Retained | Number of LB players retained from previous season |
| 66 | LB_Players_Departed | Number of LB players who left from previous season |
| 67 | LB_Players_New | Number of new LB players added |
| 68 | LB_Retention_Rate_Pct | Percentage of LB players retained |
| 69 | LB_Departure_Rate_Pct | Percentage of LB players departed |
| 70 | LB_New_Player_Rate_Pct | Percentage of roster that is new LB players |
| 71 | LB_Net_Change | Net change in LB players (new minus departed) |

#### CB Turnover (72-78)
| # | Feature | Definition |
|---|---------|------------|
| 72 | CB_Players_Retained | Number of CB players retained from previous season |
| 73 | CB_Players_Departed | Number of CB players who left from previous season |
| 74 | CB_Players_New | Number of new CB players added |
| 75 | CB_Retention_Rate_Pct | Percentage of CB players retained |
| 76 | CB_Departure_Rate_Pct | Percentage of CB players departed |
| 77 | CB_New_Player_Rate_Pct | Percentage of roster that is new CB players |
| 78 | CB_Net_Change | Net change in CB players (new minus departed) |

#### S Turnover (79-85)
| # | Feature | Definition |
|---|---------|------------|
| 79 | S_Players_Retained | Number of S players retained from previous season |
| 80 | S_Players_Departed | Number of S players who left from previous season |
| 81 | S_Players_New | Number of new S players added |
| 82 | S_Retention_Rate_Pct | Percentage of S players retained |
| 83 | S_Departure_Rate_Pct | Percentage of S players departed |
| 84 | S_New_Player_Rate_Pct | Percentage of roster that is new S players |
| 85 | S_Net_Change | Net change in S players (new minus departed) |

### Starter Turnover Features (86-112)
*Turnover rates specifically for starting players (not full roster)*

| # | Feature | Definition |
|---|---------|------------|
| 86 | QB_Retention_Rate_Pct_crosstab | Percentage of starting QB players retained from previous season |
| 87 | QB_Departure_Rate_Pct_crosstab | Percentage of starting QB players who departed from previous season |
| 88 | QB_New_Player_Rate_Pct_crosstab | Percentage of starting QB players who are new to the team |
| 89 | RB_Retention_Rate_Pct_crosstab | Percentage of starting RB players retained from previous season |
| 90 | RB_Departure_Rate_Pct_crosstab | Percentage of starting RB players who departed from previous season |
| 91 | RB_New_Player_Rate_Pct_crosstab | Percentage of starting RB players who are new to the team |
| 92 | WR_Retention_Rate_Pct_crosstab | Percentage of starting WR players retained from previous season |
| 93 | WR_Departure_Rate_Pct_crosstab | Percentage of starting WR players who departed from previous season |
| 94 | WR_New_Player_Rate_Pct_crosstab | Percentage of starting WR players who are new to the team |
| 95 | TE_Retention_Rate_Pct_crosstab | Percentage of starting TE players retained from previous season |
| 96 | TE_Departure_Rate_Pct_crosstab | Percentage of starting TE players who departed from previous season |
| 97 | TE_New_Player_Rate_Pct_crosstab | Percentage of starting TE players who are new to the team |
| 98 | OL_Retention_Rate_Pct_crosstab | Percentage of starting OL players retained from previous season |
| 99 | OL_Departure_Rate_Pct_crosstab | Percentage of starting OL players who departed from previous season |
| 100 | OL_New_Player_Rate_Pct_crosstab | Percentage of starting OL players who are new to the team |
| 101 | DL_Retention_Rate_Pct_crosstab | Percentage of starting DL players retained from previous season |
| 102 | DL_Departure_Rate_Pct_crosstab | Percentage of starting DL players who departed from previous season |
| 103 | DL_New_Player_Rate_Pct_crosstab | Percentage of starting DL players who are new to the team |
| 104 | LB_Retention_Rate_Pct_crosstab | Percentage of starting LB players retained from previous season |
| 105 | LB_Departure_Rate_Pct_crosstab | Percentage of starting LB players who departed from previous season |
| 106 | LB_New_Player_Rate_Pct_crosstab | Percentage of starting LB players who are new to the team |
| 107 | CB_Retention_Rate_Pct_crosstab | Percentage of starting CB players retained from previous season |
| 108 | CB_Departure_Rate_Pct_crosstab | Percentage of starting CB players who departed from previous season |
| 109 | CB_New_Player_Rate_Pct_crosstab | Percentage of starting CB players who are new to the team |
| 110 | S_Retention_Rate_Pct_crosstab | Percentage of starting S players retained from previous season |
| 111 | S_Departure_Rate_Pct_crosstab | Percentage of starting S players who departed from previous season |
| 112 | S_New_Player_Rate_Pct_crosstab | Percentage of starting S players who are new to the team |

### Injury/Availability Features (113-149)
*Games missed percentages and player counts by position*

| # | Feature | Definition |
|---|---------|------------|
| 113 | QB_Avg_Games_Missed_Pct | Average percentage of games missed by QB starters |
| 114 | QB_Max_Games_Missed_Pct | Maximum percentage of games missed by any QB starter |
| 115 | QB_Min_Games_Missed_Pct | Minimum percentage of games missed by QB starters |
| 116 | QB_Players_Count | Total number of QB players on roster |
| 117 | RB_Avg_Games_Missed_Pct | Average percentage of games missed by RB starters |
| 118 | RB_Max_Games_Missed_Pct | Maximum percentage of games missed by any RB starter |
| 119 | RB_Min_Games_Missed_Pct | Minimum percentage of games missed by RB starters |
| 120 | RB_Players_Count | Total number of RB players on roster |
| 121 | WR_Avg_Games_Missed_Pct | Average percentage of games missed by WR starters |
| 122 | WR_Max_Games_Missed_Pct | Maximum percentage of games missed by any WR starter |
| 123 | WR_Min_Games_Missed_Pct | Minimum percentage of games missed by WR starters |
| 124 | WR_Players_Count | Total number of WR players on roster |
| 125 | TE_Avg_Games_Missed_Pct | Average percentage of games missed by TE starters |
| 126 | TE_Max_Games_Missed_Pct | Maximum percentage of games missed by any TE starter |
| 127 | TE_Min_Games_Missed_Pct | Minimum percentage of games missed by TE starters |
| 128 | TE_Players_Count | Total number of TE players on roster |
| 129 | OL_Avg_Games_Missed_Pct | Average percentage of games missed by OL starters |
| 130 | OL_Max_Games_Missed_Pct | Maximum percentage of games missed by any OL starter |
| 131 | OL_Min_Games_Missed_Pct | Minimum percentage of games missed by OL starters |
| 132 | OL_Players_Count | Total number of OL players on roster |
| 133 | DL_Avg_Games_Missed_Pct | Average percentage of games missed by DL starters |
| 134 | DL_Max_Games_Missed_Pct | Maximum percentage of games missed by any DL starter |
| 135 | DL_Min_Games_Missed_Pct | Minimum percentage of games missed by DL starters |
| 136 | DL_Players_Count | Total number of DL players on roster |
| 137 | LB_Avg_Games_Missed_Pct | Average percentage of games missed by LB starters |
| 138 | LB_Max_Games_Missed_Pct | Maximum percentage of games missed by any LB starter |
| 139 | LB_Min_Games_Missed_Pct | Minimum percentage of games missed by LB starters |
| 140 | LB_Players_Count | Total number of LB players on roster |
| 141 | CB_Avg_Games_Missed_Pct | Average percentage of games missed by CB starters |
| 142 | CB_Max_Games_Missed_Pct | Maximum percentage of games missed by any CB starter |
| 143 | CB_Min_Games_Missed_Pct | Minimum percentage of games missed by CB starters |
| 144 | CB_Players_Count | Total number of CB players on roster |
| 145 | S_Avg_Games_Missed_Pct | Average percentage of games missed by S starters |
| 146 | S_Max_Games_Missed_Pct | Maximum percentage of games missed by any S starter |
| 147 | S_Min_Games_Missed_Pct | Minimum percentage of games missed by S starters |
| 148 | S_Players_Count | Total number of S players on roster |

### Player Demographics Features (149-193)
*Age and experience metrics for starters and roster*

#### Overall Demographics (149-156)
| # | Feature | Definition |
|---|---------|------------|
| 149 | Avg_Starter_Age | Average age of all starting players |
| 150 | StdDev_Starter_Age | Standard deviation of starter ages |
| 151 | Avg_Starter_Experience | Average years of NFL experience for starters |
| 152 | StdDev_Starter_Experience | Standard deviation of starter experience |
| 153 | Avg_Roster_Age | Average age of all roster players |
| 154 | StdDev_Roster_Age | Standard deviation of roster ages |
| 155 | Avg_Roster_Experience | Average years of NFL experience for roster |
| 156 | StdDev_Roster_Experience | Standard deviation of roster experience |

#### Position-Specific Demographics (157-192)
*Each position has 4 metrics: starter age, starter experience, roster age, roster experience*

| # | Feature | Definition |
|---|---------|------------|
| 157 | Avg_Starter_Age_QB | Average age of QB starters |
| 158 | Avg_Starter_Exp_QB | Average experience of QB starters |
| 159 | Avg_Roster_Age_QB | Average age of QB roster players |
| 160 | Avg_Roster_Exp_QB | Average experience of QB roster players |
| 161 | Avg_Starter_Age_RB | Average age of RB starters |
| 162 | Avg_Starter_Exp_RB | Average experience of RB starters |
| 163 | Avg_Roster_Age_RB | Average age of RB roster players |
| 164 | Avg_Roster_Exp_RB | Average experience of RB roster players |
| 165 | Avg_Starter_Age_WR | Average age of WR starters |
| 166 | Avg_Starter_Exp_WR | Average experience of WR starters |
| 167 | Avg_Roster_Age_WR | Average age of WR roster players |
| 168 | Avg_Roster_Exp_WR | Average experience of WR roster players |
| 169 | Avg_Starter_Age_TE | Average age of TE starters |
| 170 | Avg_Starter_Exp_TE | Average experience of TE starters |
| 171 | Avg_Roster_Age_TE | Average age of TE roster players |
| 172 | Avg_Roster_Exp_TE | Average experience of TE roster players |
| 173 | Avg_Starter_Age_OL | Average age of OL starters |
| 174 | Avg_Starter_Exp_OL | Average experience of OL starters |
| 175 | Avg_Roster_Age_OL | Average age of OL roster players |
| 176 | Avg_Roster_Exp_OL | Average experience of OL roster players |
| 177 | Avg_Starter_Age_DL | Average age of DL starters |
| 178 | Avg_Starter_Exp_DL | Average experience of DL starters |
| 179 | Avg_Roster_Age_DL | Average age of DL roster players |
| 180 | Avg_Roster_Exp_DL | Average experience of DL roster players |
| 181 | Avg_Starter_Age_LB | Average age of LB starters |
| 182 | Avg_Starter_Exp_LB | Average experience of LB starters |
| 183 | Avg_Roster_Age_LB | Average age of LB roster players |
| 184 | Avg_Roster_Exp_LB | Average experience of LB roster players |
| 185 | Avg_Starter_Age_CB | Average age of CB starters |
| 186 | Avg_Starter_Exp_CB | Average experience of CB starters |
| 187 | Avg_Roster_Age_CB | Average age of CB roster players |
| 188 | Avg_Roster_Exp_CB | Average experience of CB roster players |
| 189 | Avg_Starter_Age_S | Average age of S starters |
| 190 | Avg_Starter_Exp_S | Average experience of S starters |
| 191 | Avg_Roster_Age_S | Average age of S roster players |
| 192 | Avg_Roster_Exp_S | Average experience of S roster players |

### Player Performance Features (193-214)
*Approximate Value (AV) metrics measuring player contribution*

#### Overall AV Metrics (193-196)
| # | Feature | Definition |
|---|---------|------------|
| 193 | Avg_Starter_AV | Average Approximate Value of starting players |
| 194 | StdDev_Starter_AV | Standard deviation of starter AV scores |
| 195 | Avg_Roster_AV | Average Approximate Value of roster players |
| 196 | StdDev_Roster_AV | Standard deviation of roster AV scores |

#### Position-Specific AV (197-214)
*Each position has 2 metrics: starter AV and roster AV*

| # | Feature | Definition |
|---|---------|------------|
| 197 | Avg_Starter_AV_QB | Average AV of QB starters |
| 198 | Avg_Roster_AV_QB | Average AV of QB roster players |
| 199 | Avg_Starter_AV_RB | Average AV of RB starters |
| 200 | Avg_Roster_AV_RB | Average AV of RB roster players |
| 201 | Avg_Starter_AV_WR | Average AV of WR starters |
| 202 | Avg_Roster_AV_WR | Average AV of WR roster players |
| 203 | Avg_Starter_AV_TE | Average AV of TE starters |
| 204 | Avg_Roster_AV_TE | Average AV of TE roster players |
| 205 | Avg_Starter_AV_OL | Average AV of OL starters |
| 206 | Avg_Roster_AV_OL | Average AV of OL roster players |
| 207 | Avg_Starter_AV_DL | Average AV of DL starters |
| 208 | Avg_Roster_AV_DL | Average AV of DL roster players |
| 209 | Avg_Starter_AV_LB | Average AV of LB starters |
| 210 | Avg_Roster_AV_LB | Average AV of LB roster players |
| 211 | Avg_Starter_AV_CB | Average AV of CB starters |
| 212 | Avg_Roster_AV_CB | Average AV of CB roster players |
| 213 | Avg_Starter_AV_S | Average AV of S starters |
| 214 | Avg_Roster_AV_S | Average AV of S roster players |

### Penalty/Opponent Context Features (215-221)
| # | Feature | Definition |
|---|---------|------------|
| 215 | Team_Int_Passing_Norm | Team interceptions thrown (normalized) |
| 216 | Team_Pen_Norm | Team penalties committed (normalized) |
| 217 | Team_Yds_Penalties_Norm | Team penalty yards (normalized) |
| 218 | Opp_Int_Passing_Norm | Opponent interceptions thrown (normalized) |
| 219 | Opp_Pen_Norm | Opponent penalties committed (normalized) |
| 220 | Opp_Yds_Penalties_Norm | Opponent penalty yards (normalized) |
| 221 | SoS | Strength of Schedule rating |

### Draft Capital Features (222-250)
*Historical draft picks by round and timeframe*

#### Current Year Draft Picks (222-228)
| # | Feature | Definition |
|---|---------|------------|
| 222 | Current_Round_1_Picks | Number of 1st round picks in current draft |
| 223 | Current_Round_2_Picks | Number of 2nd round picks in current draft |
| 224 | Current_Round_3_Picks | Number of 3rd round picks in current draft |
| 225 | Current_Round_4_Picks | Number of 4th round picks in current draft |
| 226 | Current_Round_5_Picks | Number of 5th round picks in current draft |
| 227 | Current_Round_6_Picks | Number of 6th round picks in current draft |
| 228 | Current_Round_7Plus_Picks | Number of 7th+ round picks in current draft |

#### Previous Year Draft History (229-249)
*Rolling averages of draft picks from previous 1-4 years*

| # | Feature | Definition |
|---|---------|------------|
| 229 | Prev_1Yr_Round_1_Picks | 1st round picks from 1 year ago |
| 230 | Prev_1Yr_Round_2_Picks | 2nd round picks from 1 year ago |
| 231 | Prev_1Yr_Round_3_Picks | 3rd round picks from 1 year ago |
| 232 | Prev_1Yr_Round_4_Picks | 4th round picks from 1 year ago |
| 233 | Prev_1Yr_Round_5_Picks | 5th round picks from 1 year ago |
| 234 | Prev_1Yr_Round_6_Picks | 6th round picks from 1 year ago |
| 235 | Prev_1Yr_Round_7Plus_Picks | 7th+ round picks from 1 year ago |
| 236 | Prev_2Yr_Round_1_Picks | Average 1st round picks from 2 years ago |
| 237 | Prev_2Yr_Round_2_Picks | Average 2nd round picks from 2 years ago |
| 238 | Prev_2Yr_Round_3_Picks | Average 3rd round picks from 2 years ago |
| 239 | Prev_2Yr_Round_4_Picks | Average 4th round picks from 2 years ago |
| 240 | Prev_2Yr_Round_5_Picks | Average 5th round picks from 2 years ago |
| 241 | Prev_2Yr_Round_6_Picks | Average 6th round picks from 2 years ago |
| 242 | Prev_2Yr_Round_7Plus_Picks | Average 7th+ round picks from 2 years ago |
| 243 | Prev_3Yr_Round_1_Picks | Average 1st round picks from 3 years ago |
| 244 | Prev_3Yr_Round_2_Picks | Average 2nd round picks from 3 years ago |
| 245 | Prev_3Yr_Round_3_Picks | Average 3rd round picks from 3 years ago |
| 246 | Prev_3Yr_Round_4_Picks | Average 4th round picks from 3 years ago |
| 247 | Prev_3Yr_Round_5_Picks | Average 5th round picks from 3 years ago |
| 248 | Prev_3Yr_Round_6_Picks | Average 6th round picks from 3 years ago |
| 249 | Prev_3Yr_Round_7Plus_Picks | Average 7th+ round picks from 3 years ago |
| 250 | Prev_4Yr_Round_1_Picks | Average 1st round picks from 4 years ago |

### Coaching Experience Features (251-257)
| # | Feature | Definition |
|---|---------|------------|
| 251 | num_times_hc | Number of times served as head coach |
| 252 | num_yr_col_pos | Years of college coaching (position coach) |
| 253 | num_yr_col_coor | Years of college coaching (coordinator) |
| 254 | num_yr_col_hc | Years of college head coaching |
| 255 | num_yr_nfl_pos | Years of NFL coaching (position coach) |
| 256 | num_yr_nfl_coor | Years of NFL coaching (coordinator) |
| 257 | num_yr_nfl_hc | Years of NFL head coaching |

### Team Performance Features (258-389)
*All performance metrics are normalized (z-scores) for fair comparison across eras*
*Suffix meanings: __oc = Offensive Coordinator, __dc = Defensive Coordinator, __hc = Head Coach, __opp = Opponent performance*

#### Offensive Coordinator Performance (258-290)
| # | Feature | Definition |
|---|---------|------------|
| 258 | PF (Points For)__oc_Norm | Points scored under OC (normalized) |
| 259 | Yds__oc_Norm | Total yards gained under OC (normalized) |
| 260 | Y/P__oc_Norm | Yards per play under OC (normalized) |
| 261 | TO__oc_Norm | Turnovers committed under OC (normalized) |
| 262 | 1stD__oc_Norm | First downs gained under OC (normalized) |
| 263 | Cmp Passing__oc_Norm | Pass completions under OC (normalized) |
| 264 | Att Passing__oc_Norm | Pass attempts under OC (normalized) |
| 265 | Yds Passing__oc_Norm | Passing yards under OC (normalized) |
| 266 | TD Passing__oc_Norm | Passing touchdowns under OC (normalized) |
| 267 | Int Passing__oc_Norm | Interceptions thrown under OC (normalized) |
| 268 | NY/A Passing__oc_Norm | Net yards per pass attempt under OC (normalized) |
| 269 | 1stD Passing__oc_Norm | First downs via passing under OC (normalized) |
| 270 | Att Rushing__oc_Norm | Rushing attempts under OC (normalized) |
| 271 | Yds Rushing__oc_Norm | Rushing yards under OC (normalized) |
| 272 | TD Rushing__oc_Norm | Rushing touchdowns under OC (normalized) |
| 273 | Y/A Rushing__oc_Norm | Yards per rush attempt under OC (normalized) |
| 274 | 1stD Rushing__oc_Norm | First downs via rushing under OC (normalized) |
| 275 | Pen__oc_Norm | Penalties committed under OC (normalized) |
| 276 | Yds Penalties__oc_Norm | Penalty yards under OC (normalized) |
| 277 | 1stPy__oc_Norm | First downs via penalties under OC (normalized) |
| 278 | #Dr__oc_Norm | Number of drives under OC (normalized) |
| 279 | Sc%__oc_Norm | Scoring percentage per drive under OC (normalized) |
| 280 | TO%__oc_Norm | Turnover percentage per drive under OC (normalized) |
| 281 | Time Average Drive__oc_Norm | Average drive time under OC (normalized) |
| 282 | Plays Average Drive__oc_Norm | Average plays per drive under OC (normalized) |
| 283 | Yds Average Drive__oc_Norm | Average yards per drive under OC (normalized) |
| 284 | Pts Average Drive__oc_Norm | Average points per drive under OC (normalized) |
| 285 | 3DAtt__oc_Norm | Third down attempts under OC (normalized) |
| 286 | 3D%__oc_Norm | Third down conversion rate under OC (normalized) |
| 287 | 4DAtt__oc_Norm | Fourth down attempts under OC (normalized) |
| 288 | 4D%__oc_Norm | Fourth down conversion rate under OC (normalized) |
| 289 | RZAtt__oc_Norm | Red zone attempts under OC (normalized) |
| 290 | RZPct__oc_Norm | Red zone scoring percentage under OC (normalized) |

#### Defensive Coordinator Performance (291-323)
*Same metrics as OC but for defensive performance*

| # | Feature | Definition |
|---|---------|------------|
| 291 | PF (Points For)__dc_Norm | Points allowed under DC (normalized) |
| 292 | Yds__dc_Norm | Total yards allowed under DC (normalized) |
| 293 | Y/P__dc_Norm | Yards per play allowed under DC (normalized) |
| 294 | TO__dc_Norm | Turnovers forced under DC (normalized) |
| 295 | 1stD__dc_Norm | First downs allowed under DC (normalized) |
| 296-323 | [Similar pattern] | Passing, rushing, penalty, and drive metrics allowed under DC |

#### Head Coach Performance (324-356)
*Same offensive metrics but attributed to head coach*

| # | Feature | Definition |
|---|---------|------------|
| 324 | PF (Points For)__hc_Norm | Points scored under HC (normalized) |
| 325 | Yds__hc_Norm | Total yards gained under HC (normalized) |
| 326-356 | [Similar pattern] | Complete offensive performance metrics under HC |

#### Opponent Performance (357-389)
*Performance metrics of teams faced*

| # | Feature | Definition |
|---|---------|------------|
| 357 | PF (Points For)__opp__hc_Norm | Average points scored by opponents faced |
| 358 | Yds__opp__hc_Norm | Average yards gained by opponents faced |
| 359-389 | [Similar pattern] | Complete opponent offensive performance metrics |

---

## Data Sources and Methods

### Data Collection
- **Pro Football Reference**: Primary source for team, player, and coaching statistics
- **Spotrac**: Salary cap and financial data
- **Custom Processing**: Comprehensive data transformation and feature engineering

### Normalization
- All performance metrics use **z-score normalization** for fair comparison across different eras
- Formula: `(value - league_mean) / league_std_dev`
- Enables comparison between 1970s and 2020s performance

### Missing Value Handling
- **SVD Matrix Completion**: Advanced imputation technique preserving data relationships
- **0.24% missing values** eliminated (159,052 → 0)

### Coaching Attribution
- **Coaching features** marked with `_Norm` suffix when normalized
- **Replacement level** calculated as league averages for coaching-specific metrics
- **WAR calculation**: `(Impact × Games_per_Season)` where Impact = Actual - Replacement

---

## Usage Notes

### Feature Importance
Top 5 most important features according to XGBoost model:
1. **Avg_Starter_AV** (#193) - Overall starter talent level
2. **StdDev_Roster_AV** (#196) - Depth and talent distribution
3. **StdDev_Starter_AV** (#194) - Starter talent consistency
4. **TE_Players_New** (#46) - Tight end roster turnover
5. **RB_Min_Games_Missed_Pct** (#119) - Running back availability

### Model Performance
- **R² Score**: 0.8971
- **RMSE**: 0.061892
- **Features**: 389 total, 388 used (excluding Win_Pct target)
- **Sample Size**: 1,683 team-seasons (1970-2024)

### Coaching WAR Results
- **Average coaching impact**: +0.0029 per season (+0.05 WAR)
- **Top coach**: Chuck Pagano (+0.996 WAR per season)
- **Range**: -0.709 to +0.996 WAR per season

---

## Important Clarification: Roster vs. Starter Turnover

### Why Both Feature Sets Exist
The model includes **two distinct but complementary turnover measurement systems**:

#### **Roster Turnover (Features 23-85): Organizational Management**
- Measures turnover across **entire team roster** 
- Includes starters, backups, practice squad, reserves
- Captures organizational stability and depth management
- Shows coaching ability to maintain institutional knowledge

#### **Starter Turnover (Features 86-112): Core Performance Impact**  
- Measures turnover among **starting players only**
- Focuses on positions that directly impact game outcomes
- Captures coaching ability to retain key talent
- More directly correlated with on-field performance

### Strategic Coaching Analysis
- **High roster, low starter turnover**: Maintains core while developing depth
- **Low roster, high starter turnover**: Organizational stability but key position instability
- **Both high**: Complete team overhaul/rebuilding phase
- **Both low**: Maximum continuity at all levels

### Model Value
These **90 complementary features** (63 roster + 27 starter) provide the model with nuanced understanding of different coaching roster management philosophies and their effectiveness.