"""
Script to document all features in the coaching WAR dataset based on actual column names.
Outputs a CSV with index, feature name, category, and description.
"""

import pandas as pd
import os
import argparse

def generate_feature_descriptions(include_av=False):
    """Generate descriptions for all features in the dataset with categories based on actual data.
    
    Args:
        include_av (bool): Whether to include Approximate Value (AV) metrics in documentation
    """
    
    # Define features with (description, category) tuples based on actual column names
    features = {
        # Metadata columns (not features)
        'Team': ('Team abbreviation', 'Metadata'),
        'Year': ('Season year', 'Metadata'),
        
        # Salary Cap Features (19 features)
        'Total CapAllocations_Pct': ('Total cap allocations as percentage of maximum cap', 'Salary Cap'),
        'Cap SpaceAll_Pct': ('Cap space as percentage of maximum cap', 'Salary Cap'),
        'Active53-Man_Pct': ('Active 53-man roster cap as percentage of maximum', 'Salary Cap'),
        'ReservesIR/PUP/NFI/SUSP_Pct': ('Reserve/injured/suspended cap as percentage of maximum', 'Salary Cap'),
        'DeadCap_Pct': ('Dead cap as percentage of maximum cap', 'Salary Cap'),
        'QB_Pct': ('Quarterback spending as percentage of total cap', 'Salary Cap'),
        'RB_Pct': ('Running back spending as percentage of total cap', 'Salary Cap'),
        'WR_Pct': ('Wide receiver spending as percentage of total cap', 'Salary Cap'),
        'TE_Pct': ('Tight end spending as percentage of total cap', 'Salary Cap'),
        'OL_Pct': ('Offensive line spending as percentage of total cap', 'Salary Cap'),
        'DL_Pct': ('Defensive line spending as percentage of total cap', 'Salary Cap'),
        'LB_Pct': ('Linebacker spending as percentage of total cap', 'Salary Cap'),
        'SEC_Pct': ('Secondary (CB/S) spending as percentage of total cap', 'Salary Cap'),
        'K_Pct': ('Kicker spending as percentage of total cap', 'Salary Cap'),
        'P_Pct': ('Punter spending as percentage of total cap', 'Salary Cap'),
        'LS_Pct': ('Long snapper spending as percentage of total cap', 'Salary Cap'),
        'Off_Pct': ('Total offensive spending as percentage of total cap', 'Salary Cap'),
        'Def_Pct': ('Total defensive spending as percentage of total cap', 'Salary Cap'),
        'SPT_Pct': ('Special teams spending as percentage of total cap', 'Salary Cap'),
        
        # Roster Turnover Features - Detailed (63 features - 9 positions x 7 metrics each)
        'QB_Players_Retained': ('Number of QBs retained from previous year', 'Roster Turnover'),
        'QB_Players_Departed': ('Number of QBs departed from previous year', 'Roster Turnover'),
        'QB_Players_New': ('Number of new QBs added to roster', 'Roster Turnover'),
        'QB_Retention_Rate_Pct': ('Percentage of QBs retained from previous year', 'Roster Turnover'),
        'QB_Departure_Rate_Pct': ('Percentage of QBs departed from previous year', 'Roster Turnover'),
        'QB_New_Player_Rate_Pct': ('Percentage of new QBs on roster', 'Roster Turnover'),
        'QB_Net_Change': ('Net change in number of QBs', 'Roster Turnover'),
        
        'RB_Players_Retained': ('Number of RBs retained from previous year', 'Roster Turnover'),
        'RB_Players_Departed': ('Number of RBs departed from previous year', 'Roster Turnover'),
        'RB_Players_New': ('Number of new RBs added to roster', 'Roster Turnover'),
        'RB_Retention_Rate_Pct': ('Percentage of RBs retained from previous year', 'Roster Turnover'),
        'RB_Departure_Rate_Pct': ('Percentage of RBs departed from previous year', 'Roster Turnover'),
        'RB_New_Player_Rate_Pct': ('Percentage of new RBs on roster', 'Roster Turnover'),
        'RB_Net_Change': ('Net change in number of RBs', 'Roster Turnover'),
        
        'WR_Players_Retained': ('Number of WRs retained from previous year', 'Roster Turnover'),
        'WR_Players_Departed': ('Number of WRs departed from previous year', 'Roster Turnover'),
        'WR_Players_New': ('Number of new WRs added to roster', 'Roster Turnover'),
        'WR_Retention_Rate_Pct': ('Percentage of WRs retained from previous year', 'Roster Turnover'),
        'WR_Departure_Rate_Pct': ('Percentage of WRs departed from previous year', 'Roster Turnover'),
        'WR_New_Player_Rate_Pct': ('Percentage of new WRs on roster', 'Roster Turnover'),
        'WR_Net_Change': ('Net change in number of WRs', 'Roster Turnover'),
        
        'TE_Players_Retained': ('Number of TEs retained from previous year', 'Roster Turnover'),
        'TE_Players_Departed': ('Number of TEs departed from previous year', 'Roster Turnover'),
        'TE_Players_New': ('Number of new TEs added to roster', 'Roster Turnover'),
        'TE_Retention_Rate_Pct': ('Percentage of TEs retained from previous year', 'Roster Turnover'),
        'TE_Departure_Rate_Pct': ('Percentage of TEs departed from previous year', 'Roster Turnover'),
        'TE_New_Player_Rate_Pct': ('Percentage of new TEs on roster', 'Roster Turnover'),
        'TE_Net_Change': ('Net change in number of TEs', 'Roster Turnover'),
        
        'OL_Players_Retained': ('Number of offensive linemen retained', 'Roster Turnover'),
        'OL_Players_Departed': ('Number of offensive linemen departed', 'Roster Turnover'),
        'OL_Players_New': ('Number of new offensive linemen added', 'Roster Turnover'),
        'OL_Retention_Rate_Pct': ('Percentage of offensive linemen retained', 'Roster Turnover'),
        'OL_Departure_Rate_Pct': ('Percentage of offensive linemen departed', 'Roster Turnover'),
        'OL_New_Player_Rate_Pct': ('Percentage of new offensive linemen', 'Roster Turnover'),
        'OL_Net_Change': ('Net change in number of offensive linemen', 'Roster Turnover'),
        
        'DL_Players_Retained': ('Number of defensive linemen retained', 'Roster Turnover'),
        'DL_Players_Departed': ('Number of defensive linemen departed', 'Roster Turnover'),
        'DL_Players_New': ('Number of new defensive linemen added', 'Roster Turnover'),
        'DL_Retention_Rate_Pct': ('Percentage of defensive linemen retained', 'Roster Turnover'),
        'DL_Departure_Rate_Pct': ('Percentage of defensive linemen departed', 'Roster Turnover'),
        'DL_New_Player_Rate_Pct': ('Percentage of new defensive linemen', 'Roster Turnover'),
        'DL_Net_Change': ('Net change in number of defensive linemen', 'Roster Turnover'),
        
        'LB_Players_Retained': ('Number of linebackers retained', 'Roster Turnover'),
        'LB_Players_Departed': ('Number of linebackers departed', 'Roster Turnover'),
        'LB_Players_New': ('Number of new linebackers added', 'Roster Turnover'),
        'LB_Retention_Rate_Pct': ('Percentage of linebackers retained', 'Roster Turnover'),
        'LB_Departure_Rate_Pct': ('Percentage of linebackers departed', 'Roster Turnover'),
        'LB_New_Player_Rate_Pct': ('Percentage of new linebackers', 'Roster Turnover'),
        'LB_Net_Change': ('Net change in number of linebackers', 'Roster Turnover'),
        
        'CB_Players_Retained': ('Number of cornerbacks retained', 'Roster Turnover'),
        'CB_Players_Departed': ('Number of cornerbacks departed', 'Roster Turnover'),
        'CB_Players_New': ('Number of new cornerbacks added', 'Roster Turnover'),
        'CB_Retention_Rate_Pct': ('Percentage of cornerbacks retained', 'Roster Turnover'),
        'CB_Departure_Rate_Pct': ('Percentage of cornerbacks departed', 'Roster Turnover'),
        'CB_New_Player_Rate_Pct': ('Percentage of new cornerbacks', 'Roster Turnover'),
        'CB_Net_Change': ('Net change in number of cornerbacks', 'Roster Turnover'),
        
        'S_Players_Retained': ('Number of safeties retained', 'Roster Turnover'),
        'S_Players_Departed': ('Number of safeties departed', 'Roster Turnover'),
        'S_Players_New': ('Number of new safeties added', 'Roster Turnover'),
        'S_Retention_Rate_Pct': ('Percentage of safeties retained', 'Roster Turnover'),
        'S_Departure_Rate_Pct': ('Percentage of safeties departed', 'Roster Turnover'),
        'S_New_Player_Rate_Pct': ('Percentage of new safeties', 'Roster Turnover'),
        'S_Net_Change': ('Net change in number of safeties', 'Roster Turnover'),
        
        # Roster Turnover Features - Crosstab (27 features - 9 positions x 3 rates each)
        'QB_Retention_Rate_Pct_crosstab': ('QB retention rate from crosstab analysis', 'Roster Turnover'),
        'QB_Departure_Rate_Pct_crosstab': ('QB departure rate from crosstab analysis', 'Roster Turnover'),
        'QB_New_Player_Rate_Pct_crosstab': ('QB new player rate from crosstab analysis', 'Roster Turnover'),
        'RB_Retention_Rate_Pct_crosstab': ('RB retention rate from crosstab analysis', 'Roster Turnover'),
        'RB_Departure_Rate_Pct_crosstab': ('RB departure rate from crosstab analysis', 'Roster Turnover'),
        'RB_New_Player_Rate_Pct_crosstab': ('RB new player rate from crosstab analysis', 'Roster Turnover'),
        'WR_Retention_Rate_Pct_crosstab': ('WR retention rate from crosstab analysis', 'Roster Turnover'),
        'WR_Departure_Rate_Pct_crosstab': ('WR departure rate from crosstab analysis', 'Roster Turnover'),
        'WR_New_Player_Rate_Pct_crosstab': ('WR new player rate from crosstab analysis', 'Roster Turnover'),
        'TE_Retention_Rate_Pct_crosstab': ('TE retention rate from crosstab analysis', 'Roster Turnover'),
        'TE_Departure_Rate_Pct_crosstab': ('TE departure rate from crosstab analysis', 'Roster Turnover'),
        'TE_New_Player_Rate_Pct_crosstab': ('TE new player rate from crosstab analysis', 'Roster Turnover'),
        'OL_Retention_Rate_Pct_crosstab': ('OL retention rate from crosstab analysis', 'Roster Turnover'),
        'OL_Departure_Rate_Pct_crosstab': ('OL departure rate from crosstab analysis', 'Roster Turnover'),
        'OL_New_Player_Rate_Pct_crosstab': ('OL new player rate from crosstab analysis', 'Roster Turnover'),
        'DL_Retention_Rate_Pct_crosstab': ('DL retention rate from crosstab analysis', 'Roster Turnover'),
        'DL_Departure_Rate_Pct_crosstab': ('DL departure rate from crosstab analysis', 'Roster Turnover'),
        'DL_New_Player_Rate_Pct_crosstab': ('DL new player rate from crosstab analysis', 'Roster Turnover'),
        'LB_Retention_Rate_Pct_crosstab': ('LB retention rate from crosstab analysis', 'Roster Turnover'),
        'LB_Departure_Rate_Pct_crosstab': ('LB departure rate from crosstab analysis', 'Roster Turnover'),
        'LB_New_Player_Rate_Pct_crosstab': ('LB new player rate from crosstab analysis', 'Roster Turnover'),
        'CB_Retention_Rate_Pct_crosstab': ('CB retention rate from crosstab analysis', 'Roster Turnover'),
        'CB_Departure_Rate_Pct_crosstab': ('CB departure rate from crosstab analysis', 'Roster Turnover'),
        'CB_New_Player_Rate_Pct_crosstab': ('CB new player rate from crosstab analysis', 'Roster Turnover'),
        'S_Retention_Rate_Pct_crosstab': ('S retention rate from crosstab analysis', 'Roster Turnover'),
        'S_Departure_Rate_Pct_crosstab': ('S departure rate from crosstab analysis', 'Roster Turnover'),
        'S_New_Player_Rate_Pct_crosstab': ('S new player rate from crosstab analysis', 'Roster Turnover'),
        
        # Injury/Availability Features (37 features)
        'Total_Games_In_Season': ('Total games in NFL season (16 or 17)', 'Injury/Availability'),
        'QB_Avg_Games_Missed_Pct': ('Average percentage of games missed by QBs', 'Injury/Availability'),
        'QB_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a QB', 'Injury/Availability'),
        'QB_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a QB', 'Injury/Availability'),
        'QB_Players_Count': ('Number of QBs on roster', 'Injury/Availability'),
        'RB_Avg_Games_Missed_Pct': ('Average percentage of games missed by RBs', 'Injury/Availability'),
        'RB_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a RB', 'Injury/Availability'),
        'RB_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a RB', 'Injury/Availability'),
        'RB_Players_Count': ('Number of RBs on roster', 'Injury/Availability'),
        'WR_Avg_Games_Missed_Pct': ('Average percentage of games missed by WRs', 'Injury/Availability'),
        'WR_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a WR', 'Injury/Availability'),
        'WR_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a WR', 'Injury/Availability'),
        'WR_Players_Count': ('Number of WRs on roster', 'Injury/Availability'),
        'TE_Avg_Games_Missed_Pct': ('Average percentage of games missed by TEs', 'Injury/Availability'),
        'TE_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a TE', 'Injury/Availability'),
        'TE_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a TE', 'Injury/Availability'),
        'TE_Players_Count': ('Number of TEs on roster', 'Injury/Availability'),
        'OL_Avg_Games_Missed_Pct': ('Average percentage of games missed by O-line', 'Injury/Availability'),
        'OL_Max_Games_Missed_Pct': ('Maximum percentage of games missed by O-line', 'Injury/Availability'),
        'OL_Min_Games_Missed_Pct': ('Minimum percentage of games missed by O-line', 'Injury/Availability'),
        'OL_Players_Count': ('Number of offensive linemen on roster', 'Injury/Availability'),
        'DL_Avg_Games_Missed_Pct': ('Average percentage of games missed by D-line', 'Injury/Availability'),
        'DL_Max_Games_Missed_Pct': ('Maximum percentage of games missed by D-line', 'Injury/Availability'),
        'DL_Min_Games_Missed_Pct': ('Minimum percentage of games missed by D-line', 'Injury/Availability'),
        'DL_Players_Count': ('Number of defensive linemen on roster', 'Injury/Availability'),
        'LB_Avg_Games_Missed_Pct': ('Average percentage of games missed by LBs', 'Injury/Availability'),
        'LB_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a LB', 'Injury/Availability'),
        'LB_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a LB', 'Injury/Availability'),
        'LB_Players_Count': ('Number of linebackers on roster', 'Injury/Availability'),
        'CB_Avg_Games_Missed_Pct': ('Average percentage of games missed by CBs', 'Injury/Availability'),
        'CB_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a CB', 'Injury/Availability'),
        'CB_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a CB', 'Injury/Availability'),
        'CB_Players_Count': ('Number of cornerbacks on roster', 'Injury/Availability'),
        'S_Avg_Games_Missed_Pct': ('Average percentage of games missed by safeties', 'Injury/Availability'),
        'S_Max_Games_Missed_Pct': ('Maximum percentage of games missed by a safety', 'Injury/Availability'),
        'S_Min_Games_Missed_Pct': ('Minimum percentage of games missed by a safety', 'Injury/Availability'),
        'S_Players_Count': ('Number of safeties on roster', 'Injury/Availability'),
        
        # Player Demographics - Age & Experience (44 features)
        'Avg_Starter_Age': ('Average age of all starters', 'Player Demographics'),
        'StdDev_Starter_Age': ('Standard deviation of starter ages', 'Player Demographics'),
        'Avg_Starter_Experience': ('Average experience of all starters', 'Player Demographics'),
        'StdDev_Starter_Experience': ('Standard deviation of starter experience', 'Player Demographics'),
        'Avg_Roster_Age': ('Average age of entire roster', 'Player Demographics'),
        'StdDev_Roster_Age': ('Standard deviation of roster ages', 'Player Demographics'),
        'Avg_Roster_Experience': ('Average experience of entire roster', 'Player Demographics'),
        'StdDev_Roster_Experience': ('Standard deviation of roster experience', 'Player Demographics'),
        
        'Avg_Starter_Age_QB': ('Average age of starting QBs', 'Player Demographics'),
        'Avg_Starter_Exp_QB': ('Average experience of starting QBs', 'Player Demographics'),
        'Avg_Roster_Age_QB': ('Average age of all QBs on roster', 'Player Demographics'),
        'Avg_Roster_Exp_QB': ('Average experience of all QBs on roster', 'Player Demographics'),
        
        'Avg_Starter_Age_RB': ('Average age of starting RBs', 'Player Demographics'),
        'Avg_Starter_Exp_RB': ('Average experience of starting RBs', 'Player Demographics'),
        'Avg_Roster_Age_RB': ('Average age of all RBs on roster', 'Player Demographics'),
        'Avg_Roster_Exp_RB': ('Average experience of all RBs on roster', 'Player Demographics'),
        
        'Avg_Starter_Age_WR': ('Average age of starting WRs', 'Player Demographics'),
        'Avg_Starter_Exp_WR': ('Average experience of starting WRs', 'Player Demographics'),
        'Avg_Roster_Age_WR': ('Average age of all WRs on roster', 'Player Demographics'),
        'Avg_Roster_Exp_WR': ('Average experience of all WRs on roster', 'Player Demographics'),
        
        'Avg_Starter_Age_TE': ('Average age of starting TEs', 'Player Demographics'),
        'Avg_Starter_Exp_TE': ('Average experience of starting TEs', 'Player Demographics'),
        'Avg_Roster_Age_TE': ('Average age of all TEs on roster', 'Player Demographics'),
        'Avg_Roster_Exp_TE': ('Average experience of all TEs on roster', 'Player Demographics'),
        
        'Avg_Starter_Age_OL': ('Average age of starting offensive linemen', 'Player Demographics'),
        'Avg_Starter_Exp_OL': ('Average experience of starting offensive linemen', 'Player Demographics'),
        'Avg_Roster_Age_OL': ('Average age of all offensive linemen', 'Player Demographics'),
        'Avg_Roster_Exp_OL': ('Average experience of all offensive linemen', 'Player Demographics'),
        
        'Avg_Starter_Age_DL': ('Average age of starting defensive linemen', 'Player Demographics'),
        'Avg_Starter_Exp_DL': ('Average experience of starting defensive linemen', 'Player Demographics'),
        'Avg_Roster_Age_DL': ('Average age of all defensive linemen', 'Player Demographics'),
        'Avg_Roster_Exp_DL': ('Average experience of all defensive linemen', 'Player Demographics'),
        
        'Avg_Starter_Age_LB': ('Average age of starting linebackers', 'Player Demographics'),
        'Avg_Starter_Exp_LB': ('Average experience of starting linebackers', 'Player Demographics'),
        'Avg_Roster_Age_LB': ('Average age of all linebackers', 'Player Demographics'),
        'Avg_Roster_Exp_LB': ('Average experience of all linebackers', 'Player Demographics'),
        
        'Avg_Starter_Age_CB': ('Average age of starting cornerbacks', 'Player Demographics'),
        'Avg_Starter_Exp_CB': ('Average experience of starting cornerbacks', 'Player Demographics'),
        'Avg_Roster_Age_CB': ('Average age of all cornerbacks', 'Player Demographics'),
        'Avg_Roster_Exp_CB': ('Average experience of all cornerbacks', 'Player Demographics'),
        
        'Avg_Starter_Age_S': ('Average age of starting safeties', 'Player Demographics'),
        'Avg_Starter_Exp_S': ('Average experience of starting safeties', 'Player Demographics'),
        'Avg_Roster_Age_S': ('Average age of all safeties', 'Player Demographics'),
        'Avg_Roster_Exp_S': ('Average experience of all safeties', 'Player Demographics'),
        
    }
    
    # Add AV features only if requested
    if include_av:
        av_features = {
            # Player Demographics - Approximate Value (22 features)
            'Avg_Starter_AV': ('Average approximate value of all starters', 'Player Demographics'),
            'StdDev_Starter_AV': ('Standard deviation of starter approximate values', 'Player Demographics'),
            'Avg_Roster_AV': ('Average approximate value of entire roster', 'Player Demographics'),
            'StdDev_Roster_AV': ('Standard deviation of roster approximate values', 'Player Demographics'),
            
            'Avg_Starter_AV_QB': ('Average AV of starting QBs', 'Player Demographics'),
            'Avg_Roster_AV_QB': ('Average AV of all QBs on roster', 'Player Demographics'),
            'Avg_Starter_AV_RB': ('Average AV of starting RBs', 'Player Demographics'),
            'Avg_Roster_AV_RB': ('Average AV of all RBs on roster', 'Player Demographics'),
            'Avg_Starter_AV_WR': ('Average AV of starting WRs', 'Player Demographics'),
            'Avg_Roster_AV_WR': ('Average AV of all WRs on roster', 'Player Demographics'),
            'Avg_Starter_AV_TE': ('Average AV of starting TEs', 'Player Demographics'),
            'Avg_Roster_AV_TE': ('Average AV of all TEs on roster', 'Player Demographics'),
            'Avg_Starter_AV_OL': ('Average AV of starting offensive linemen', 'Player Demographics'),
            'Avg_Roster_AV_OL': ('Average AV of all offensive linemen', 'Player Demographics'),
            'Avg_Starter_AV_DL': ('Average AV of starting defensive linemen', 'Player Demographics'),
            'Avg_Roster_AV_DL': ('Average AV of all defensive linemen', 'Player Demographics'),
            'Avg_Starter_AV_LB': ('Average AV of starting linebackers', 'Player Demographics'),
            'Avg_Roster_AV_LB': ('Average AV of all linebackers', 'Player Demographics'),
            'Avg_Starter_AV_CB': ('Average AV of starting cornerbacks', 'Player Demographics'),
            'Avg_Roster_AV_CB': ('Average AV of all cornerbacks', 'Player Demographics'),
            'Avg_Starter_AV_S': ('Average AV of starting safeties', 'Player Demographics'),
            'Avg_Roster_AV_S': ('Average AV of all safeties', 'Player Demographics'),
        }
        features.update(av_features)
    
    # Continue with other feature categories
    penalty_features = {
        # Penalty/Opponent Features (7 features)
        'Team_Int_Passing_Norm': ('Normalized team interceptions thrown', 'Penalty/Opponent'),
        'Team_Pen_Norm': ('Normalized team penalties committed', 'Penalty/Opponent'),
        'Team_Yds_Penalties_Norm': ('Normalized team penalty yards', 'Penalty/Opponent'),
        'Opp_Int_Passing_Norm': ('Normalized opponent interceptions thrown', 'Penalty/Opponent'),
        'Opp_Pen_Norm': ('Normalized opponent penalties committed', 'Penalty/Opponent'),
        'Opp_Yds_Penalties_Norm': ('Normalized opponent penalty yards', 'Penalty/Opponent'),
        'SoS': ('Strength of Schedule - average win percentage of opponents', 'Penalty/Opponent'),
    }
    features.update(penalty_features)
    
    # Draft Capital Features (29 features)
    draft_features = {
        'Current_Round_1_Picks': ('Number of first round picks in current year', 'Draft Capital'),
        'Current_Round_2_Picks': ('Number of second round picks in current year', 'Draft Capital'),
        'Current_Round_3_Picks': ('Number of third round picks in current year', 'Draft Capital'),
        'Current_Round_4_Picks': ('Number of fourth round picks in current year', 'Draft Capital'),
        'Current_Round_5_Picks': ('Number of fifth round picks in current year', 'Draft Capital'),
        'Current_Round_6_Picks': ('Number of sixth round picks in current year', 'Draft Capital'),
        'Current_Round_7Plus_Picks': ('Number of seventh round or later picks in current year', 'Draft Capital'),
        
        'Prev_1Yr_Round_1_Picks': ('First round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_2_Picks': ('Second round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_3_Picks': ('Third round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_4_Picks': ('Fourth round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_5_Picks': ('Fifth round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_6_Picks': ('Sixth round picks 1 year ago', 'Draft Capital'),
        'Prev_1Yr_Round_7Plus_Picks': ('Seventh+ round picks 1 year ago', 'Draft Capital'),
        
        'Prev_2Yr_Round_1_Picks': ('Average first round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_2_Picks': ('Average second round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_3_Picks': ('Average third round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_4_Picks': ('Average fourth round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_5_Picks': ('Average fifth round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_6_Picks': ('Average sixth round picks over past 2 years', 'Draft Capital'),
        'Prev_2Yr_Round_7Plus_Picks': ('Average seventh+ round picks over past 2 years', 'Draft Capital'),
        
        'Prev_3Yr_Round_1_Picks': ('Average first round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_2_Picks': ('Average second round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_3_Picks': ('Average third round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_4_Picks': ('Average fourth round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_5_Picks': ('Average fifth round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_6_Picks': ('Average sixth round picks over past 3 years', 'Draft Capital'),
        'Prev_3Yr_Round_7Plus_Picks': ('Average seventh+ round picks over past 3 years', 'Draft Capital'),
        
        'Prev_4Yr_Round_1_Picks': ('Average first round picks over past 4 years', 'Draft Capital'),
    }
    features.update(draft_features)
    
    # Coaching Experience Features (7 features)
    coaching_features = {
        'num_times_hc': ('Number of times as head coach', 'Coaching Experience'),
        'num_yr_col_pos': ('Years coaching college position groups', 'Coaching Experience'),
        'num_yr_col_coor': ('Years as college coordinator', 'Coaching Experience'),
        'num_yr_col_hc': ('Years as college head coach', 'Coaching Experience'),
        'num_yr_nfl_pos': ('Years coaching NFL position groups', 'Coaching Experience'),
        'num_yr_nfl_coor': ('Years as NFL coordinator', 'Coaching Experience'),
        'num_yr_nfl_hc': ('Years as NFL head coach', 'Coaching Experience'),
    }
    features.update(coaching_features)
    
    # Add OC performance features
    oc_features = {
        'PF (Points For)__oc_Norm': ('Normalized points scored under OC', 'Prior Team Performance (OC)'),
        'Yds__oc_Norm': ('Normalized offensive yards under OC', 'Prior Team Performance (OC)'),
        'Y/P__oc_Norm': ('Normalized yards per play under OC', 'Prior Team Performance (OC)'),
        'TO__oc_Norm': ('Normalized turnovers under OC', 'Prior Team Performance (OC)'),
        '1stD__oc_Norm': ('Normalized first downs under OC', 'Prior Team Performance (OC)'),
        'Cmp Passing__oc_Norm': ('Normalized pass completions under OC', 'Prior Team Performance (OC)'),
        'Att Passing__oc_Norm': ('Normalized pass attempts under OC', 'Prior Team Performance (OC)'),
        'Yds Passing__oc_Norm': ('Normalized passing yards under OC', 'Prior Team Performance (OC)'),
        'TD Passing__oc_Norm': ('Normalized passing touchdowns under OC', 'Prior Team Performance (OC)'),
        'Int Passing__oc_Norm': ('Normalized interceptions thrown under OC', 'Prior Team Performance (OC)'),
        'NY/A Passing__oc_Norm': ('Normalized net yards per pass attempt under OC', 'Prior Team Performance (OC)'),
        '1stD Passing__oc_Norm': ('Normalized passing first downs under OC', 'Prior Team Performance (OC)'),
        'Att Rushing__oc_Norm': ('Normalized rush attempts under OC', 'Prior Team Performance (OC)'),
        'Yds Rushing__oc_Norm': ('Normalized rushing yards under OC', 'Prior Team Performance (OC)'),
        'TD Rushing__oc_Norm': ('Normalized rushing touchdowns under OC', 'Prior Team Performance (OC)'),
        'Y/A Rushing__oc_Norm': ('Normalized yards per rush under OC', 'Prior Team Performance (OC)'),
        '1stD Rushing__oc_Norm': ('Normalized rushing first downs under OC', 'Prior Team Performance (OC)'),
        'Pen__oc_Norm': ('Normalized penalties under OC', 'Prior Team Performance (OC)'),
        'Yds Penalties__oc_Norm': ('Normalized penalty yards under OC', 'Prior Team Performance (OC)'),
        '1stPy__oc_Norm': ('Normalized first downs by penalty under OC', 'Prior Team Performance (OC)'),
        '#Dr__oc_Norm': ('Normalized number of drives under OC', 'Prior Team Performance (OC)'),
        'Sc%__oc_Norm': ('Normalized scoring percentage under OC', 'Prior Team Performance (OC)'),
        'TO%__oc_Norm': ('Normalized turnover percentage under OC', 'Prior Team Performance (OC)'),
        'Time Average Drive__oc_Norm': ('Normalized average drive time under OC', 'Prior Team Performance (OC)'),
        'Plays Average Drive__oc_Norm': ('Normalized average plays per drive under OC', 'Prior Team Performance (OC)'),
        'Yds Average Drive__oc_Norm': ('Normalized average yards per drive under OC', 'Prior Team Performance (OC)'),
        'Pts Average Drive__oc_Norm': ('Normalized average points per drive under OC', 'Prior Team Performance (OC)'),
        '3DAtt__oc_Norm': ('Normalized third down attempts under OC', 'Prior Team Performance (OC)'),
        '3D%__oc_Norm': ('Normalized third down conversion rate under OC', 'Prior Team Performance (OC)'),
        '4DAtt__oc_Norm': ('Normalized fourth down attempts under OC', 'Prior Team Performance (OC)'),
        '4D%__oc_Norm': ('Normalized fourth down conversion rate under OC', 'Prior Team Performance (OC)'),
        'RZAtt__oc_Norm': ('Normalized red zone attempts under OC', 'Prior Team Performance (OC)'),
        'RZPct__oc_Norm': ('Normalized red zone scoring percentage under OC', 'Prior Team Performance (OC)'),
    }
    features.update(oc_features)
    
    # Add DC performance features
    dc_features = {
        'PF (Points For)__dc_Norm': ('Normalized points allowed under DC', 'Prior Team Performance (DC)'),
        'Yds__dc_Norm': ('Normalized yards allowed under DC', 'Prior Team Performance (DC)'),
        'Y/P__dc_Norm': ('Normalized yards per play allowed under DC', 'Prior Team Performance (DC)'),
        'TO__dc_Norm': ('Normalized takeaways under DC', 'Prior Team Performance (DC)'),
        '1stD__dc_Norm': ('Normalized first downs allowed under DC', 'Prior Team Performance (DC)'),
        'Cmp Passing__dc_Norm': ('Normalized pass completions allowed under DC', 'Prior Team Performance (DC)'),
        'Att Passing__dc_Norm': ('Normalized pass attempts faced under DC', 'Prior Team Performance (DC)'),
        'Yds Passing__dc_Norm': ('Normalized passing yards allowed under DC', 'Prior Team Performance (DC)'),
        'TD Passing__dc_Norm': ('Normalized passing touchdowns allowed under DC', 'Prior Team Performance (DC)'),
        'Int Passing__dc_Norm': ('Normalized interceptions by defense under DC', 'Prior Team Performance (DC)'),
        'NY/A Passing__dc_Norm': ('Normalized net yards per pass allowed under DC', 'Prior Team Performance (DC)'),
        '1stD Passing__dc_Norm': ('Normalized passing first downs allowed under DC', 'Prior Team Performance (DC)'),
        'Att Rushing__dc_Norm': ('Normalized rush attempts faced under DC', 'Prior Team Performance (DC)'),
        'Yds Rushing__dc_Norm': ('Normalized rushing yards allowed under DC', 'Prior Team Performance (DC)'),
        'TD Rushing__dc_Norm': ('Normalized rushing touchdowns allowed under DC', 'Prior Team Performance (DC)'),
        'Y/A Rushing__dc_Norm': ('Normalized yards per rush allowed under DC', 'Prior Team Performance (DC)'),
        '1stD Rushing__dc_Norm': ('Normalized rushing first downs allowed under DC', 'Prior Team Performance (DC)'),
        'Pen__dc_Norm': ('Normalized opponent penalties under DC', 'Prior Team Performance (DC)'),
        'Yds Penalties__dc_Norm': ('Normalized opponent penalty yards under DC', 'Prior Team Performance (DC)'),
        '1stPy__dc_Norm': ('Normalized opponent first downs by penalty under DC', 'Prior Team Performance (DC)'),
        '#Dr__dc_Norm': ('Normalized opponent drives under DC', 'Prior Team Performance (DC)'),
        'Sc%__dc_Norm': ('Normalized opponent scoring percentage under DC', 'Prior Team Performance (DC)'),
        'TO%__dc_Norm': ('Normalized opponent turnover percentage under DC', 'Prior Team Performance (DC)'),
        'Time Average Drive__dc_Norm': ('Normalized opponent average drive time under DC', 'Prior Team Performance (DC)'),
        'Plays Average Drive__dc_Norm': ('Normalized opponent average plays per drive under DC', 'Prior Team Performance (DC)'),
        'Yds Average Drive__dc_Norm': ('Normalized opponent average yards per drive under DC', 'Prior Team Performance (DC)'),
        'Pts Average Drive__dc_Norm': ('Normalized opponent average points per drive under DC', 'Prior Team Performance (DC)'),
        '3DAtt__dc_Norm': ('Normalized opponent third down attempts under DC', 'Prior Team Performance (DC)'),
        '3D%__dc_Norm': ('Normalized opponent third down conversion rate under DC', 'Prior Team Performance (DC)'),
        '4DAtt__dc_Norm': ('Normalized opponent fourth down attempts under DC', 'Prior Team Performance (DC)'),
        '4D%__dc_Norm': ('Normalized opponent fourth down conversion rate under DC', 'Prior Team Performance (DC)'),
        'RZAtt__dc_Norm': ('Normalized opponent red zone attempts under DC', 'Prior Team Performance (DC)'),
        'RZPct__dc_Norm': ('Normalized opponent red zone scoring percentage under DC', 'Prior Team Performance (DC)'),
    }
    features.update(dc_features)
    
    # Add HC team performance features
    hc_team_features = {
        'PF (Points For)__hc_Norm': ('Normalized points scored under HC', 'Prior Team Performance (HC)'),
        'Yds__hc_Norm': ('Normalized offensive yards under HC', 'Prior Team Performance (HC)'),
        'Y/P__hc_Norm': ('Normalized yards per play under HC', 'Prior Team Performance (HC)'),
        'TO__hc_Norm': ('Normalized turnovers under HC', 'Prior Team Performance (HC)'),
        '1stD__hc_Norm': ('Normalized first downs under HC', 'Prior Team Performance (HC)'),
        'Cmp Passing__hc_Norm': ('Normalized pass completions under HC', 'Prior Team Performance (HC)'),
        'Att Passing__hc_Norm': ('Normalized pass attempts under HC', 'Prior Team Performance (HC)'),
        'Yds Passing__hc_Norm': ('Normalized passing yards under HC', 'Prior Team Performance (HC)'),
        'TD Passing__hc_Norm': ('Normalized passing touchdowns under HC', 'Prior Team Performance (HC)'),
        'Int Passing__hc_Norm': ('Normalized interceptions thrown under HC', 'Prior Team Performance (HC)'),
        'NY/A Passing__hc_Norm': ('Normalized net yards per pass attempt under HC', 'Prior Team Performance (HC)'),
        '1stD Passing__hc_Norm': ('Normalized passing first downs under HC', 'Prior Team Performance (HC)'),
        'Att Rushing__hc_Norm': ('Normalized rush attempts under HC', 'Prior Team Performance (HC)'),
        'Yds Rushing__hc_Norm': ('Normalized rushing yards under HC', 'Prior Team Performance (HC)'),
        'TD Rushing__hc_Norm': ('Normalized rushing touchdowns under HC', 'Prior Team Performance (HC)'),
        'Y/A Rushing__hc_Norm': ('Normalized yards per rush under HC', 'Prior Team Performance (HC)'),
        '1stD Rushing__hc_Norm': ('Normalized rushing first downs under HC', 'Prior Team Performance (HC)'),
        'Pen__hc_Norm': ('Normalized penalties under HC', 'Prior Team Performance (HC)'),
        'Yds Penalties__hc_Norm': ('Normalized penalty yards under HC', 'Prior Team Performance (HC)'),
        '1stPy__hc_Norm': ('Normalized first downs by penalty under HC', 'Prior Team Performance (HC)'),
        '#Dr__hc_Norm': ('Normalized number of drives under HC', 'Prior Team Performance (HC)'),
        'Sc%__hc_Norm': ('Normalized scoring percentage under HC', 'Prior Team Performance (HC)'),
        'TO%__hc_Norm': ('Normalized turnover percentage under HC', 'Prior Team Performance (HC)'),
        'Time Average Drive__hc_Norm': ('Normalized average drive time under HC', 'Prior Team Performance (HC)'),
        'Plays Average Drive__hc_Norm': ('Normalized average plays per drive under HC', 'Prior Team Performance (HC)'),
        'Yds Average Drive__hc_Norm': ('Normalized average yards per drive under HC', 'Prior Team Performance (HC)'),
        'Pts Average Drive__hc_Norm': ('Normalized average points per drive under HC', 'Prior Team Performance (HC)'),
        '3DAtt__hc_Norm': ('Normalized third down attempts under HC', 'Prior Team Performance (HC)'),
        '3D%__hc_Norm': ('Normalized third down conversion rate under HC', 'Prior Team Performance (HC)'),
        '4DAtt__hc_Norm': ('Normalized fourth down attempts under HC', 'Prior Team Performance (HC)'),
        '4D%__hc_Norm': ('Normalized fourth down conversion rate under HC', 'Prior Team Performance (HC)'),
        'RZAtt__hc_Norm': ('Normalized red zone attempts under HC', 'Prior Team Performance (HC)'),
        'RZPct__hc_Norm': ('Normalized red zone scoring percentage under HC', 'Prior Team Performance (HC)'),
    }
    features.update(hc_team_features)
    
    # Add HC opponent performance features
    hc_opp_features = {
        'PF (Points For)__opp__hc_Norm': ('Normalized points allowed under HC', 'Prior Team Performance (HC)'),
        'Yds__opp__hc_Norm': ('Normalized yards allowed under HC', 'Prior Team Performance (HC)'),
        'Y/P__opp__hc_Norm': ('Normalized yards per play allowed under HC', 'Prior Team Performance (HC)'),
        'TO__opp__hc_Norm': ('Normalized takeaways under HC', 'Prior Team Performance (HC)'),
        '1stD__opp__hc_Norm': ('Normalized first downs allowed under HC', 'Prior Team Performance (HC)'),
        'Cmp Passing__opp__hc_Norm': ('Normalized pass completions allowed under HC', 'Prior Team Performance (HC)'),
        'Att Passing__opp__hc_Norm': ('Normalized pass attempts faced under HC', 'Prior Team Performance (HC)'),
        'Yds Passing__opp__hc_Norm': ('Normalized passing yards allowed under HC', 'Prior Team Performance (HC)'),
        'TD Passing__opp__hc_Norm': ('Normalized passing touchdowns allowed under HC', 'Prior Team Performance (HC)'),
        'Int Passing__opp__hc_Norm': ('Normalized interceptions by defense under HC', 'Prior Team Performance (HC)'),
        'NY/A Passing__opp__hc_Norm': ('Normalized net yards per pass allowed under HC', 'Prior Team Performance (HC)'),
        '1stD Passing__opp__hc_Norm': ('Normalized passing first downs allowed under HC', 'Prior Team Performance (HC)'),
        'Att Rushing__opp__hc_Norm': ('Normalized rush attempts faced under HC', 'Prior Team Performance (HC)'),
        'Yds Rushing__opp__hc_Norm': ('Normalized rushing yards allowed under HC', 'Prior Team Performance (HC)'),
        'TD Rushing__opp__hc_Norm': ('Normalized rushing touchdowns allowed under HC', 'Prior Team Performance (HC)'),
        'Y/A Rushing__opp__hc_Norm': ('Normalized yards per rush allowed under HC', 'Prior Team Performance (HC)'),
        '1stD Rushing__opp__hc_Norm': ('Normalized rushing first downs allowed under HC', 'Prior Team Performance (HC)'),
        'Pen__opp__hc_Norm': ('Normalized opponent penalties under HC', 'Prior Team Performance (HC)'),
        'Yds Penalties__opp__hc_Norm': ('Normalized opponent penalty yards under HC', 'Prior Team Performance (HC)'),
        '1stPy__opp__hc_Norm': ('Normalized opponent first downs by penalty under HC', 'Prior Team Performance (HC)'),
        '#Dr__opp__hc_Norm': ('Normalized opponent drives under HC', 'Prior Team Performance (HC)'),
        'Sc%__opp__hc_Norm': ('Normalized opponent scoring percentage under HC', 'Prior Team Performance (HC)'),
        'TO%__opp__hc_Norm': ('Normalized opponent turnover percentage under HC', 'Prior Team Performance (HC)'),
        'Time Average Drive__opp__hc_Norm': ('Normalized opponent average drive time under HC', 'Prior Team Performance (HC)'),
        'Plays Average Drive__opp__hc_Norm': ('Normalized opponent average plays per drive under HC', 'Prior Team Performance (HC)'),
        'Yds Average Drive__opp__hc_Norm': ('Normalized opponent average yards per drive under HC', 'Prior Team Performance (HC)'),
        'Pts Average Drive__opp__hc_Norm': ('Normalized opponent average points per drive under HC', 'Prior Team Performance (HC)'),
        '3DAtt__opp__hc_Norm': ('Normalized opponent third down attempts under HC', 'Prior Team Performance (HC)'),
        '3D%__opp__hc_Norm': ('Normalized opponent third down conversion rate under HC', 'Prior Team Performance (HC)'),
        '4DAtt__opp__hc_Norm': ('Normalized opponent fourth down attempts under HC', 'Prior Team Performance (HC)'),
        '4D%__opp__hc_Norm': ('Normalized opponent fourth down conversion rate under HC', 'Prior Team Performance (HC)'),
        'RZAtt__opp__hc_Norm': ('Normalized opponent red zone attempts under HC', 'Prior Team Performance (HC)'),
        'RZPct__opp__hc_Norm': ('Normalized opponent red zone scoring percentage under HC', 'Prior Team Performance (HC)'),
    }
    features.update(hc_opp_features)
    
    # Target variable
    features['Win_Pct'] = ('Winning percentage - target variable for prediction', 'Hiring Team Context')
    
    # Create DataFrame
    feature_list = []
    for idx, (name, (description, category)) in enumerate(features.items(), 1):
        feature_list.append({
            'Index': idx,
            'Feature_Name': name,
            'Category': category,
            'Description': description
        })
    
    df = pd.DataFrame(feature_list)
    
    # Save to CSV
    output_path = os.path.join('data', 'final', 'feature_documentation_corrected.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Feature documentation saved to {output_path}")
    print(f"Total features documented: {len(features)}")
    
    # Print category counts
    category_counts = df.groupby('Category').size().sort_values(ascending=False)
    print("\nFeatures by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Document features in coaching WAR dataset')
    parser.add_argument('--with-av', action='store_true', 
                        help='Include Approximate Value (AV) metrics in documentation (default: exclude)')
    
    args = parser.parse_args()
    
    df = generate_feature_descriptions(include_av=args.with_av)
    
    av_status = "included" if args.with_av else "excluded"
    print(f"\nAV features {av_status}")
    print("\nFirst 10 features:")
    print(df.head(10).to_string())