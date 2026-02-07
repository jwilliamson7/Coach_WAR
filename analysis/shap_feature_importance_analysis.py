"""
SHAP Feature Importance Analysis for Coaching WAR XGBoost Model

Computes SHAP values using TreeExplainer and provides category-level
importance analysis across 8 feature categories. Generates visualizations
showing total and average importance by feature type.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import argparse
import warnings
import sys
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Add parent directory to path for figure_utils import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure_utils import configure_matplotlib_fonts, get_color_palette


class TeeOutput:
    """Tee output to both stdout and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_and_prepare_data(filepath, exclude_av=True):
    """Load the combined dataset and prepare features."""
    print("Loading data...")
    df = pd.read_csv(filepath)

    if 'Win_Pct' not in df.columns:
        raise ValueError("Win_Pct column not found in dataset")

    X = df.drop(['Win_Pct'], axis=1)
    y = df['Win_Pct']

    # Exclude AV features by default
    if exclude_av:
        av_cols = [col for col in X.columns if 'AV' in col or 'Approximate_Value' in col]
        if av_cols:
            X = X.drop(columns=av_cols)
            print(f"Excluded {len(av_cols)} AV features")

    # Store team and year for later analysis
    if 'Team' in X.columns and 'Year' in X.columns:
        team_year_info = df[['Team', 'Year']].copy()
        if X['Team'].dtype == 'object':
            X = X.drop(['Team', 'Year'], axis=1)
        else:
            X = X.drop(['Year'], axis=1)
    else:
        team_year_info = pd.DataFrame(index=df.index)

    # Convert any remaining object columns to numeric
    for col in X.select_dtypes(include=['object']).columns:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            X = X.drop([col], axis=1)
            print(f"Dropped non-numeric column: {col}")

    X = X.fillna(0)

    print(f"Data shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, team_year_info, df


def categorize_features(feature_names):
    """
    Map each feature to one of 8 categories using pattern matching.

    Categories (checked in priority order to avoid overlap):
      1. Coaching Performance - ends with __oc_Norm, __dc_Norm, __hc_Norm, __opp__hc_Norm
      2. Coaching Experience  - exact match: num_times_hc, num_yr_*
      3. Draft Strategy       - contains 'Round' and 'Picks'
      4. Approximate Value    - contains 'AV' (only present when --with-av)
      5. Team Performance     - Team_*_Norm, Opp_*_Norm, SoS
      6. Player Age & Exp     - Avg/StdDev for Age/Experience/Exp
      7. Injury/Availability  - Games_Missed, Players_Count, Total_Games_In_Season
      8. Roster Turnover      - Players_Retained/Departed/New, *_Rate_*, Net_Change, crosstab
      9. Salary Cap           - remaining *_Pct columns (cap/positional spending)

    Returns:
        dict: {feature_name: category_name}
    """
    coaching_perf_suffixes = ('__oc_Norm', '__dc_Norm', '__hc_Norm', '__opp__hc_Norm')
    coaching_exp_features = {
        'num_times_hc', 'num_yr_col_pos', 'num_yr_col_coor',
        'num_yr_col_hc', 'num_yr_nfl_pos', 'num_yr_nfl_coor', 'num_yr_nfl_hc'
    }

    categories = {}
    for feat in feature_names:
        if any(feat.endswith(s) for s in coaching_perf_suffixes):
            categories[feat] = 'Coaching Performance'
        elif feat in coaching_exp_features:
            categories[feat] = 'Coaching Experience'
        elif 'Round' in feat and 'Picks' in feat:
            categories[feat] = 'Draft Strategy'
        elif 'AV' in feat:
            categories[feat] = 'Approximate Value'
        elif (feat.startswith(('Team_', 'Opp_')) and '_Norm' in feat) or feat == 'SoS':
            categories[feat] = 'Team Performance'
        elif any(x in feat for x in [
            'Avg_Starter_Age', 'StdDev_Starter_Age',
            'Avg_Starter_Exp', 'StdDev_Starter_Exp',
            'Avg_Roster_Age', 'StdDev_Roster_Age',
            'Avg_Roster_Exp', 'StdDev_Roster_Exp',
            'Avg_Starter_Experience', 'StdDev_Starter_Experience',
            'Avg_Roster_Experience', 'StdDev_Roster_Experience'
        ]):
            categories[feat] = 'Player Age & Experience'
        elif any(x in feat for x in ['Games_Missed', 'Players_Count']) or feat == 'Total_Games_In_Season':
            categories[feat] = 'Injury/Availability'
        elif any(x in feat for x in [
            'Players_Retained', 'Players_Departed', 'Players_New',
            'Retention_Rate', 'Departure_Rate', 'New_Player_Rate',
            'Net_Change', 'crosstab'
        ]):
            categories[feat] = 'Roster Turnover'
        elif feat.endswith('_Pct'):
            categories[feat] = 'Salary Cap'
        else:
            categories[feat] = 'Uncategorized'

    # Report categorization results
    from collections import Counter
    counts = Counter(categories.values())
    print(f"\nFeature categorization ({len(feature_names)} features):")
    for cat in sorted(counts.keys()):
        print(f"  {cat}: {counts[cat]}")

    uncategorized = [f for f, c in categories.items() if c == 'Uncategorized']
    if uncategorized:
        print(f"\n  WARNING: {len(uncategorized)} uncategorized features:")
        for f in uncategorized:
            print(f"    - {f}")

    return categories


def train_model(X, y, team_year_info, use_tuning=True, cv_folds=5, n_iter=50,
                test_size=0.2, random_state=42):
    """Train XGBoost model with coach-based stratification."""

    # Load coach data for stratification
    try:
        coach_df = pd.read_csv('data/processed/Coaching/team_year_head_coaches.csv')

        if 'Team' in team_year_info.columns and 'Year' in team_year_info.columns:
            stratify_df = team_year_info.reset_index(drop=True).copy()
            stratify_df = stratify_df.merge(
                coach_df[['Team', 'Year', 'Primary_Coach']],
                on=['Team', 'Year'],
                how='left'
            )

            coach_seasons = stratify_df.groupby('Primary_Coach').size()
            print(f"\nFound {len(coach_seasons)} unique coaches")
            print(f"Coach season distribution: min={coach_seasons.min()}, max={coach_seasons.max()}, mean={coach_seasons.mean():.1f}")

            unique_coaches = coach_seasons.index.tolist()
            coaches_train, coaches_test = train_test_split(
                unique_coaches, test_size=test_size, random_state=random_state
            )

            train_mask = stratify_df['Primary_Coach'].isin(coaches_train)
            test_mask = stratify_df['Primary_Coach'].isin(coaches_test)

            missing_coach_mask = stratify_df['Primary_Coach'].isna()
            train_mask = train_mask | missing_coach_mask

            print(f"Coach-based split:")
            print(f"  Training coaches: {len(coaches_train)} (seasons: {train_mask.sum()})")
            print(f"  Test coaches: {len(coaches_test)} (seasons: {test_mask.sum()})")
            print(f"  Missing coach data: {missing_coach_mask.sum()} (assigned to train)")
        else:
            raise ValueError("Missing Team/Year for coach stratification")

    except Exception as e:
        print(f"Warning: Could not perform coach-based stratification ({e}). Using random split.")
        train_mask = np.random.RandomState(random_state).random(len(X)) >= test_size
        test_mask = ~train_mask

    X_reset = X.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)

    X_train = X_reset[train_mask].copy()
    X_test = X_reset[test_mask].copy()
    y_train = y_reset[train_mask].copy()
    y_test = y_reset[test_mask].copy()

    print(f"\nFinal split sizes:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    if use_tuning:
        print(f"\nPerforming hyperparameter tuning with RandomizedSearchCV...")
        print(f"CV folds: {cv_folds}, Iterations: {n_iter}")

        param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05],
            'max_depth': [2, 3, 4, 5],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'reg_alpha': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0, 1.5, 2.0],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=random_state,
            verbosity=0,
            n_jobs=-1
        )

        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )

        random_search.fit(X_train, y_train)
        model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        print(f"\nBest CV R² score (training): {best_score:.4f}")
        print("Best parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print("\nUsing default hyperparameters (no tuning)...")
        model_params = {
            'n_estimators': 250,
            'learning_rate': 0.05,
            'max_depth': 3,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'objective': 'reg:squarederror',
            'random_state': random_state,
            'verbosity': 0,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_metrics = {
        'r2': r2_score(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred)
    }
    test_metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred)
    }

    print(f"\n{'='*60}")
    print("TRAIN/TEST PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<12} {'Training':<15} {'Test':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'R²':<12} {train_metrics['r2']:<15.4f} {test_metrics['r2']:<15.4f} {test_metrics['r2'] - train_metrics['r2']:<15.4f}")
    print(f"{'RMSE':<12} {train_metrics['rmse']:<15.4f} {test_metrics['rmse']:<15.4f} {test_metrics['rmse'] - train_metrics['rmse']:<15.4f}")
    print(f"{'MAE':<12} {train_metrics['mae']:<15.4f} {test_metrics['mae']:<15.4f} {test_metrics['mae'] - train_metrics['mae']:<15.4f}")

    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff > 0.1:
        print(f"\nWARNING: Possible overfitting detected (R² difference: {r2_diff:.4f})")
    elif r2_diff > 0.05:
        print(f"\nCAUTION: Moderate train/test performance gap (R² difference: {r2_diff:.4f})")
    else:
        print(f"\nGood generalization (R² difference: {r2_diff:.4f})")

    return model, X_train, X_test, y_train, y_test, train_metrics, test_metrics


def compute_shap_values(model, X):
    """Compute SHAP values using TreeExplainer."""
    print(f"\nComputing SHAP values using TreeExplainer...")
    print(f"  Dataset shape: {X.shape}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    print(f"  SHAP values shape: {shap_values.values.shape}")
    print(f"  Expected value (base rate): {explainer.expected_value:.4f}")

    return shap_values, explainer


def analyze_category_importance(shap_values, feature_names, categories):
    """
    Compute per-feature and per-category SHAP importance.

    Returns:
        feature_df: Per-feature importance with categories
        category_df: Category-level aggregated importance
    """
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Category': [categories[f] for f in feature_names]
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    feature_df['Rank'] = range(1, len(feature_df) + 1)

    total_importance = feature_df['Mean_Abs_SHAP'].sum()

    # Category-level aggregation
    category_agg = feature_df.groupby('Category').agg(
        Total_SHAP=('Mean_Abs_SHAP', 'sum'),
        Avg_SHAP=('Mean_Abs_SHAP', 'mean'),
        Max_Feature_SHAP=('Mean_Abs_SHAP', 'max'),
        Feature_Count=('Mean_Abs_SHAP', 'count')
    ).reset_index()

    category_agg['Pct_Total_Importance'] = (category_agg['Total_SHAP'] / total_importance * 100).round(2)
    category_agg = category_agg.sort_values('Total_SHAP', ascending=False).reset_index(drop=True)

    # Print category importance sorted by total
    print(f"\n{'='*90}")
    print("FEATURE CATEGORY IMPORTANCE (sorted by Total SHAP)")
    print(f"{'='*90}")
    print(f"\n{'Category':<28} {'Total SHAP':<14} {'Avg SHAP':<14} {'Max Feature':<14} {'Count':<8} {'% Total'}")
    print("-" * 90)
    for _, row in category_agg.iterrows():
        print(f"{row['Category']:<28} {row['Total_SHAP']:<14.6f} {row['Avg_SHAP']:<14.6f} "
              f"{row['Max_Feature_SHAP']:<14.6f} {int(row['Feature_Count']):<8} {row['Pct_Total_Importance']:.1f}%")
    print("-" * 90)
    print(f"{'TOTAL':<28} {total_importance:<14.6f} {'':<14} {'':<14} {len(feature_names):<8} 100.0%")

    # Print category importance sorted by average
    category_by_avg = category_agg.sort_values('Avg_SHAP', ascending=False)
    print(f"\n{'='*90}")
    print("FEATURE CATEGORY IMPORTANCE (sorted by Average SHAP per feature)")
    print(f"{'='*90}")
    print(f"\n{'Category':<28} {'Avg SHAP':<14} {'Total SHAP':<14} {'Max Feature':<14} {'Count':<8} {'% Total'}")
    print("-" * 90)
    for _, row in category_by_avg.iterrows():
        print(f"{row['Category']:<28} {row['Avg_SHAP']:<14.6f} {row['Total_SHAP']:<14.6f} "
              f"{row['Max_Feature_SHAP']:<14.6f} {int(row['Feature_Count']):<8} {row['Pct_Total_Importance']:.1f}%")

    return feature_df, category_agg


def print_top_features_by_category(feature_df, n=10):
    """Print the top N features within each category."""
    categories_sorted = (feature_df.groupby('Category')['Mean_Abs_SHAP']
                         .sum().sort_values(ascending=False).index)

    for category in categories_sorted:
        cat_features = feature_df[feature_df['Category'] == category].head(n)
        print(f"\n{'='*70}")
        print(f"TOP {min(n, len(cat_features))} FEATURES: {category} ({len(feature_df[feature_df['Category'] == category])} total)")
        print(f"{'='*70}")
        print(f"{'Rank':<6} {'Feature':<50} {'Mean |SHAP|'}")
        print("-" * 70)
        for _, row in cat_features.iterrows():
            print(f"{int(row['Rank']):<6} {row['Feature']:<50} {row['Mean_Abs_SHAP']:.6f}")


def analyze_coaching_subcategories(shap_values, feature_names):
    """
    Drill down into Coaching Performance features by role subcategory.

    Subcategories:
      - Coaching: OC          - Offensive coordinator track record (__oc_Norm)
      - Coaching: DC          - Defensive coordinator track record (__dc_Norm)
      - Coaching: HC Offense  - HC's team offensive performance (__hc_Norm)
      - Coaching: HC Defense  - HC's team defensive performance (__opp__hc_Norm)

    Returns:
        coaching_feature_df: Per-feature importance with subcategories
        coaching_subcat_df: Subcategory-level aggregated importance
    """
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)

    subcategories = {}
    for i, feat in enumerate(feature_names):
        if feat.endswith('__oc_Norm'):
            subcategories[feat] = 'Coaching: OC'
        elif feat.endswith('__dc_Norm'):
            subcategories[feat] = 'Coaching: DC'
        elif feat.endswith('__opp__hc_Norm'):
            subcategories[feat] = 'Coaching: HC Defense'
        elif feat.endswith('__hc_Norm'):
            subcategories[feat] = 'Coaching: HC Offense'

    # Build feature-level dataframe for coaching features only
    coaching_data = []
    for i, feat in enumerate(feature_names):
        if feat in subcategories:
            coaching_data.append({
                'Feature': feat,
                'Mean_Abs_SHAP': mean_abs_shap[i],
                'Subcategory': subcategories[feat]
            })

    coaching_feature_df = pd.DataFrame(coaching_data)
    coaching_feature_df = coaching_feature_df.sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    coaching_feature_df['Rank'] = range(1, len(coaching_feature_df) + 1)

    total_coaching_importance = coaching_feature_df['Mean_Abs_SHAP'].sum()

    # Subcategory-level aggregation
    coaching_subcat_df = coaching_feature_df.groupby('Subcategory').agg(
        Total_SHAP=('Mean_Abs_SHAP', 'sum'),
        Avg_SHAP=('Mean_Abs_SHAP', 'mean'),
        Max_Feature_SHAP=('Mean_Abs_SHAP', 'max'),
        Feature_Count=('Mean_Abs_SHAP', 'count')
    ).reset_index()

    coaching_subcat_df['Pct_Coaching_Importance'] = (
        coaching_subcat_df['Total_SHAP'] / total_coaching_importance * 100
    ).round(2)
    coaching_subcat_df = coaching_subcat_df.sort_values('Total_SHAP', ascending=False).reset_index(drop=True)

    # Print subcategory importance
    print(f"\n{'='*90}")
    print("COACHING PERFORMANCE SUBCATEGORY IMPORTANCE (sorted by Total SHAP)")
    print(f"{'='*90}")
    print(f"\n{'Subcategory':<28} {'Total SHAP':<14} {'Avg SHAP':<14} {'Max Feature':<14} {'Count':<8} {'% Coaching'}")
    print("-" * 90)
    for _, row in coaching_subcat_df.iterrows():
        print(f"{row['Subcategory']:<28} {row['Total_SHAP']:<14.6f} {row['Avg_SHAP']:<14.6f} "
              f"{row['Max_Feature_SHAP']:<14.6f} {int(row['Feature_Count']):<8} {row['Pct_Coaching_Importance']:.1f}%")
    print("-" * 90)
    print(f"{'TOTAL COACHING':<28} {total_coaching_importance:<14.6f}")

    # Print by average
    subcat_by_avg = coaching_subcat_df.sort_values('Avg_SHAP', ascending=False)
    print(f"\n{'='*90}")
    print("COACHING PERFORMANCE SUBCATEGORY IMPORTANCE (sorted by Average SHAP)")
    print(f"{'='*90}")
    print(f"\n{'Subcategory':<28} {'Avg SHAP':<14} {'Total SHAP':<14} {'Max Feature':<14} {'Count':<8} {'% Coaching'}")
    print("-" * 90)
    for _, row in subcat_by_avg.iterrows():
        print(f"{row['Subcategory']:<28} {row['Avg_SHAP']:<14.6f} {row['Total_SHAP']:<14.6f} "
              f"{row['Max_Feature_SHAP']:<14.6f} {int(row['Feature_Count']):<8} {row['Pct_Coaching_Importance']:.1f}%")

    # Print top 5 features per subcategory
    for subcat in coaching_subcat_df['Subcategory']:
        sub_features = coaching_feature_df[coaching_feature_df['Subcategory'] == subcat].head(5)
        print(f"\n  Top 5 in {subcat}:")
        for _, row in sub_features.iterrows():
            # Strip the suffix for cleaner display
            clean_name = row['Feature']
            for suffix in ['__oc_Norm', '__dc_Norm', '__opp__hc_Norm', '__hc_Norm']:
                clean_name = clean_name.replace(suffix, '')
            print(f"    {clean_name:<45} {row['Mean_Abs_SHAP']:.6f}")

    return coaching_feature_df, coaching_subcat_df


def generate_visualizations(shap_values, feature_df, category_df, coaching_subcat_df,
                            top_n=30, font='serif'):
    """Generate all SHAP visualizations."""
    configure_matplotlib_fonts(font)
    colors = get_color_palette()

    output_dir = 'analysis/outputs/png'
    os.makedirs(output_dir, exist_ok=True)

    # Assign a distinct color to each category
    category_colors = [
        colors['primary'],      # Blue
        colors['secondary'],    # Purple
        colors['tertiary'],     # Orange
        colors['quaternary'],   # Red
        colors['highlight1'],   # Teal
        colors['highlight2'],   # Gold
        colors['neutral'],      # Grey
        '#5B8C5A',              # Green
        '#E07A5F',              # Salmon (fallback)
    ]

    # Map categories to colors (sorted by total importance)
    cat_sorted = category_df.sort_values('Total_SHAP', ascending=False)['Category'].tolist()
    cat_color_map = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(cat_sorted)}

    # --- Chart 1: Total SHAP importance by category ---
    print("\nGenerating category total importance chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data = category_df.sort_values('Total_SHAP', ascending=True)
    bar_colors = [cat_color_map[c] for c in plot_data['Category']]
    bars = ax.barh(plot_data['Category'], plot_data['Total_SHAP'], color=bar_colors, edgecolor='white')

    # Add percentage labels
    for bar, pct in zip(bars, plot_data['Pct_Total_Importance']):
        ax.text(bar.get_width() + plot_data['Total_SHAP'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center', fontsize=12)

    ax.set_xlabel('Total SHAP Importance (sum of mean |SHAP| values)')
    ax.set_title('Feature Category Importance: Total SHAP')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_category_total_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    # --- Chart 2: Average SHAP importance by category ---
    print("Generating category average importance chart...")
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_data = category_df.sort_values('Avg_SHAP', ascending=True)
    bar_colors = [cat_color_map[c] for c in plot_data['Category']]
    bars = ax.barh(plot_data['Category'], plot_data['Avg_SHAP'], color=bar_colors, edgecolor='white')

    # Add feature count labels
    for bar, count in zip(bars, plot_data['Feature_Count']):
        ax.text(bar.get_width() + plot_data['Avg_SHAP'].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'n={int(count)}', va='center', fontsize=12)

    ax.set_xlabel('Average SHAP Importance (mean |SHAP| per feature)')
    ax.set_title('Feature Category Importance: Average SHAP per Feature')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_category_avg_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    # --- Chart 3: SHAP beeswarm plot for top features ---
    print(f"Generating SHAP beeswarm plot (top {top_n} features)...")
    fig = plt.figure(figsize=(14, max(10, top_n * 0.35)))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.title(f'SHAP Feature Importance (Top {top_n} Features)')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_beeswarm_top_features.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    # --- Chart 4: SHAP bar plot for top features ---
    print(f"Generating SHAP bar plot (top {top_n} features)...")
    fig = plt.figure(figsize=(12, max(8, top_n * 0.3)))
    shap.plots.bar(shap_values, max_display=top_n, show=False)
    plt.title(f'Mean |SHAP| Value (Top {top_n} Features)')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_bar_top_features.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    # --- Chart 5: Top features within each category (multi-panel) ---
    print("Generating per-category top features grid...")
    categories_sorted = category_df.sort_values('Total_SHAP', ascending=False)['Category'].tolist()
    n_cats = len(categories_sorted)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()

    for i, category in enumerate(categories_sorted):
        ax = axes[i]
        cat_features = feature_df[feature_df['Category'] == category].head(10)
        cat_features_plot = cat_features.sort_values('Mean_Abs_SHAP', ascending=True)

        # Shorten feature names for readability
        short_names = []
        for name in cat_features_plot['Feature']:
            if len(name) > 35:
                name = name[:32] + '...'
            short_names.append(name)

        ax.barh(range(len(cat_features_plot)), cat_features_plot['Mean_Abs_SHAP'],
                color=cat_color_map[category], edgecolor='white')
        ax.set_yticks(range(len(cat_features_plot)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_title(f'{category}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Mean |SHAP|', fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for i in range(n_cats, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Top Features by Category (SHAP Importance)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_top_features_by_category.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")

    # --- Chart 6: Coaching Performance subcategory breakdown ---
    print("Generating coaching subcategory breakdown chart...")
    subcat_colors = {
        'Coaching: OC': colors['tertiary'],         # Orange
        'Coaching: DC': colors['primary'],           # Blue
        'Coaching: HC Offense': colors['quaternary'], # Red
        'Coaching: HC Defense': colors['highlight1'], # Teal
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Left panel: Total SHAP by subcategory
    plot_data = coaching_subcat_df.sort_values('Total_SHAP', ascending=True)
    bar_colors = [subcat_colors.get(c, colors['neutral']) for c in plot_data['Subcategory']]
    bars = ax1.barh(plot_data['Subcategory'], plot_data['Total_SHAP'], color=bar_colors, edgecolor='white')
    for bar, pct in zip(bars, plot_data['Pct_Coaching_Importance']):
        ax1.text(bar.get_width() + plot_data['Total_SHAP'].max() * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'{pct:.1f}%', va='center', fontsize=12)
    ax1.set_xlabel('Total SHAP Importance')
    ax1.set_title('Coaching Performance: Total SHAP by Role')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: Average SHAP by subcategory
    plot_data = coaching_subcat_df.sort_values('Avg_SHAP', ascending=True)
    bar_colors = [subcat_colors.get(c, colors['neutral']) for c in plot_data['Subcategory']]
    bars = ax2.barh(plot_data['Subcategory'], plot_data['Avg_SHAP'], color=bar_colors, edgecolor='white')
    for bar, count in zip(bars, plot_data['Feature_Count']):
        ax2.text(bar.get_width() + plot_data['Avg_SHAP'].max() * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'n={int(count)}', va='center', fontsize=12)
    ax2.set_xlabel('Average SHAP per Feature')
    ax2.set_title('Coaching Performance: Avg SHAP by Role')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.suptitle('Coaching Performance Subcategory Breakdown', fontsize=16, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'shap_coaching_subcategory_breakdown.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def save_results(feature_df, category_df, coaching_feature_df, coaching_subcat_df):
    """Save SHAP analysis results to CSV files."""
    print("\nSaving results...")

    feature_file = 'data/final/shap_feature_importance.csv'
    feature_df.to_csv(feature_file, index=False)
    print(f"  Per-feature importance saved to: {feature_file}")

    category_file = 'data/final/shap_category_importance.csv'
    category_df.to_csv(category_file, index=False)
    print(f"  Category importance saved to: {category_file}")

    coaching_feature_file = 'data/final/shap_coaching_feature_importance.csv'
    coaching_feature_df.to_csv(coaching_feature_file, index=False)
    print(f"  Coaching feature importance saved to: {coaching_feature_file}")

    coaching_subcat_file = 'data/final/shap_coaching_subcategory_importance.csv'
    coaching_subcat_df.to_csv(coaching_subcat_file, index=False)
    print(f"  Coaching subcategory importance saved to: {coaching_subcat_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='SHAP Feature Importance Analysis for Coaching WAR')
    parser.add_argument('--with-av', action='store_true',
                        help='Include AV (Approximate Value) features in analysis')
    parser.add_argument('--no-tuning', action='store_true',
                        help='Skip hyperparameter tuning and use default parameters')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds for tuning (default: 5)')
    parser.add_argument('--n-iter', type=int, default=50,
                        help='Number of iterations for RandomizedSearchCV (default: 50)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Number of top features to display in plots (default: 30)')
    parser.add_argument('--font', type=str, default='serif',
                        help='Font family for plots (default: serif)')
    args = parser.parse_args()

    # Create log file
    os.makedirs('analysis/outputs/logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'analysis/outputs/logs/shap_analysis_log_{timestamp}.txt'

    tee = TeeOutput(log_filename)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        filepath = 'data/final/imputed_final_data.csv'
        exclude_av = not args.with_av
        dataset_type = "with ALL features (including AV)" if args.with_av else "WITHOUT AV features"

        print("=" * 80)
        print("SHAP FEATURE IMPORTANCE ANALYSIS")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Using dataset {dataset_type}")
        if not args.no_tuning:
            print(f"Hyperparameter tuning: ENABLED ({args.n_iter} iterations, {args.cv_folds} CV folds)")
        else:
            print("Hyperparameter tuning: DISABLED (using default parameters)")
        print(f"Log file: {log_filename}")
        print("=" * 80)

        # Step 1: Load data
        X, y, team_year_info, full_df = load_and_prepare_data(filepath, exclude_av=exclude_av)

        # Step 2: Categorize features
        categories = categorize_features(list(X.columns))

        # Step 3: Train model
        print("\nTraining XGBoost model...")
        use_tuning = not args.no_tuning
        model, X_train, X_test, y_train, y_test, train_metrics, test_metrics = train_model(
            X, y, team_year_info,
            use_tuning=use_tuning, cv_folds=args.cv_folds,
            n_iter=args.n_iter, random_state=args.random_state
        )

        # Step 4: Compute SHAP values on full dataset
        shap_values, explainer = compute_shap_values(model, X)

        # Step 5: Analyze category importance
        feature_df, category_df = analyze_category_importance(shap_values, list(X.columns), categories)

        # Step 6: Print top features by category
        print_top_features_by_category(feature_df, n=10)

        # Step 7: Coaching Performance subcategory drill-down
        coaching_feature_df, coaching_subcat_df = analyze_coaching_subcategories(
            shap_values, list(X.columns))

        # Step 8: Generate visualizations
        generate_visualizations(shap_values, feature_df, category_df, coaching_subcat_df,
                                top_n=args.top_n, font=args.font)

        # Step 9: Save results
        save_results(feature_df, category_df, coaching_feature_df, coaching_subcat_df)

        # Summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        print(f"\nSummary:")
        print(f"  Dataset: {dataset_type}")
        print(f"  Features analyzed: {len(X.columns)}")
        print(f"  Categories: {len(category_df)}")
        print(f"  Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}")
        print(f"\n  Most important category (total): {category_df.iloc[0]['Category']} ({category_df.iloc[0]['Pct_Total_Importance']:.1f}%)")
        cat_by_avg = category_df.sort_values('Avg_SHAP', ascending=False)
        print(f"  Most important category (avg):   {cat_by_avg.iloc[0]['Category']} (avg SHAP: {cat_by_avg.iloc[0]['Avg_SHAP']:.6f})")
        print(f"  Top feature overall: {feature_df.iloc[0]['Feature']} ({feature_df.iloc[0]['Category']}, SHAP: {feature_df.iloc[0]['Mean_Abs_SHAP']:.6f})")
        print(f"\nOutputs:")
        print(f"  - data/final/shap_feature_importance.csv")
        print(f"  - data/final/shap_category_importance.csv")
        print(f"  - data/final/shap_coaching_feature_importance.csv")
        print(f"  - data/final/shap_coaching_subcategory_importance.csv")
        print(f"  - analysis/outputs/png/shap_category_total_importance.png")
        print(f"  - analysis/outputs/png/shap_category_avg_importance.png")
        print(f"  - analysis/outputs/png/shap_beeswarm_top_features.png")
        print(f"  - analysis/outputs/png/shap_bar_top_features.png")
        print(f"  - analysis/outputs/png/shap_top_features_by_category.png")
        print(f"  - analysis/outputs/png/shap_coaching_subcategory_breakdown.png")
        print(f"  - {log_filename}")

    finally:
        sys.stdout = original_stdout
        tee.close()
        print(f"\nAnalysis complete. Log saved to: {log_filename}")


if __name__ == "__main__":
    main()
