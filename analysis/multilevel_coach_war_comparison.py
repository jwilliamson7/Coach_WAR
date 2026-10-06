"""
Multilevel (random-intercept) coach-WAR robustness comparison.

The reviewer asked why XGBoost over multilevel models. This fits a Yurko-style
varying-intercept model where the COACH is a random effect (a latent value with
partial pooling), team CONTEXT enters as fixed effects, and all coach-history
features are excluded so the coach intercept absorbs the full coaching signal.
We then correlate the resulting coach rankings with the primary XGBoost career
WAR (Spearman rho), as a robustness check across model families.

Implementation note: statsmodels' iterative REML estimator for MixedLM collapsed
to the zero-variance boundary here (a known instability), even though the true
between-coach variance is clearly nonzero (raw ICC ~ 0.24). We therefore estimate
the one-way random-effects variance components in closed form (method of moments)
and compute the coach effects as empirical-Bayes / BLUP estimates,
  hat_u_i = bar_e_i * tau^2 / (tau^2 + sigma^2 / n_i),
which is exactly the random-intercept model's shrinkage estimator (Gelman and
Hill 2007). Context enters via a first-stage OLS; the coach effect is the
shrunk mean residual.

In-season team-performance outcomes (the non-coach *_Norm features: points,
yards, interceptions, etc.) are excluded from the fixed effects because they are
mediators of coaching, not pre-season context; controlling for them would partial
out the coaching effect we want the random intercept to capture.

Two fixed-effect specifications:
  (1) SHAP-selected: top-N context features by mean|SHAP|, collinearity-pruned.
  (2) PCA: principal components of all context features (orthogonal, stable).

Output: analysis/outputs/csv/multilevel_coach_war_comparison.csv
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.xgboost_coaching_impact_analysis import (  # noqa: E402
    identify_coach_features, load_and_prepare_data,
)

TOP_N = 20
VIF_THRESH = 10.0
PCA_VAR = 0.90
MIN_SEASONS = 3

DATA = 'data/final/imputed_final_data.csv'
SHAP = ROOT / 'data/final/shap_feature_importance.csv'
COACHES = ROOT / 'data/processed/Coaching/team_year_head_coaches.csv'
XGB_CAREER = ROOT / 'data/final/coach_career_impact_stats.csv'
OUT = ROOT / 'analysis/outputs/csv/multilevel_coach_war_comparison.csv'


def variance_components(values, coach, n_per):
    """One-way random-effects variance components (method of moments, unbalanced).

    Returns (tau2 between-coach, sigma2 within, ICC, group_means)."""
    df = pd.DataFrame({'v': values, 'c': coach})
    g = df.groupby('c')['v']
    ni = g.size()
    mi = g.mean()
    N = len(df)
    k = len(ni)
    gm = df['v'].mean()
    SSB = (ni * (mi - gm) ** 2).sum()
    MSB = SSB / (k - 1)
    SSW = ((df['v'] - df['c'].map(mi)) ** 2).sum()
    MSW = SSW / (N - k)
    n0 = (N - (ni ** 2).sum() / N) / (k - 1)
    tau2 = max((MSB - MSW) / n0, 0.0)
    sigma2 = MSW
    icc = tau2 / (tau2 + sigma2) if (tau2 + sigma2) > 0 else 0.0
    return tau2, sigma2, icc, mi


def eb_blup(resid, coach):
    """Empirical-Bayes (random-intercept BLUP) coach effects from residuals."""
    n_per = pd.Series(coach).value_counts()
    tau2, sigma2, icc, mi = variance_components(resid, coach, n_per)
    shrink = tau2 / (tau2 + sigma2 / n_per.reindex(mi.index))
    blup = mi * shrink
    return blup, icc, shrink


def ols_residuals(wins, design):
    """Residuals after a first-stage OLS of wins on standardized design + const."""
    Z = StandardScaler().fit_transform(design)
    Z = sm.add_constant(Z)
    return wins.values - sm.OLS(wins.values, Z).fit().predict(Z)


def prune_vif(df, feats):
    keep = list(feats)
    while len(keep) > 2:
        Z = sm.add_constant(StandardScaler().fit_transform(df[keep]))
        vifs = [variance_inflation_factor(Z, i + 1) for i in range(len(keep))]
        mx = int(np.argmax(vifs))
        if vifs[mx] > VIF_THRESH:
            print(f"   drop {keep[mx]} (VIF={vifs[mx]:.1f})")
            keep.pop(mx)
        else:
            break
    return keep


def compare(blup, xgb, label):
    common = xgb.index.intersection(blup.index)
    sub = xgb.loc[common]
    keep = sub[sub['Seasons'] >= MIN_SEASONS].index
    rho = stats.spearmanr(sub.loc[keep, 'Avg_WAR'], blup.loc[keep]).correlation
    r = stats.pearsonr(sub.loc[keep, 'Avg_WAR'], blup.loc[keep])[0]
    print(f"[{label}] vs XGBoost WAR (>= {MIN_SEASONS} seasons, n={len(keep)}): "
          f"Spearman rho = {rho:.3f}, Pearson r = {r:.3f}")
    return rho


def main():
    print('=' * 80)
    print('MULTILEVEL (RANDOM-INTERCEPT) COACH-WAR ROBUSTNESS COMPARISON')
    print('=' * 80)
    X, y, ty, _ = load_and_prepare_data(DATA, exclude_av=True)
    coach_feats = identify_coach_features(X)
    # context = non-coach features that are NOT in-season performance outcomes
    context_pool = [c for c in X.columns
                    if c not in coach_feats and not c.endswith('_Norm')]
    n_mediator = sum(1 for c in X.columns
                     if c not in coach_feats and c.endswith('_Norm'))
    print(f"\nExcluded {len(coach_feats)} coach features and {n_mediator} in-season "
          f"_Norm outcome (mediator) features.")
    print(f"Context pool: {len(context_pool)} pre-season features.")

    cm = pd.read_csv(COACHES)[['Team', 'Year', 'Primary_Coach']]
    b = ty.reset_index(drop=True).copy()
    b['wins'] = y.values * 16
    b = pd.concat([b, X.reset_index(drop=True)], axis=1).merge(cm, on=['Team', 'Year'], how='left')
    b = b[b['Primary_Coach'].notna()].reset_index(drop=True)
    coach = b['Primary_Coach']
    xgb = pd.read_csv(XGB_CAREER).set_index('Primary_Coach')

    # Unconditional coach ICC (no fixed effects) for reference
    _, _, icc0, _ = variance_components(b['wins'].values, coach, None)
    print(f"\nUnconditional between-coach ICC (raw wins): {icc0:.3f}")

    # ---- Spec 1: SHAP-selected context ----
    print("\n--- Spec 1: SHAP-selected context fixed effects ---")
    shap = pd.read_csv(SHAP)
    top = (shap[shap['Feature'].isin(context_pool)]
           .sort_values('Mean_Abs_SHAP', ascending=False).head(TOP_N)['Feature'].tolist())
    kept = prune_vif(b, top)
    print(f"Kept {len(kept)} context fixed effects.")
    resid1 = ols_residuals(b['wins'], b[kept])
    blup1, icc1, shrink1 = eb_blup(resid1, coach)
    print(f"Context-adjusted ICC: {icc1:.3f}; shrinkage factor "
          f"{shrink1.min():.2f} (few seasons) to {shrink1.max():.2f} (many seasons)")
    rho1 = compare(blup1, xgb, 'SHAP controls')

    # ---- Spec 2: PCA context ----
    print("\n--- Spec 2: PCA context fixed effects ---")
    pcs = PCA(n_components=PCA_VAR, svd_solver='full').fit_transform(
        StandardScaler().fit_transform(b[context_pool]))
    print(f"PCA: {pcs.shape[1]} components retain {PCA_VAR:.0%} variance.")
    resid2 = ols_residuals(b['wins'], pd.DataFrame(pcs))
    blup2, icc2, _ = eb_blup(resid2, coach)
    print(f"Context-adjusted ICC: {icc2:.3f}")
    rho2 = compare(blup2, xgb, 'PCA controls')

    # ---- save ----
    keep = xgb[xgb['Seasons'] >= MIN_SEASONS].index
    out = pd.DataFrame({
        'Seasons': xgb.loc[keep, 'Seasons'],
        'XGB_AvgWAR_games': xgb.loc[keep, 'Avg_WAR'] * 16,
        'Mixed_SHAP_effect_games': blup1.reindex(keep),
        'Mixed_PCA_effect_games': blup2.reindex(keep),
    }).dropna().sort_values('XGB_AvgWAR_games', ascending=False)
    out.to_csv(OUT)

    print('\n' + '=' * 80)
    print('SUMMARY')
    print(f"  Unconditional coach ICC (raw):  {icc0:.3f}")
    print(f"  Spec 1 SHAP controls:  context ICC = {icc1:.3f},  Spearman rho = {rho1:.3f}")
    print(f"  Spec 2 PCA controls:   context ICC = {icc2:.3f},  Spearman rho = {rho2:.3f}")
    print(f"  Saved: {OUT}")
    print('=' * 80)


if __name__ == '__main__':
    main()
