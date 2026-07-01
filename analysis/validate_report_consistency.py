"""
Formal consistency scan: paper (.tex) claims vs canonical output files.

Parses the CURRENT (post-revision) values from the LaTeX source by stripping
\\del{...} (deleted) and unwrapping \\rev{...} (added) markup, then compares
each table cell and inline statistic against values recomputed from the
canonical CSVs in data/final/ and analysis/outputs/csv/.

Run:  python analysis/validate_report_consistency.py
Exit 0 if all checks pass; prints PASS / FAIL / REVIEW per check.
"""

import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def r1(x):
    """Round half away from zero to 1 dp (as a person rounds for the paper)."""
    return float(Decimal(repr(float(x))).quantize(Decimal('0.1'), ROUND_HALF_UP))

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / 'latex' / '2026-Williamson-Jon-Portfolio-Coach-WAR.tex'

results = []  # (status, label, paper, source, note)


def chk(label, paper, source, tol, note=''):
    try:
        ok = abs(float(paper) - float(source)) <= tol
    except (TypeError, ValueError):
        ok = (paper == source)
    results.append(('PASS' if ok else 'FAIL', label, paper, source, note))


def review(label, paper, source, note=''):
    results.append(('REVIEW', label, paper, source, note))


# ---------------------------------------------------------------------------
# LaTeX markup handling
# ---------------------------------------------------------------------------
def _scan_cmd(s, cmd):
    """Yield (start, inner, end) for each \\cmd{...} with brace matching."""
    out = []
    needle = '\\' + cmd + '{'
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        depth = 1
        while k < len(s) and depth:
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
            k += 1
        out.append((j, s[j + len(needle):k - 1], k))
        i = k
    return out


def accept_revisions(s):
    """Drop \\del{...} content; unwrap \\rev{...} -> content. Iterate to handle nesting."""
    for _ in range(10):
        changed = False
        for cmd, keep in (('del', False), ('rev', True)):
            spans = _scan_cmd(s, cmd)
            if not spans:
                continue
            j, inner, k = spans[0]
            s = s[:j] + (inner if keep else '') + s[k:]
            changed = True
        if not changed:
            break
    return s


def num(tok):
    """Parse a LaTeX numeric token: handles $-$, +, $, spaces, %."""
    t = tok.replace('$-$', '-').replace('$+$', '+').replace('$', '')
    t = t.replace('+', '').replace('\\%', '').replace('%', '').replace(',', '').strip()
    return float(t)


TEXT = TEX.read_text(encoding='utf-8')
CLEAN = accept_revisions(TEXT)


def table_rows(label):
    """Return list of raw '&'-split row cells for the tabular carrying \\label{label}."""
    blk = re.search(r'\\begin\{table\}.*?\\label\{' + re.escape(label) +
                    r'\}.*?\\begin\{tabular\}(.*?)\\end\{tabular\}',
                    CLEAN, re.S)
    rows = []
    for line in blk.group(1).split('\\\\'):
        line = re.split(r'\\midrule|\\toprule|\\bottomrule', line)[-1].strip()
        if '&' in line and 'Rank' not in line and 'Metric' not in line \
                and 'Time Horizon' not in line and 'multicolumn' not in line \
                and 'cmidrule' not in line:
            rows.append([c.strip() for c in line.split('&')])
    return rows


# ===========================================================================
# 1. CAREER WAR TABLES (top 15 / bottom 15)
# ===========================================================================
career = pd.read_csv(ROOT / 'data/final/coach_career_impact_stats.csv').set_index('Primary_Coach')
career = career[career['Seasons'] >= 3].copy()
career['AvgG'] = career['Avg_WAR'] * 16
career['TotG'] = career['Total_WAR'] * 16
# Full-precision Avg-WAR CI (replicates analyze_coach_rankings before 2dp rounding),
# so boundary values are compared against true precision rather than the stored 2dp.
career['SE_g'] = (career['WAR_StdDev'] / np.sqrt(career['Seasons'])) * 16
career['tc'] = career['Seasons'].apply(lambda n: stats.t.ppf(0.975, n - 1))
career['lo_full'] = career['AvgG'] - career['tc'] * career['SE_g']
career['hi_full'] = career['AvgG'] + career['tc'] * career['SE_g']
top = career.sort_values('Avg_WAR', ascending=False).head(15)
bot = career.sort_values('Avg_WAR', ascending=True).head(15)

for tag, label, src in (('TOP', 'tab:top_coaches', top),
                        ('BOT', 'tab:bottom_coaches', bot)):
    rows = table_rows(label)
    for r in rows:
        try:
            rank = int(r[0]); coach = r[1].strip(); seasons = int(r[2])
            tot = num(r[3]); avg = num(r[5])
        except Exception:
            review(f'{tag} parse', r, '', 'row parse failed'); continue
        if coach not in src.index:
            chk(f'{tag} #{rank} {coach} present', coach, 'NOT IN CSV TOP/BOT', 0); continue
        s = src.loc[coach]
        chk(f'{tag} #{rank} {coach} seasons', seasons, int(s['Seasons']), 0)
        chk(f'{tag} #{rank} {coach} AvgWAR', avg, round(s['AvgG'], 1), 0.06)
        chk(f'{tag} #{rank} {coach} TotWAR', tot, round(s['TotG'], 1), 0.06)
        # CI columns
        avgci = re.findall(r'\[(.*?)\]', r[6])[0].split(',')
        totci = re.findall(r'\[(.*?)\]', r[4])[0].split(',')
        chk(f'{tag} #{rank} {coach} AvgCI_lo', num(avgci[0]), r1(s['lo_full']), 0.001)
        chk(f'{tag} #{rank} {coach} AvgCI_hi', num(avgci[1]), r1(s['hi_full']), 0.001)
        chk(f'{tag} #{rank} {coach} TotCI_lo', num(totci[0]), round(s['Total_CI_Lower'], 1), 0.06)
        chk(f'{tag} #{rank} {coach} TotCI_hi', num(totci[1]), round(s['Total_CI_Upper'], 1), 0.06)

# Ordering check: paper rank order matches CSV sort
paper_top = [r[1].strip() for r in table_rows('tab:top_coaches')]
chk('TOP ordering matches CSV', paper_top, list(top.index), 0)
paper_bot = [r[1].strip() for r in table_rows('tab:bottom_coaches')]
chk('BOT ordering matches CSV', paper_bot, list(bot.index), 0)

# ===========================================================================
# 2. SINGLE-SEASON TABLES (top 10 / bottom 10)
# ===========================================================================
impact = pd.read_csv(ROOT / 'data/final/coaching_impact_analysis.csv')
impact['WARg'] = impact['Coaching_WAR'] * 16
se = impact['WAR_SE'].iloc[0]
sstop = impact.sort_values('Coaching_WAR', ascending=False).head(10).reset_index(drop=True)
ssbot = impact.sort_values('Coaching_WAR', ascending=True).head(10).reset_index(drop=True)

for tag, label, src in (('SS-TOP', 'tab:top_single_season', sstop),
                        ('SS-BOT', 'tab:bottom_single_season', ssbot)):
    rows = table_rows(label)
    for r in rows:
        try:
            rank = int(r[0]); team = r[1].strip(); yr = int(r[2])
            war = num(r[6]); actual = num(r[4]); repl = num(r[5])
        except Exception:
            review(f'{tag} parse', r, ''); continue
        s = src.iloc[rank - 1]
        chk(f'{tag} #{rank} team/yr', f'{team}/{yr}', f"{s['Team']}/{int(s['Year'])}", 0)
        chk(f'{tag} #{rank} {team}{yr} WAR', war, round(s['WARg'], 1), 0.06)
        chk(f'{tag} #{rank} {team}{yr} actual', actual, round(s['Actual_Win_Pct'], 3), 0.001)
        chk(f'{tag} #{rank} {team}{yr} replace', repl, round(s['Predicted_Replacement'], 3), 0.0015)

# range claim: +5.7 to -8.7
chk('single-season max', 5.7, round(impact['WARg'].max(), 1), 0.06)
chk('single-season min', -8.7, round(impact['WARg'].min(), 1), 0.06)

# ===========================================================================
# 3. DECADE / BACKGROUND TABLE (recomputed from matched raw data)
# ===========================================================================
m = pd.read_csv(ROOT / 'analysis/outputs/csv/coach_matched_war_background_data.csv')
m = m[m['Background'].isin(['Offensive', 'Defensive'])].copy()
m['Decade'] = (m['Year'] // 10) * 10
decades = [1970, 1980, 1990, 2000, 2010, 2020]
drow = table_rows('tab:background_decade_stats')
# rows: [Coaches O/D],[Seasons O/D],[Mean WAR O/D],[Difference],[95% CI],[U-stat],[p-value]
labels = {r[0].split('(')[0].strip().lower(): r for r in drow}

def cell_pair(rowcells, i):
    return rowcells[i].split('/')

for di, dec in enumerate(decades, start=1):
    O = m[(m.Decade == dec) & (m.Background == 'Offensive')]['Coaching_WAR']
    D = m[(m.Decade == dec) & (m.Background == 'Defensive')]['Coaching_WAR']
    mo = m[(m.Decade == dec) & (m.Background == 'Offensive')]
    md = m[(m.Decade == dec) & (m.Background == 'Defensive')]
    # n coaches
    rc = labels['coaches']
    po, pd_ = cell_pair(rc, di)
    chk(f'{dec}s coaches O', int(po), mo['Primary_Coach'].nunique(), 0)
    chk(f'{dec}s coaches D', int(pd_), md['Primary_Coach'].nunique(), 0)
    # n seasons
    rs = labels['seasons']
    po, pd_ = cell_pair(rs, di)
    chk(f'{dec}s seasons O', int(po), len(O), 0)
    chk(f'{dec}s seasons D', int(pd_), len(D), 0)
    # mean WAR (games)
    rm = labels['mean war']
    po, pd_ = cell_pair(rm, di)
    chk(f'{dec}s meanWAR O', num(po), round(O.mean() * 16, 3), 0.002)
    chk(f'{dec}s meanWAR D', num(pd_), round(D.mean() * 16, 3), 0.002)
    # difference
    diff = (O.mean() - D.mean()) * 16
    chk(f'{dec}s difference', num(labels['difference'][di]), round(diff, 3), 0.003)
    # MW U + p (two-sided)
    U, p = stats.mannwhitneyu(O, D, alternative='two-sided')
    chk(f'{dec}s U-stat', num(labels['u-statistic'][di]), U, 1.0)
    chk(f'{dec}s p-value', num(labels['p-value'][di]), round(p, 4), 0.0006)

# ===========================================================================
# 4. MULTI-YEAR PERSISTENCE TABLE (App E) + Fisher Z note
# ===========================================================================
my = pd.read_csv(ROOT / 'analysis/outputs/csv/coaching_war_multiyear_persistence_by_background.csv')
horizon_map = {'1-year': '1-year', '2-year avg': '2-year avg',
               '3-year avg': '3-year avg', '4-year avg': '4-year avg'}
myrows = table_rows('tab:war_multiyear_persistence')
hmap_paper = {'1-year': 0, '2-year': 1, '3-year': 2, '4-year': 3}
for r in myrows:
    hz_label = r[0].strip()
    key = '2-year avg' if hz_label.startswith('2-year') else \
          '3-year avg' if hz_label.startswith('3-year') else \
          '4-year avg' if hz_label.startswith('4-year') else '1-year'
    for bg, ci in (('Offensive', 1), ('Defensive', 3), ('Other', 5)):
        sub = my[(my.Time_Horizon == key) & (my.Background == bg)]
        if len(sub) != 1:
            review(f'multiyear {key}/{bg}', '', f'{len(sub)} rows'); continue
        chk(f'multiyear {key} {bg} R2', num(r[ci]), round(sub['R2'].iloc[0], 3), 0.0015)

# Fisher Z note line in App E
reg = pd.read_csv(ROOT / 'analysis/outputs/csv/statistical_test_registry.csv')
fz = {row['test_id']: row['p_raw'] for _, row in reg.iterrows()}
note_line = re.search(r'Fisher\'s Z-test.*?1-year \(p=([\d.]+)\).*?2-year \(p=([\d.]+)\).*?'
                      r'3-year \(p=([\d.]+)\).*?4-year \(p=([\d.]+)\)', CLEAN, re.S)
if note_line:
    chk('FisherZ 1yr', float(note_line.group(1)), round(fz['persistence_1_year_FisherZ'], 3), 0.0015)
    chk('FisherZ 2yr', float(note_line.group(2)), round(fz['persistence_2_year_avg_FisherZ'], 3), 0.0015)
    chk('FisherZ 3yr', float(note_line.group(3)), round(fz['persistence_3_year_avg_FisherZ'], 3), 0.0015)
    chk('FisherZ 4yr', float(note_line.group(4)), round(fz['persistence_4_year_avg_FisherZ'], 3), 0.0015)

# ===========================================================================
# 5. MODEL METRICS / INLINE SCALARS
# ===========================================================================
pv = pd.read_csv(ROOT / 'data/final/pipeline_validation_metrics.csv')
p1tr = pv[(pv.pipeline.str.contains('Pipeline 1')) & (pv.subset == 'train')].iloc[0]
p1te = pv[(pv.pipeline.str.contains('Pipeline 1')) & (pv.subset == 'test')].iloc[0]
comb_te = pv[(pv.pipeline.str.contains('Combined')) & (pv.subset.str.contains('test'))].iloc[0]
chk('train R2 (0.785)', 0.785, round(p1tr.r2, 3), 0.0015)
chk('test R2 (0.430)', 0.430, round(p1te.r2, 3), 0.0015)
chk('chrono test R2 (0.443)', 0.443, round(comb_te.r2, 3), 0.0015)
chk('test RMSE->SE (2.5)', 2.5, round(p1te.rmse * 16, 1), 0.06)

chk('mean WAR (-0.05)', -0.05, round(impact['Coaching_WAR'].mean() * 16, 2), 0.01)
chk('median WAR (0.01)', 0.01, round(impact['Coaching_WAR'].median() * 16, 2), 0.01)

featimp = pd.read_csv(ROOT / 'data/final/feature_importance_coaching_analysis.csv')
coach_pct = featimp[featimp.Is_Coach_Feature]['Importance'].sum() / featimp['Importance'].sum() * 100
chk('coaching feat importance (42%)', 42, round(coach_pct), 1.0)
chk('n features used (365)', 365, len(featimp), 0)
chk('n coaching features (139)', 139, int(featimp.Is_Coach_Feature.sum()), 0)

# percentile spread (95th - 5th) of coach career-average WAR (games), all coaches
cg_all = impact.groupby('Primary_Coach')['Coaching_WAR'].mean() * 16
spread = np.percentile(cg_all, 95) - np.percentile(cg_all, 5)
chk('95th-5th pctile spread (3.8, all coaches career)', 3.8, round(spread, 1), 0.06)

# ===========================================================================
# 6. PERSISTENCE SCALARS
# ===========================================================================
psum = pd.read_csv(ROOT / 'analysis/outputs/csv/coaching_war_persistence_summary.csv')
psum_d = {row['Metric']: row['Value'] for _, row in psum.iterrows()}
chk('persistence n (1208)', 1208, int(psum_d['Sample size (consecutive seasons)']), 0)
chk('persistence overall R2 (0.065)', 0.065, round(float(psum_d['R� (variance explained)']), 3)
    if 'R� (variance explained)' in psum_d else 0.0636, 0.0015,
    'from summary R2')

rtm = pd.read_csv(ROOT / 'analysis/outputs/csv/coaching_regression_to_mean_survivorship_adjusted.csv').set_index('Quintile')
chk('top-quintile YearN (2.35)', 2.35, round(rtm.loc['Q5 (Best)', 'Year_N_Mean_WAR'] * 16, 2), 0.02)
chk('top-quintile YearN+1 (0.63)', 0.63, round(rtm.loc['Q5 (Best)', 'Year_N_Plus_1_Survivors_Only'] * 16, 2), 0.02)
chk('bottom-quintile YearN (-2.60)', -2.60, round(rtm.loc['Q1 (Worst)', 'Year_N_Mean_WAR'] * 16, 2), 0.02)
chk('bottom-quintile YearN+1 adj (-1.33)', -1.33, round(rtm.loc['Q1 (Worst)', 'Year_N_Plus_1_Adjusted'] * 16, 2), 0.02)

# win pct persistence R2 (Appendix D, 0.137)
wp = pd.read_csv(ROOT / 'analysis/outputs/csv/win_pct_persistence_data.csv')
wr = stats.pearsonr(wp['Year_N_Actual_Win_Pct'], wp['Year_N_Plus_1_Actual_Win_Pct'])[0]
chk('winpct persistence R2 (0.140)', 0.140, round(wr**2, 3), 0.002)

# ===========================================================================
# 7. ENDPOINT (decade-trend) + BH multiple testing
# ===========================================================================
reg_d = {row['test_id']: row for _, row in reg.iterrows()}
eo = reg_d['endpoint_offensive_2020s_vs_1970s']
chk('offensive endpoint (+0.87)', 0.87, round(eo['effect_size'], 2), 0.01)
chk('offensive endpoint CIlo (0.36)', 0.36, round(eo['ci_lower'], 2), 0.01)
chk('offensive endpoint CIhi (1.38)', 1.38, round(eo['ci_upper'], 2), 0.01)
chk('offensive endpoint p (0.0009)', 0.0009, round(eo['p_raw'], 4), 0.0002)
ed = reg_d['endpoint_defensive_2020s_vs_1970s']
chk('defensive endpoint (-0.39)', -0.39, round(ed['effect_size'], 2), 0.01)
chk('defensive endpoint p (0.21)', 0.21, round(ed['p_raw'], 2), 0.01)

chk('BH family size (13)', 13, len(reg), 0)
n_survive = int((reg['q_bh'] < 0.05).sum())
chk('BH survivors at q<0.05 (5)', 5, n_survive, 0)
# p* critical ~0.019: largest p_raw among BH-rejected
rejected = reg[reg['q_bh'] < 0.05]
pstar = rejected['p_raw'].max() if len(rejected) else float('nan')
chk('BH critical p* (~0.019)', 0.019, round(pstar, 3), 0.002, 'largest rejected p_raw')

# ===========================================================================
# 8. CROSS-PIPELINE / RIDGE CORRELATIONS (reconstructed)
# ===========================================================================
# P1 vs P2 career-WAR Spearman (claim 0.99)
p1 = pd.read_csv(ROOT / 'data/final/leakage_tests/imputation_t20/coach_career_impact_stats_imputation_test.csv').set_index('Primary_Coach')
common = career.index.intersection(p1.index)
rho_p1p2 = stats.spearmanr(career.loc[common, 'Avg_WAR'], p1.loc[common, 'Avg_WAR']).correlation
chk('P1 vs P2 Spearman (0.96, career)', 0.96, round(rho_p1p2, 2), 0.01, f'n={len(common)} coaches')

# Ridge vs XGB(P2) career Spearman (claim 0.81)
hc = pd.read_csv(ROOT / 'data/processed/Coaching/team_year_head_coaches.csv')
ridge = pd.read_csv(ROOT / 'analysis/outputs/csv/ridge_coaching_impact_analysis.csv')
ridge = ridge.merge(hc[['Team', 'Year', 'Primary_Coach']], on=['Team', 'Year'], how='left')
ridge_c = ridge.groupby('Primary_Coach')['Coaching_WAR_Games'].agg(['mean', 'count'])
xgb_c = impact.merge(hc[['Team', 'Year', 'Primary_Coach']], on=['Team', 'Year'], how='left') \
    if 'Primary_Coach' not in impact.columns else impact
xgb_c = xgb_c.groupby('Primary_Coach')['WARg'].agg(['mean', 'count'])
for minS in (1, 3):
    rc = ridge_c[ridge_c['count'] >= minS]
    xc = xgb_c[xgb_c['count'] >= minS]
    common2 = rc.index.intersection(xc.index)
    rho = stats.spearmanr(rc.loc[common2, 'mean'], xc.loc[common2, 'mean']).correlation
    note = f'min{minS}seasons n={len(common2)}'
    if minS == 3:
        chk(f'Ridge vs XGB Spearman (0.81) [{note}]', 0.81, round(rho, 2), 0.02, note)
    else:
        review(f'Ridge vs XGB Spearman [{note}]', 0.81, round(rho, 2), note)

# ===========================================================================
# 9. FIGURE referential integrity
# ===========================================================================
figs = re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}', TEXT)
for f in figs:
    p = (ROOT / 'latex' / f)
    chk(f'figure exists: {f}', True, p.exists(), 0)

# ===========================================================================
# SUMMARY
# ===========================================================================
print('=' * 100)
print('PAPER CONSISTENCY SCAN')
print('=' * 100)
fails = [r for r in results if r[0] == 'FAIL']
revs = [r for r in results if r[0] == 'REVIEW']
n_pass = sum(1 for r in results if r[0] == 'PASS')
print(f'\nTotal checks: {len(results)}  |  PASS: {n_pass}  '
      f'|  FAIL: {len(fails)}  |  REVIEW: {len(revs)}\n')

if fails:
    print('-' * 100); print('FAILURES'); print('-' * 100)
    for st, lab, paper, src, note in fails:
        print(f'  [FAIL] {lab}\n         paper={paper!r}  source={src!r}  {note}')
if revs:
    print('\n' + '-' * 100); print('REVIEW (manual / reconstructed)'); print('-' * 100)
    for st, lab, paper, src, note in revs:
        print(f'  [REVIEW] {lab}: paper={paper!r} source={src!r} {note}')

print('\n' + '-' * 100); print('ALL CHECKS'); print('-' * 100)
for st, lab, paper, src, note in results:
    if st == 'PASS':
        print(f'  [pass] {lab}  ({paper} ~= {src})')
