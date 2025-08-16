import pandas as pd

df = pd.read_csv('C:/Personal/Coach_WAR/data/processed/Coaching/yearly_coach_performance.csv')
print(f'Dataset shape: {df.shape}')
print('\nFeatures with data (non-null count > 0):')

count = 0
for col in df.columns:
    non_null = df[col].notna().sum()
    if non_null > 0 and col not in ['Coach', 'Year', 'Team', 'Role', 'Age']:
        count += 1
        if count <= 10:  # Show first 10 as examples
            print(f'  {col}: {non_null} non-null values')

print(f'\nTotal features with data: {count} out of {len(df.columns) - 5} features')

# Check specific feature groups
oc_features = [c for c in df.columns if '__oc' in c]
dc_features = [c for c in df.columns if '__dc' in c]
hc_features = [c for c in df.columns if '__hc' in c and '__opp__' not in c]
opp_features = [c for c in df.columns if '__opp__hc' in c]

print(f'\nOC features with data: {sum(df[c].notna().sum() > 0 for c in oc_features)}/{len(oc_features)}')
print(f'DC features with data: {sum(df[c].notna().sum() > 0 for c in dc_features)}/{len(dc_features)}')
print(f'HC features with data: {sum(df[c].notna().sum() > 0 for c in hc_features)}/{len(hc_features)}')
print(f'HC opponent features with data: {sum(df[c].notna().sum() > 0 for c in opp_features)}/{len(opp_features)}')