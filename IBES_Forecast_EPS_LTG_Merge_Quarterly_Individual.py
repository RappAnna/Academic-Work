# ==============================================================================
# GUARANTEED 1-TO-1 MERGE: EPS TO LTG
# ==============================================================================

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("STARTING GUARANTEED 1-TO-1 MERGE")
print("="*80)

# ==============================================================================
# 1. Preparation for Merge
# ==============================================================================

# Preparation: renaming columns of dataframes
eps_data = eps_data.rename(columns={
    "estimated_value": "estimated_eps",
    "forecast_revision": "revision_eps"
})

ltg_for_merge = ltg_data[['cusip', 'estimator', 'estimate_date', 'estimated_value', 'forecast_revision']].copy()
ltg_for_merge = ltg_for_merge.rename(columns={
    "estimated_value": "estimated_ltg",
    "forecast_revision": "revision_ltg"
})

# Ensure merge keys have identical dtypes
eps_data['estimate_date'] = pd.to_datetime(eps_data['estimate_date'])
ltg_for_merge['estimate_date'] = pd.to_datetime(ltg_for_merge['estimate_date'])

# ==============================================================================
# 2. Data Cleaning Before Merge
# ==============================================================================
print("\nCleaning data before merge...")

# Remove rows with missing merge keys
eps_clean = eps_data[
    eps_data['cusip'].notna() & 
    eps_data['estimator'].notna() & 
    eps_data['estimate_date'].notna()
].copy()

ltg_clean = ltg_for_merge[
    ltg_for_merge['cusip'].notna() & 
    ltg_for_merge['estimator'].notna() & 
    ltg_for_merge['estimate_date'].notna()
].copy()

print(f"EPS: {len(eps_data)} -> {len(eps_clean)} (removed {len(eps_data) - len(eps_clean)} with missing keys)")
print(f"LTG: {len(ltg_for_merge)} -> {len(ltg_clean)} (removed {len(ltg_for_merge) - len(ltg_clean)} with missing keys)")

# Ensure cusip and estimator are strings for consistent sorting
eps_clean['cusip'] = eps_clean['cusip'].astype(str)
eps_clean['estimator'] = eps_clean['estimator'].astype(str)
ltg_clean['cusip'] = ltg_clean['cusip'].astype(str)
ltg_clean['estimator'] = ltg_clean['estimator'].astype(str)

# ==============================================================================
# 3. Create All Potential Matches Within Tolerance Window
# ==============================================================================
print("\nCreating all potential matches within 31-day window...")

# Add unique IDs
eps_clean = eps_clean.reset_index(drop=True)
ltg_clean = ltg_clean.reset_index(drop=True)
eps_clean['eps_id'] = eps_clean.index
ltg_clean['ltg_id'] = ltg_clean.index

# Merge on cusip and estimator to get all combinations
potential_matches = eps_clean[['eps_id', 'cusip', 'estimator', 'estimate_date']].merge(
    ltg_clean[['ltg_id', 'cusip', 'estimator', 'estimate_date', 'estimated_ltg', 'revision_ltg']],
    on=['cusip', 'estimator'],
    suffixes=('_eps', '_ltg'),
    how='inner'
)

print(f"  Created {len(potential_matches):,} potential pairs")

# ==============================================================================
# 4. Filter to Matches Within Tolerance (31 Days)
# ==============================================================================
print("\nFiltering to matches within 31-day tolerance...")

# Calculate absolute date difference
potential_matches['date_diff'] = (
    potential_matches['estimate_date_eps'] - potential_matches['estimate_date_ltg']
).dt.days.abs()

# Keep only matches within tolerance
tolerance_days = 31
valid_matches = potential_matches[potential_matches['date_diff'] <= tolerance_days].copy()

print(f"  {len(valid_matches):,} valid matches within tolerance")
print(f"  Removed {len(potential_matches) - len(valid_matches):,} matches outside tolerance")

# ==============================================================================
# 5. Rank Matches and Select Best 1-to-1 Pairing
# ==============================================================================
print("\nRanking matches and selecting best 1-to-1 pairing...")

if len(valid_matches) == 0:
    print("  No valid matches found!")
    eps_ltg = eps_clean.copy()
    eps_ltg['estimated_ltg'] = np.nan
    eps_ltg['revision_ltg'] = np.nan
else:
    # Sort by date_diff to prioritize closest matches
    valid_matches = valid_matches.sort_values('date_diff')
    
    # Track which IDs have been matched
    matched_eps = set()
    matched_ltg = set()
    final_matches = []
    
    print(f"  Processing {len(valid_matches):,} candidate matches...")
    
    # Iterate through sorted matches and accept only 1-to-1 pairings
    for idx, row in valid_matches.iterrows():
        eps_id = row['eps_id']
        ltg_id = row['ltg_id']
        
        # Only accept if both IDs haven't been matched yet
        if eps_id not in matched_eps and ltg_id not in matched_ltg:
            final_matches.append({
                'eps_id': eps_id,
                'ltg_id': ltg_id,
                'estimated_ltg': row['estimated_ltg'],
                'revision_ltg': row['revision_ltg'],
                'date_diff': row['date_diff']
            })
            matched_eps.add(eps_id)
            matched_ltg.add(ltg_id)
    
    print(f"  ✓ Secured {len(final_matches):,} unique 1-to-1 matches")
    
    # ==============================================================================
    # 6. Merge Back to EPS Data
    # ==============================================================================
    print("\nMerging matched LTG data back to EPS...")
    
    # Convert final matches to DataFrame
    matches_df = pd.DataFrame(final_matches)
    
    # Merge with EPS data
    eps_ltg = eps_clean.merge(
        matches_df[['eps_id', 'estimated_ltg', 'revision_ltg', 'date_diff']],
        on='eps_id',
        how='left'
    )
    
    # Verify no duplicate LTG usage
    ltg_matched = eps_ltg[eps_ltg['estimated_ltg'].notna()]
    ltg_value_counts = ltg_matched.groupby(['estimated_ltg', 'revision_ltg']).size()
    max_usage = ltg_value_counts.max() if len(ltg_value_counts) > 0 else 0
    
    print(f"  ✓ Maximum times any LTG row was used: {max_usage}")

# ==============================================================================
# 7. Prepare Final Output
# ==============================================================================
print("\nPreparing final output...")

# Rename columns for clarity
eps_ltg = eps_ltg.rename(columns={
    "actual": "actual_eps",
    "forecast_error": "forecast_error_eps"
})

# Keep only requested columns in order
cols_order = [
    "ticker", "permno", "cusip", "estimator", "estimate_date",
    "fiscal_period_end", "fpi", "estimated_eps", "actual_eps",
    "forecast_error_eps", "revision_eps", "estimated_ltg",
    "revision_ltg", "in_sp500"
]

# Only keep columns that exist
cols_order = [col for col in cols_order if col in eps_ltg.columns]
eps_ltg = eps_ltg[cols_order]

# ==============================================================================
# 8. Final Summary Statistics
# ==============================================================================
print("\n" + "="*80)
print("MERGE RESULTS - GUARANTEED 1-TO-1")
print("="*80)
print(f"Total EPS rows: {len(eps_ltg):,}")
print(f"Matched with LTG: {eps_ltg['estimated_ltg'].notna().sum():,}")
print(f"Unmatched EPS rows: {eps_ltg['estimated_ltg'].isna().sum():,}")
print(f"Match rate: {eps_ltg['estimated_ltg'].notna().sum() / len(eps_ltg) * 100:.1f}%")

# Final verification - check by ltg_id (the actual row identifier)
if 'ltg_id' in matches_df.columns:
    ltg_id_counts = matches_df['ltg_id'].value_counts()
    duplicated_ltg = (ltg_id_counts > 1).sum()
    
    if duplicated_ltg > 0:
        print(f"\n❌ CRITICAL ERROR: {duplicated_ltg} unique LTG forecasts were matched multiple times!")
        print("   This should not happen with the guaranteed 1-to-1 logic.")
    else:
        print(f"\n✓✓✓ SUCCESS: Each LTG forecast matched at most once (1-to-1 verified)")
        print(f"✓✓✓ {len(matched_ltg):,} unique LTG rows used out of {len(ltg_clean):,} available")

# Show date difference statistics for matched pairs
if 'date_diff' in eps_ltg.columns and eps_ltg['date_diff'].notna().any():
    print(f"\nDate Difference Statistics (days):")
    print(f"  Mean: {eps_ltg['date_diff'].mean():.1f}")
    print(f"  Median: {eps_ltg['date_diff'].median():.1f}")
    print(f"  Max: {eps_ltg['date_diff'].max():.0f}")
    print(f"  Min: {eps_ltg['date_diff'].min():.0f}")

print("\nFirst few rows of merged data:")
display_cols = [col for col in ['ticker', 'estimator', 'estimate_date', 'estimated_eps', 'estimated_ltg', 'revision_ltg'] if col in eps_ltg.columns]
print(eps_ltg[display_cols].head(10))

print("\n" + "="*80)
print("MERGE COMPLETE")
print("="*80)