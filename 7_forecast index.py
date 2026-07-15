import pandas as pd
import numpy as np

# Aggregation of forecasts to index level

# Prepare data
def build_index_forecasts_with_consistent_info(
    *,
    compustat: pd.DataFrame,
    forecasts_1yr: pd.DataFrame,
    forecasts_2yr: pd.DataFrame,
    sp500_quarterly_crsp: pd.DataFrame,
    all_comp_agg: pd.DataFrame,
    kind: str  
) -> tuple:
    
    comp = compustat.copy()
    comp["quarter_end"] = pd.to_datetime(comp["quarter_end"]).dt.normalize()
    
    sp = sp500_quarterly_crsp.copy()
    sp["quarter_end"] = pd.to_datetime(sp["quarter_end"]).dt.normalize()
    
    agg = all_comp_agg.copy()
    agg["quarter_end"] = pd.to_datetime(agg["quarter_end"]).dt.normalize()
    
    f1 = forecasts_1yr.copy()
    f1["estimate_date"] = pd.to_datetime(f1["estimate_date"]).dt.normalize()
    f1["quarter_end"] = pd.to_datetime(f1["quarter_end"]).dt.normalize()
    
    f2 = forecasts_2yr.copy()
    f2["estimate_date"] = pd.to_datetime(f2["estimate_date"]).dt.normalize()
    f2["quarter_end"] = pd.to_datetime(f2["quarter_end"]).dt.normalize()
    
    # Merge with Compustat at estimate_date 
    merged_1yr = f1.merge(
        comp[["permno", "quarter_end", "shares_out_scaled", "mktcap"]],
        left_on=["permno", "estimate_date"],
        right_on=["permno", "quarter_end"],
        how="left",
        suffixes=("", "_comp")
    ).drop(columns=["quarter_end_comp"], errors="ignore")
    
    merged_2yr = f2.merge(
        comp[["permno", "quarter_end", "shares_out_scaled", "mktcap"]],
        left_on=["permno", "estimate_date"],
        right_on=["permno", "quarter_end"],
        how="left",
        suffixes=("", "_comp")
    ).drop(columns=["quarter_end_comp"], errors="ignore")
    
    missing_1yr = merged_1yr['mktcap'].isna().sum()
    missing_2yr = merged_2yr['mktcap'].isna().sum()
    
    if missing_1yr > 0:
        merged_1yr = merged_1yr.dropna(subset=['mktcap', 'shares_out_scaled'])
    
    if missing_2yr > 0:
        merged_2yr = merged_2yr.dropna(subset=['mktcap', 'shares_out_scaled'])
    
    # Convert to dollars and index points    
    merged_1yr["forecast_dollars"] = merged_1yr["forecast"] * merged_1yr["shares_out_scaled"]
    merged_2yr["forecast_dollars"] = merged_2yr["forecast"] * merged_2yr["shares_out_scaled"]
    
    merged_1yr = merged_1yr.merge(
        sp[["quarter_end", "divisor"]],
        left_on="estimate_date",
        right_on="quarter_end",
        how="left",
        suffixes=("", "_div")
    ).drop(columns=["quarter_end_div"], errors="ignore")
    
    merged_2yr = merged_2yr.merge(
        sp[["quarter_end", "divisor"]],
        left_on="estimate_date",
        right_on="quarter_end",
        how="left",
        suffixes=("", "_div")
    ).drop(columns=["quarter_end_div"], errors="ignore")
    
    missing_div_1yr = merged_1yr['divisor'].isna().sum()
    missing_div_2yr = merged_2yr['divisor'].isna().sum()
    
    if missing_div_1yr > 0:
        merged_1yr = merged_1yr.dropna(subset=['divisor'])
    
    if missing_div_2yr > 0:
        merged_2yr = merged_2yr.dropna(subset=['divisor'])
    
    # Calculate index contribution
    merged_1yr["forecast_index_contrib"] = merged_1yr["forecast_dollars"] / merged_1yr["divisor"]
    merged_2yr["forecast_index_contrib"] = merged_2yr["forecast_dollars"] / merged_2yr["divisor"]
        
    # Aggregate to index level by estimate_date    
    out_1yr = (
        merged_1yr.groupby("estimate_date", as_index=False)
                  .agg(
                      forecast_dollars=("forecast_dollars", "sum"),
                      forecast_index_firms=("forecast_index_contrib", "sum"),
                      mktcap_firms=("mktcap", "sum"),
                      num_firms=("permno", "nunique"),
                      divisor=("divisor", "first"),
                      quarter_end=("quarter_end", "first")
                  )
    )
    
    out_2yr = (
        merged_2yr.groupby("estimate_date", as_index=False)
                  .agg(
                      forecast_dollars=("forecast_dollars", "sum"),
                      forecast_index_firms=("forecast_index_contrib", "sum"),
                      mktcap_firms=("mktcap", "sum"),
                      num_firms=("permno", "nunique"),
                      divisor=("divisor", "first"),
                      quarter_end=("quarter_end", "first")
                  )
    )
        
    # Merge total market cap and actual indices at estimate_date    
    out_1yr = out_1yr.merge(
        agg[["quarter_end", "total_mktcap", "dividend_index", "earnings_index"]],
        left_on="estimate_date",
        right_on="quarter_end",
        how="left",
        suffixes=("_forecast", "")
    ).drop(columns=["quarter_end"], errors="ignore")
    if "quarter_end_forecast" in out_1yr.columns:
        out_1yr = out_1yr.rename(columns={"quarter_end_forecast": "quarter_end"})
    
    out_2yr = out_2yr.merge(
        agg[["quarter_end", "total_mktcap", "dividend_index", "earnings_index"]],
        left_on="estimate_date",
        right_on="quarter_end",
        how="left",
        suffixes=("_forecast", "")
    ).drop(columns=["quarter_end"], errors="ignore")
    if "quarter_end_forecast" in out_2yr.columns:
        out_2yr = out_2yr.rename(columns={"quarter_end_forecast": "quarter_end"})
    
    missing_totalmkt_1yr = out_1yr['total_mktcap'].isna().sum()
    missing_totalmkt_2yr = out_2yr['total_mktcap'].isna().sum()
    
    if missing_totalmkt_1yr > 0:
        out_1yr = out_1yr.dropna(subset=['total_mktcap'])
    
    if missing_totalmkt_2yr > 0:
        out_2yr = out_2yr.dropna(subset=['total_mktcap'])
        
    # Calculate scaling ratio and final index    
    out_1yr["scaling_ratio"] = out_1yr["total_mktcap"] / out_1yr["mktcap_firms"]
    out_2yr["scaling_ratio"] = out_2yr["total_mktcap"] / out_2yr["mktcap_firms"]
    
    colname_1yr = f"{kind}_index_1yr"
    colname_2yr = f"{kind}_index_2yr"
    
    out_1yr[colname_1yr] = out_1yr["forecast_index_firms"] * out_1yr["scaling_ratio"]
    out_2yr[colname_2yr] = out_2yr["forecast_index_firms"] * out_2yr["scaling_ratio"]
        
    return out_1yr, out_2yr

# Calculate growth rates
def calculate_growth_rates_consistent(
    index_1yr: pd.DataFrame,
    index_2yr: pd.DataFrame,
    kind: str  
) -> pd.DataFrame:
    
    i1 = index_1yr.copy()
    i2 = index_2yr.copy()
    
    if kind == "dividends":
        col_1yr = "dividends_index_1yr"
        col_2yr = "dividends_index_2yr"
        actual_col = "dividend_index"
        growth_1yr_col = "div_growth_1yr"
        growth_2yr_col = "div_growth_2yr"
    else:  
        col_1yr = "earnings_index_1yr"
        col_2yr = "earnings_index_2yr"
        actual_col = "earnings_index"
        growth_1yr_col = "ear_growth_1yr"
        growth_2yr_col = "ear_growth_2yr"
    
    # Calculate 1-year growth: log(forecast_1yr) - log(actual_t)
    i1[growth_1yr_col] = np.where(
        (i1[col_1yr] > 0) & (i1[actual_col] > 0),
        np.log(i1[col_1yr]) - np.log(i1[actual_col]),
        np.nan
    )
    
    # Merge 1yr and 2yr indices on estimate_date
    combined = i1.merge(
        i2[["estimate_date", col_2yr]],
        on="estimate_date",
        how="left"
    )
    
    # Calculate 2-year growth: log(forecast_2yr) - log(forecast_1yr)
    combined[growth_2yr_col] = np.where(
        (combined[col_2yr] > 0) & (combined[col_1yr] > 0),
        np.log(combined[col_2yr]) - np.log(combined[col_1yr]),
        np.nan
    )
    
    valid_1yr = combined[growth_1yr_col].notna().sum()
    valid_2yr = combined[growth_2yr_col].notna().sum()
    
    return combined

# Execution
eps_1_index, eps_2_index = build_index_forecasts_with_consistent_info(
    compustat=all_comp_sp500,
    forecasts_1yr=eps_1yr,
    forecasts_2yr=eps_2yr,
    sp500_quarterly_crsp=sp500_quarterly_crsp,
    all_comp_agg=all_comp_agg,
    kind="earnings"
)

eps_combined = calculate_growth_rates_consistent(
    index_1yr=eps_1_index,
    index_2yr=eps_2_index,
    kind="earnings"
)

dps_1_index, dps_2_index = build_index_forecasts_with_consistent_info(
    compustat=all_comp_sp500,
    forecasts_1yr=dps_1yr,
    forecasts_2yr=dps_2yr,
    sp500_quarterly_crsp=sp500_quarterly_crsp,
    all_comp_agg=all_comp_agg,
    kind="dividends"
)

dps_combined = calculate_growth_rates_consistent(
    index_1yr=dps_1_index,
    index_2yr=dps_2_index,
    kind="dividends"
)

# Consistency check
eps_check = eps_1_index[['estimate_date', 'mktcap_firms', 'num_firms', 'scaling_ratio']].merge(
    eps_2_index[['estimate_date', 'mktcap_firms', 'num_firms', 'scaling_ratio']],
    on='estimate_date',
    suffixes=('_1yr', '_2yr')
)

mktcap_diff = (eps_check['mktcap_firms_1yr'] - eps_check['mktcap_firms_2yr']).abs()
firms_diff = (eps_check['num_firms_1yr'] - eps_check['num_firms_2yr']).abs()
scaling_diff = (eps_check['scaling_ratio_1yr'] - eps_check['scaling_ratio_2yr']).abs()

dps_check = dps_1_index[['estimate_date', 'mktcap_firms', 'num_firms', 'scaling_ratio']].merge(
    dps_2_index[['estimate_date', 'mktcap_firms', 'num_firms', 'scaling_ratio']],
    on='estimate_date',
    suffixes=('_1yr', '_2yr')
)

mktcap_diff = (dps_check['mktcap_firms_1yr'] - dps_check['mktcap_firms_2yr']).abs()
firms_diff = (dps_check['num_firms_1yr'] - dps_check['num_firms_2yr']).abs()
scaling_diff = (dps_check['scaling_ratio_1yr'] - dps_check['scaling_ratio_2yr']).abs()

# Merge growth rates back into original index dataframes
eps_1_index = eps_1_index.merge(
    eps_combined[['estimate_date', 'ear_growth_1yr']],
    on='estimate_date',
    how='left'
)

eps_2_index = eps_2_index.merge(
    eps_combined[['estimate_date', 'ear_growth_2yr']],
    on='estimate_date',
    how='left'
)

dps_1_index = dps_1_index.merge(
    dps_combined[['estimate_date', 'div_growth_1yr']],
    on='estimate_date',
    how='left'
)

dps_2_index = dps_2_index.merge(
    dps_combined[['estimate_date', 'div_growth_2yr']],
    on='estimate_date',
    how='left'
)