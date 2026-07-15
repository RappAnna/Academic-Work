import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Construction of Subjective Return Expectations

# Load return expectations
path = r"D:\Code\current_historical_cfo_data.xlsx"

# Pre-2020: 2001Q4–2020Q1 
pre = pd.read_excel(path, sheet_name="through_Q1_2020")

pre = pre.rename(columns={
    "sp_1_exp":  "exp_ret_1y",
    "sp_10_exp": "exp_ret_10y",
})

if {"year", "quarter"}.issubset(pre.columns):
    pre["date"] = pd.PeriodIndex(year=pre["year"].astype(int),
                                  quarter=pre["quarter"].astype(int),
                                  freq="Q").to_timestamp(how="end")
    pre = pre.set_index("date")

for c in ["exp_ret_1y", "exp_ret_10y"]:
    pre[c] = pd.to_numeric(pre[c], errors="coerce") / 100.0

pre_df = pre[["exp_ret_1y", "exp_ret_10y"]].copy()

# Post-2020 
try:
    post = pd.read_excel(path, sheet_name="CFO_SP500")
    post = post.rename(columns={
        "sp_12moexp_2_med": "exp_ret_1y",
        "sp_10yrexp_2_med": "exp_ret_10y",
    })
    if {"year", "quarter"}.issubset(post.columns):
        post["date"] = pd.PeriodIndex(year=post["year"].astype(int),
                                       quarter=post["quarter"].astype(int),
                                       freq="Q").to_timestamp(how="end")
        post = post.set_index("date")

    for c in ["exp_ret_1y", "exp_ret_10y"]:
        post[c] = pd.to_numeric(post[c], errors="coerce") / 100.0

    post_df = post[["exp_ret_1y", "exp_ret_10y"]].copy()

    df = (pd.concat([pre_df, post_df])
            .sort_index()
            .loc[~pd.concat([pre_df, post_df]).index.duplicated(keep="first")])

except ValueError:
    df = pre_df

# Restrict to 2003Q1–2023Q3
start_date = pd.Period("2003Q1", freq="Q").to_timestamp(how="end")
end_date   = pd.Period("2023Q3", freq="Q").to_timestamp(how="end")
df = df.loc[start_date:end_date]

gh_return_exp = df.copy()