import json
from pathlib import Path
import numpy as np
import polars as pl
import streamlit as st

st.set_page_config(page_title="AlphaForge | Quant Research Lab", layout="wide")
st.title("AlphaForge: Neuro-Symbolic Evolutionary Alpha Engine")
st.caption("US Equities Dollar-Neutral Statistical Arbitrage | Validated via CPCV & Deflated Sharpe Ratio")

zoo_path = Path("results/alpha_zoo.json")
ret_path = Path("results/alpha_returns.npz")

if not zoo_path.exists() or not ret_path.exists():
    st.warning("Nessun risultato trovato. Esegui prima `python run_discovery.py` nel terminale.")
    st.stop()

zoo = json.loads(zoo_path.read_text(encoding="utf-8"))
npz = np.load(ret_path, allow_pickle=True)
dates = npz["dates"]

df_table = pl.DataFrame(zoo)
st.subheader("1. Discovered Orthogonal Alpha Zoo")
st.dataframe(df_table.to_pandas(), use_container_width=True)

st.subheader("2. Cumulative Net Equity Curves (After 5 bps Transaction Costs)")
ret_matrix = np.column_stack([npz[f"alpha_{item['rank']}"] for item in zoo])
# Portafoglio combinato equamente pesato degli Alpha ortogonali
combo_ret = np.mean(ret_matrix, axis=1)
cum_df = {"Date": dates, "Combined_Alpha_Portfolio": np.cumprod(1.0 + combo_ret)}
for item in zoo:
    r_idx = item["rank"]
    cum_df[f"Alpha_{r_idx}"] = np.cumprod(1.0 + npz[f"alpha_{r_idx}"])

import pandas as pd
chart_df = pd.DataFrame(cum_df).set_index("Date")
st.line_chart(chart_df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("3. Alpha Return Correlation Matrix")
    corr = np.corrcoef(ret_matrix, rowvar=False)
    corr_df = pd.DataFrame(corr, columns=[f"Alpha_{i['rank']}" for i in zoo], index=[f"Alpha_{i['rank']}" for i in zoo])
    st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)

with col2:
    st.subheader("4. Combined Portfolio Stats")
    ann_ret = float(np.mean(combo_ret) * 252)
    ann_vol = float(np.std(combo_ret, ddof=1) * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    st.metric("Combined Net Sharpe Ratio", f"{sharpe:.2f}")
    st.metric("Annualized Net Return", f"{ann_ret:.1%}")
    st.metric("Annualized Volatility", f"{ann_vol:.1%}")
