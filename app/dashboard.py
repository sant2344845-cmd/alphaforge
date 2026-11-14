import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AlphaForge | Quant Research Terminal", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: #111827; border: 1px solid #1f2937;
        padding: 14px 18px; border-radius: 10px;
    }
    div[data-testid="stMetric"] label {color: #9ca3af !important; font-size: 0.85rem;}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {color: #f9fafb !important; font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ AlphaForge — Neuro-Symbolic Quant Research Lab")
st.caption("Cross-Sectional US Equities Statistical Arbitrage • Zero Look-Ahead Certified • Validated via CPCV & Deflated Sharpe Ratio")

zoo_path = Path("results/alpha_zoo.json")
ret_path = Path("results/alpha_returns.npz")
if not zoo_path.exists() or not ret_path.exists():
    st.error("Nessun risultato trovato. Esegui prima `python run_discovery.py`.")
    st.stop()

zoo = json.loads(zoo_path.read_text(encoding="utf-8"))
npz = np.load(ret_path, allow_pickle=True)
dates = pd.to_datetime(npz["dates"])

st.sidebar.header("⚙️ Portfolio Controls")
all_labels = [f"Alpha #{item['rank']} (SR {item['sharpe_net']:.2f})" for item in zoo]
selected_labels = st.sidebar.multiselect("Active Alpha Factors", all_labels, default=all_labels)
extra_cost_bps = st.sidebar.slider("Additional Slippage Stress Test (bps)", 0.0, 15.0, 0.0, 0.5)

selected_items = [item for item, lbl in zip(zoo, all_labels) if lbl in selected_labels]
if not selected_items:
    st.warning("Seleziona almeno un fattore Alpha dalla barra laterale.")
    st.stop()

ret_cols = {}
for item in selected_items:
    r = npz[f"alpha_{item['rank']}"].astype(float)
    r_stressed = r - item["daily_turnover"] * (extra_cost_bps * 1e-4)
    ret_cols[f"Alpha #{item['rank']}"] = r_stressed

ret_df = pd.DataFrame(ret_cols, index=dates)
combo_ret = ret_df.mean(axis=1)
cum_combo = (1.0 + combo_ret).cumprod()
dd_combo = (cum_combo / cum_combo.cummax()) - 1.0

ann_ret = float(combo_ret.mean() * 252)
ann_vol = float(combo_ret.std(ddof=1) * np.sqrt(252))
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
max_dd = float(dd_combo.min())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Combined Net Sharpe", f"{sharpe:.2f}", delta=f"{len(selected_items)} Orthogonal Alphas")
c2.metric("Annualized Net Return", f"{ann_ret:.1%}")
c3.metric("Annualized Volatility", f"{ann_vol:.1%}")
c4.metric("Max Drawdown", f"{max_dd:.1%}")
tab1, tab2, tab3 = st.tabs(["📈 Equity & Drawdown", "🧬 Alpha Zoo & Correlations", "🔬 Factor Deep-Dive"])

with tab1:
    fig_eq = go.Figure()
    for col in ret_df.columns:
        fig_eq.add_trace(go.Scatter(
            x=ret_df.index, y=(1.0 + ret_df[col]).cumprod(),
            name=col, mode="lines", line=dict(width=1.5), opacity=0.65
        ))
    fig_eq.add_trace(go.Scatter(
        x=cum_combo.index, y=cum_combo,
        name="Combined Alpha Portfolio (Equal-Weight)",
        mode="lines", line=dict(color="#10b981", width=3.2)
    ))
    fig_eq.update_layout(
        template="plotly_dark", height=430, margin=dict(l=10, r=10, t=35, b=10),
        title="Cumulative Net Wealth ($1 Invested, Dollar-Neutral)",
        yaxis_title="Growth of $1", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_eq, width="stretch")

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd_combo.index, y=dd_combo * 100, fill="tozeroy",
        name="Portfolio Drawdown (%)", line=dict(color="#ef4444", width=1.5)
    ))
    fig_dd.update_layout(
        template="plotly_dark", height=220, margin=dict(l=10, r=10, t=30, b=10),
        title="Underwater Drawdown Curve (%)", yaxis_title="Drawdown %", hovermode="x unified"
    )
    st.plotly_chart(fig_dd, width="stretch")

with tab2:
    st.subheader("Discovered Orthogonal Alpha Zoo")
    clean_table = pd.DataFrame(selected_items)[[
        "rank", "formula", "sharpe_net", "mean_ic", "ic_tstat",
        "daily_turnover", "max_drawdown", "dsr", "cpcv_mean_sharpe", "cpcv_pct_positive"
    ]]
    st.dataframe(clean_table, width="stretch", hide_index=True)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        corr = ret_df.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pairwise Return Correlation (Orthogonality Check)"
        )
        fig_corr.update_layout(template="plotly_dark", height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_corr, width="stretch")
    with col_b:
        st.info(
            "**Why Orthogonality Matters:** Combining alpha factors with pairwise return correlation "
            "below `0.45` cancels idiosyncratic factor drawdowns and boosts the Combined Portfolio Sharpe Ratio "
            "above the individual components."
        )
        st.markdown("**Active Formulas:**")
        for item in selected_items:
            st.code(f"# Alpha #{item['rank']}\n{item['formula']}", language="python")

with tab3:
    choice = st.selectbox("Select an Alpha Factor to inspect", all_labels)
    item = zoo[all_labels.index(choice)]
    st.markdown(f"### Symbolic Formula")
    st.code(item["formula"], language="python")
    st.markdown(f"**Economic / Evolutionary Lineage:** `{item['rationale']}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net Sharpe Ratio", f"{item['sharpe_net']:.2f}", f"Gross: {item['sharpe_gross']:.2f}")
    m2.metric("Mean Rank IC", f"{item['mean_ic']:.4f}", f"t-stat: {item['ic_tstat']:.2f}")
    m3.metric("Deflated Sharpe (DSR)", f"{item['dsr']:.1%}")
    m4.metric("CPCV Mean Sharpe", f"{item['cpcv_mean_sharpe']:.2f}", f"{item['cpcv_pct_positive']:.0%} pos folds")

    single_ret = ret_df[f"Alpha #{item['rank']}"] if f"Alpha #{item['rank']}" in ret_df else pd.Series(npz[f"alpha_{item['rank']}"], index=dates)
    fig_hist = px.histogram(single_ret * 100, nbins=60, title="Daily Net Return Distribution (%)", labels={"value": "Daily Return (%)"})
    fig_hist.update_layout(template="plotly_dark", height=300, showlegend=False)
    st.plotly_chart(fig_hist, width="stretch")
