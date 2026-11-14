import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data.loader import load_us_equities
from engine.ast_parser import AlphaSyntaxError, parse_alpha
from validation.cpcv import deflated_sharpe_ratio, run_cpcv_validation
from validation.metrics import evaluate_alpha_performance

st.set_page_config(page_title="AlphaForge | Quant Research Terminal", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px;}
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #064e3b 100%);
        border: 1px solid #10b98144; border-radius: 14px;
        padding: 18px 24px; margin-bottom: 18px;
    }
    .hero-title {font-size: 1.75rem; font-weight: 800; color: #f8fafc; margin: 0;}
    .hero-sub {font-size: 0.92rem; color: #cbd5e1; margin-top: 4px;}
    .badge {
        display: inline-block; background: #10b98122; color: #34d399;
        border: 1px solid #10b98155; padding: 2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    div[data-testid="stMetric"] {
        background: #0f172a; border: 1px solid #1e293b;
        padding: 14px 18px; border-radius: 12px; box-shadow: 0 4px 12px #00000033;
    }
</style>
<div class="hero-box">
    <div style="margin-bottom:6px;">
        <span class="badge">US EQUITIES • S&P 500</span>
        <span class="badge">ZERO LOOK-AHEAD CERTIFIED</span>
        <span class="badge">CPCV + DEFLATED SHARPE</span>
    </div>
    <p class="hero-title">⚡ AlphaForge — Neuro-Symbolic Quant Research Terminal</p>
    <p class="hero-sub">Evolutionary Alpha Discovery, Orthogonal Portfolio Construction & Live Symbolic Formula Backtester</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_market_data():
    return load_us_equities(start="2018-01-01", end="2026-01-01")

zoo_path, ret_path = Path("results/alpha_zoo.json"), Path("results/alpha_returns.npz")
if not zoo_path.exists() or not ret_path.exists():
    st.error("Esegui prima `python run_discovery.py` per generare i risultati.")
    st.stop()

zoo = json.loads(zoo_path.read_text(encoding="utf-8"))
npz = np.load(ret_path, allow_pickle=True)
dates = pd.to_datetime(npz["dates"])
st.sidebar.markdown("### 🎛️ Portfolio & Risk Controls")
all_labels = [f"Alpha #{item['rank']} • SR {item['sharpe_net']:.2f}" for item in zoo]
selected_labels = st.sidebar.multiselect("Active Orthogonal Factors", all_labels, default=all_labels)
weight_mode = st.sidebar.radio("Allocation Scheme", ["Equal-Weight", "Inverse-Volatility (Risk Parity)"])
extra_bps = st.sidebar.slider("Extra Slippage Stress (bps)", 0.0, 15.0, 0.0, 0.5)

selected_items = [item for item, lbl in zip(zoo, all_labels) if lbl in selected_labels]
if not selected_items:
    st.warning("Seleziona almeno un fattore Alpha nella sidebar.")
    st.stop()

ret_cols = {}
for item in selected_items:
    r = npz[f"alpha_{item['rank']}"].astype(float) - item["daily_turnover"] * (extra_bps * 1e-4)
    ret_cols[f"Alpha #{item['rank']}"] = r
ret_df = pd.DataFrame(ret_cols, index=dates)

if weight_mode == "Equal-Weight" or ret_df.shape[1] == 1:
    combo_ret = ret_df.mean(axis=1)
else:
    inv_vol = 1.0 / ret_df.std(ddof=1).replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    combo_ret = (ret_df * weights).sum(axis=1)

cum_combo = (1.0 + combo_ret).cumprod()
dd_combo = (cum_combo / cum_combo.cummax()) - 1.0
ann_ret = float(combo_ret.mean() * 252)
ann_vol = float(combo_ret.std(ddof=1) * np.sqrt(252))
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
calmar = ann_ret / abs(float(dd_combo.min())) if dd_combo.min() < 0 else 0.0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Combined Net Sharpe", f"{sharpe:.2f}", f"{weight_mode.split()[0]}")
k2.metric("Annualized Net Return", f"{ann_ret:.1%}")
k3.metric("Annualized Volatility", f"{ann_vol:.1%}")
k4.metric("Max Drawdown", f"{float(dd_combo.min()):.1%}")
k5.metric("Calmar Ratio", f"{calmar:.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tearsheet & Regimes", "🧪 Live Alpha Sandbox", "🧬 Alpha Zoo & Correlations", "🔬 Factor Deep-Dive"
])

with tab1:
    fig_eq = go.Figure()
    palette = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#60a5fa"]
    for i, col in enumerate(ret_df.columns):
        fig_eq.add_trace(go.Scatter(
            x=ret_df.index, y=(1.0 + ret_df[col]).cumprod(),
            name=col, mode="lines", line=dict(width=1.4, color=palette[i % len(palette)]), opacity=0.6
        ))
    fig_eq.add_trace(go.Scatter(
        x=cum_combo.index, y=cum_combo, name="Combined Alpha Portfolio",
        mode="lines", line=dict(color="#10b981", width=3.2)
    ))
    fig_eq.update_layout(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=400, margin=dict(l=12, r=12, t=38, b=10), hovermode="x unified",
        title="Cumulative Net Equity Growth ($1 Dollar-Neutral)", yaxis_title="Wealth ($)",
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right")
    )
    st.plotly_chart(fig_eq, width="stretch")

    c_left, c_right = st.columns(2)
    with c_left:
        roll_sr = (combo_ret.rolling(63).mean() / combo_ret.rolling(63).std()) * np.sqrt(252)
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=roll_sr.index, y=roll_sr, fill="tozeroy", line=dict(color="#38bdf8", width=1.6), name="63d Rolling Sharpe"))
        fig_rs.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_rs.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=270, margin=dict(l=10, r=10, t=36, b=10), title="Rolling 3-Month Sharpe Ratio")
        st.plotly_chart(fig_rs, width="stretch")
    with c_right:
        m_ret = combo_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
        piv = pd.DataFrame({"Year": m_ret.index.year, "Month": m_ret.index.strftime("%b"), "Ret": m_ret.values})
        months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        hm = piv.pivot(index="Year", columns="Month", values="Ret").reindex(columns=months_order)
        fig_hm = px.imshow(hm, text_auto=".1f", color_continuous_scale="RdYlGn", zmin=-5, zmax=5, title="Monthly Net Returns (%)")
        fig_hm.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=270, margin=dict(l=10, r=10, t=36, b=10))
        st.plotly_chart(fig_hm, width="stretch")
with tab2:
    st.subheader("🧪 Interactive Symbolic Alpha Sandbox")
    st.caption("Scrivi o modifica una formula Alpha e testala istantaneamente sui 2.011 giorni x 50 azioni US (Open t+1 -> Open t+2, 3 bps costi).")
    col_in, col_btn = st.columns([4, 1])
    custom_formula = col_in.text_input("Symbolic Formula (AST)", value="-cs_rank(ts_decay_linear(sub(vwap, close), 10))")
    run_custom = col_btn.button("🚀 Run Backtest", width="stretch")
    st.caption("Feature: `open, high, low, close, volume, returns, vwap, adv20` | Operatori: `cs_rank, cs_zscore, ts_mean, ts_std, ts_zscore, ts_delta, ts_corr, ts_decay_linear, add, sub, mul, div, neg`")

    if run_custom or custom_formula:
        try:
            node = parse_alpha(custom_formula)
            ds = get_market_data()
            sig = node.evaluate(ds.to_feature_dict())
            m = evaluate_alpha_performance(sig, ds.fwd_ret_1d, cost_bps=3.0)
            dsr_val = deflated_sharpe_ratio(m.net_returns, num_trials=1)
            cpcv_val = run_cpcv_validation(sig, ds.fwd_ret_1d, cost_bps=3.0)

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Net Sharpe", f"{m.sharpe_net:.2f}", f"Gross: {m.sharpe_gross:.2f}")
            s2.metric("Rank IC", f"{m.mean_ic:.4f}", f"t-stat: {m.ic_tstat:.2f}")
            s3.metric("Daily Turnover", f"{m.daily_turnover:.1%}")
            s4.metric("DSR Prob", f"{dsr_val:.1%}")
            s5.metric("CPCV Mean SR", f"{cpcv_val['cpcv_mean_sharpe']:.2f}", f"{cpcv_val['cpcv_pct_positive']:.0%} pos")

            s_cum = np.cumprod(1.0 + m.net_returns)
            fig_sb = go.Figure()
            fig_sb.add_trace(go.Scatter(x=dates, y=s_cum, name=node.to_formula(), line=dict(color="#10b981", width=2.5)))
            fig_sb.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=320, margin=dict(l=10, r=10, t=35, b=10), title=f"Live Backtest: {node.to_formula()}")
            st.plotly_chart(fig_sb, width="stretch")
        except AlphaSyntaxError as err:
            st.error(f"Errore di sintassi nella formula: {err}")

with tab3:
    clean_table = pd.DataFrame(selected_items)[[
        "rank", "formula", "sharpe_net", "mean_ic", "ic_tstat",
        "daily_turnover", "max_drawdown", "dsr", "cpcv_mean_sharpe", "cpcv_pct_positive"
    ]]
    st.dataframe(clean_table, width="stretch", hide_index=True)
    ca, cb = st.columns([1.1, 1])
    with ca:
        fig_corr = px.imshow(ret_df.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pairwise Return Correlation Matrix")
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=350)
        st.plotly_chart(fig_corr, width="stretch")
    with cb:
        st.markdown("#### Active Alpha Formulas")
        for item in selected_items:
            st.code(f"# Alpha #{item['rank']} (SR: {item['sharpe_net']:.2f} | t-stat: {item['ic_tstat']:.2f})\n{item['formula']}", language="python")

with tab4:
    choice = st.selectbox("Select Factor to Inspect", all_labels)
    item = zoo[all_labels.index(choice)]
    st.code(item["formula"], language="python")
    st.caption(f"Lineage: {item['rationale']}")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Net Sharpe", f"{item['sharpe_net']:.2f}", f"Gross: {item['sharpe_gross']:.2f}")
    d2.metric("Mean Rank IC", f"{item['mean_ic']:.4f}", f"t-stat: {item['ic_tstat']:.2f}")
    d3.metric("Deflated Sharpe", f"{item['dsr']:.1%}")
    d4.metric("CPCV Mean SR", f"{item['cpcv_mean_sharpe']:.2f}", f"{item['cpcv_pct_positive']:.0%} pos")
    s_ret = pd.Series(npz[f"alpha_{item['rank']}"], index=dates) * 100
    fig_h = px.histogram(s_ret, nbins=60, title="Daily Net Return Distribution (%)", color_discrete_sequence=["#10b981"])
    fig_h.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=300, showlegend=False)
    st.plotly_chart(fig_h, width="stretch")
