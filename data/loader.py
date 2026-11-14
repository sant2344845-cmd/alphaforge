from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import polars as pl
import yfinance as yf
from engine.operators import safe_div, ts_mean, ts_pct_change

DEFAULT_US_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "TSLA", "WMT", "UNH", "XOM", "V", "MA", "PG", "COST", "JNJ", "HD",
    "ORCL", "ABBV", "BAC", "CRM", "KO", "NFLX", "CVX", "AMD", "PEP", "TMO",
    "LIN", "ADBE", "CSCO", "ACN", "MCD", "DIS", "ABT", "WFC", "GE", "QCOM",
    "INTU", "TXN", "CAT", "VZ", "AMGN", "IBM", "PM", "NOW", "SPGI", "GS",
]

@dataclass
class MarketDataset:
    dates: np.ndarray
    tickers: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    returns: np.ndarray
    vwap: np.ndarray
    adv20: np.ndarray
    fwd_ret_1d: np.ndarray

    def to_feature_dict(self) -> dict[str, np.ndarray]:
        return {
            "open": self.open, "high": self.high, "low": self.low,
            "close": self.close, "volume": self.volume,
            "returns": self.returns, "vwap": self.vwap, "adv20": self.adv20,
        }

def build_dataset_from_matrices(
    dates: np.ndarray, tickers: list[str],
    open_m: np.ndarray, high_m: np.ndarray, low_m: np.ndarray,
    close_m: np.ndarray, volume_m: np.ndarray,
) -> MarketDataset:
    returns = ts_pct_change(close_m, 1)
    vwap = (high_m + low_m + close_m) / 3.0
    adv20 = ts_mean(close_m * volume_m, 20)
    fwd_ret_1d = np.full_like(close_m, np.nan, dtype=np.float64)
    if close_m.shape[0] > 2:
        fwd_ret_1d[:-2] = safe_div(open_m[2:] - open_m[1:-1], open_m[1:-1])
    return MarketDataset(dates, tickers, open_m, high_m, low_m, close_m, volume_m, returns, vwap, adv20, fwd_ret_1d)

def load_us_equities(
    tickers: list[str] | None = None, start: str = "2018-01-01", end: str = "2026-01-01",
    cache_path: str = "data/us_equities_cache.parquet", force_refresh: bool = False,
) -> MarketDataset:
    tickers = sorted(list(set(tickers or DEFAULT_US_UNIVERSE)))
    cache_file = Path(cache_path)
    if cache_file.exists() and not force_refresh:
        df = pl.read_parquet(cache_file)
    else:
        raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
        records = []
        dates_idx = raw.index.strftime("%Y-%m-%d").to_numpy()
        for field in ["Open", "High", "Low", "Close", "Volume"]:
            sub = raw[field].reindex(columns=tickers)
            for t_i, dt in enumerate(dates_idx):
                row_vals = sub.iloc[t_i].to_numpy(dtype=np.float64)
                for n_i, tk in enumerate(tickers):
                    records.append((dt, tk, field.lower(), float(row_vals[n_i])))
        long_df = pl.DataFrame(records, schema={"date": pl.Utf8, "ticker": pl.Utf8, "field": pl.Utf8, "value": pl.Float64}, orient="row")
        df = long_df.pivot(on="field", index=["date", "ticker"], values="value").sort(["date", "ticker"])
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_file)

    dates = df.select("date").unique().sort("date")["date"].to_numpy()
    avail_tickers = df.select("ticker").unique().sort("ticker")["ticker"].to_list()
    def _to_mat(col_name: str) -> np.ndarray:
        return df.pivot(on="ticker", index="date", values=col_name).sort("date").select(avail_tickers).to_numpy().astype(np.float64)

    return build_dataset_from_matrices(dates, avail_tickers, _to_mat("open"), _to_mat("high"), _to_mat("low"), _to_mat("close"), _to_mat("volume"))
