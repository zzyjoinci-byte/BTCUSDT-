from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from indicators import adx, atr, bollinger, ema, macd, rsi
from resample import resample_ohlcv, timeframe_to_minutes


def prepare_exec_frame(df: pd.DataFrame, params: Dict[str, float], exec_tf: str = "4h") -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = rsi(df["close"], params["rsi_length"])
    macd_line, macd_signal, macd_hist = macd(
        df["close"], params["macd_fast"], params["macd_slow"], params["macd_signal"]
    )
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    mid, upper, lower = bollinger(df["close"], params["boll_length"], params["boll_std"])
    df["boll_mid"] = mid
    df["boll_upper"] = upper
    df["boll_lower"] = lower
    df["ema_fast"] = ema(df["close"], params["ema_fast"])
    df["ema_slow"] = ema(df["close"], params["ema_slow"])
    df["adx"] = adx(df["high"], df["low"], df["close"], params["adx_length"])
    df["atr"] = atr(df["high"], df["low"], df["close"], params["atr_length"])
    df = _add_adx_4h(df, params, exec_tf)
    return df


def prepare_filter_frame(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], params["ema_fast"])
    df["ema_slow"] = ema(df["close"], params["ema_slow"])
    df["adx"] = adx(df["high"], df["low"], df["close"], params["adx_length"])
    return df


def beargate_pass(filter_row: pd.Series, params: Dict[str, float]) -> bool:
    if filter_row.isna().any():
        return False
    adx_ok = filter_row["adx"] >= params["bear_adx_threshold"]
    slope = filter_row["ema_fast"] - filter_row["ema_fast"].shift(params["ema_slope_lookback"])
    slope_val = slope.iloc[-1] if hasattr(slope, "iloc") else slope
    return adx_ok and float(slope_val) < 0


def calc_beargate_series(filter_df: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    slope = filter_df["ema_fast"] - filter_df["ema_fast"].shift(params["ema_slope_lookback"])
    return (filter_df["adx"] >= params["bear_adx_threshold"]) & (slope < 0)


def calc_range_filter(exec_df: pd.DataFrame) -> pd.Series:
    return exec_df["adx"] < 15


def build_signals(exec_df: pd.DataFrame, params: Dict[str, float] | None = None) -> pd.DataFrame:
    df = exec_df.copy()
    df["long_candidate"] = (
        (df["rsi"] > 50)
        & (df["macd_hist"] > 0)
        & (df["close"] > df["ema_fast"])
        & (df["ema_fast"] > df["ema_slow"])
        & (df["close"] > df["boll_mid"])
    )
    df["short_candidate"] = (
        (df["rsi"] < 50)
        & (df["macd_hist"] < 0)
        & (df["close"] < df["ema_fast"])
        & (df["ema_fast"] < df["ema_slow"])
        & (df["close"] < df["boll_mid"])
    )
    if params and params.get("entry_adx_filter_enabled", False):
        adx_4h = df.get("adx_4h", df.get("adx", pd.Series(np.nan, index=df.index)))
        df["entry_adx_ok_long"] = adx_4h >= float(params.get("entry_adx_min_long", 0))
        df["entry_adx_ok_short"] = adx_4h >= float(params.get("entry_adx_min_short", 0))
        df["long_signal"] = df["long_candidate"] & df["entry_adx_ok_long"]
        df["short_signal"] = df["short_candidate"] & df["entry_adx_ok_short"]
    else:
        df["entry_adx_ok_long"] = True
        df["entry_adx_ok_short"] = True
        df["long_signal"] = df["long_candidate"]
        df["short_signal"] = df["short_candidate"]
    df["touch_long"] = df["low"] <= df["boll_lower"]
    df["touch_short"] = df["high"] >= df["boll_upper"]
    return df


def calculate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
    max_notional_pct: float | None = None,
) -> float:
    if entry_price <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0
    risk_amt = equity * risk_pct
    qty = risk_amt / stop_distance
    if max_notional_pct is not None:
        max_qty = equity * max_notional_pct / entry_price
        qty = min(qty, max_qty)
    return max(qty, 0.0)


def tp2_invalid_should_tighten(
    pnl_pct: float,
    hold_bars: int,
    min_pnl_pct: float,
    min_hold_bars: int,
) -> Tuple[bool, str]:
    if pnl_pct <= 0:
        return False, "loss"
    if pnl_pct < min_pnl_pct:
        return False, "profit_gate"
    if hold_bars < min_hold_bars:
        return False, "hold_gate"
    return True, "allow"


def _add_adx_4h(df: pd.DataFrame, params: Dict[str, float], exec_tf: str) -> pd.DataFrame:
    period = int(params.get("entry_adx_period", params.get("adx_length", 14)))
    if df.empty:
        df["adx_4h"] = pd.Series(dtype=float)
        return df
    exec_minutes = timeframe_to_minutes(exec_tf)
    if exec_minutes >= 240:
        if "adx" in df.columns:
            df["adx_4h"] = df["adx"]
        else:
            df["adx_4h"] = adx(df["high"], df["low"], df["close"], period)
        return df
    base_cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    base = df[base_cols].sort_values("open_time")
    resampled = resample_ohlcv(base, "4h")
    resampled["adx_4h"] = adx(resampled["high"], resampled["low"], resampled["close"], period)
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        resampled[["open_time", "adx_4h"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    return merged
