from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from indicators import adx, atr, bollinger, ema, macd, rsi


def prepare_exec_frame(df: pd.DataFrame, params: Dict[str, float]) -> pd.DataFrame:
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


def build_signals(exec_df: pd.DataFrame) -> pd.DataFrame:
    df = exec_df.copy()
    df["long_signal"] = (
        (df["rsi"] > 50)
        & (df["macd_hist"] > 0)
        & (df["close"] > df["ema_fast"])
        & (df["ema_fast"] > df["ema_slow"])
        & (df["close"] > df["boll_mid"])
    )
    df["short_signal"] = (
        (df["rsi"] < 50)
        & (df["macd_hist"] < 0)
        & (df["close"] < df["ema_fast"])
        & (df["ema_fast"] < df["ema_slow"])
        & (df["close"] < df["boll_mid"])
    )
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
