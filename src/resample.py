from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def timeframe_to_minutes(timeframe: str) -> int:
    tf = timeframe.lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"不支持的时间周期: {timeframe}")


def to_datetime_ms(series_ms: pd.Series) -> pd.Series:
    return pd.to_datetime(series_ms, unit="ms", utc=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rule = timeframe
    df = df.copy()
    df["dt"] = to_datetime_ms(df["open_time"])
    df = df.set_index("dt")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "close_time": "last",
    }
    resampled = df.resample(rule, label="left", closed="left").agg(agg).dropna()
    resampled = resampled.reset_index(drop=False)
    resampled["open_time"] = (resampled["dt"].view("int64") // 1_000_000).astype(int)
    resampled = resampled.drop(columns=["dt"])
    return resampled


def merge_filter_to_exec(exec_df: pd.DataFrame, filter_df: pd.DataFrame) -> pd.DataFrame:
    if exec_df.empty or filter_df.empty:
        return exec_df.copy()
    left = exec_df.sort_values("open_time").copy()
    right = filter_df.sort_values("open_time").copy()
    merged = pd.merge_asof(
        left,
        right,
        on="open_time",
        direction="backward",
        suffixes=("", "_f"),
    )
    return merged


def validate_timeframe(df: pd.DataFrame, timeframe: str) -> Tuple[bool, Dict[str, object]]:
    if df.empty or len(df) < 2:
        return False, {"reason": "数据不足，无法自检"}
    interval_ms = timeframe_to_minutes(timeframe) * 60_000
    diffs = df["open_time"].diff().dropna().astype(int).tolist()
    if not diffs:
        return False, {"reason": "差值为空"}
    counts = Counter(diffs)
    mode_ms = counts.most_common(1)[0][0]
    expected_ms = interval_ms
    ok_interval = mode_ms == expected_ms

    bars_est = int((df["open_time"].iloc[-1] - df["open_time"].iloc[0]) // interval_ms) + 1
    bars_actual = int(len(df))
    ratio = max(bars_actual / max(bars_est, 1), bars_est / max(bars_actual, 1))
    ok_ratio = ratio <= 10

    ok = ok_interval and ok_ratio
    detail = {
        "mode_ms": int(mode_ms),
        "expected_ms": int(expected_ms),
        "bars_est": int(bars_est),
        "bars_actual": int(bars_actual),
        "ratio": float(ratio),
        "reason": "" if ok else "时间周期自检失败",
    }
    return ok, detail
