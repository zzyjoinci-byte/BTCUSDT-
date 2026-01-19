import pandas as pd

from backtest_engine import run_backtest


def test_beargate_only_affects_short():
    data = [
        {
            "open_time": 0,
            "open": 1.0,
            "high": 1.2,
            "low": 0.9,
            "close": 1.0,
            "volume": 1.0,
            "close_time": 1,
            "rsi": 50,
            "macd_hist": 0.0,
            "ema_fast": 1.0,
            "ema_slow": 1.0,
            "boll_mid": 1.0,
            "boll_upper": 1.2,
            "boll_lower": 0.8,
            "atr": 0.1,
            "adx": 20,
            "adx_f": 10,
            "ema_fast_f": 1.0,
        },
        {
            "open_time": 1,
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 0.9,
            "volume": 1.0,
            "close_time": 2,
            "rsi": 40,
            "macd_hist": -0.1,
            "ema_fast": 0.95,
            "ema_slow": 1.0,
            "boll_mid": 1.0,
            "boll_upper": 1.2,
            "boll_lower": 0.8,
            "atr": 0.1,
            "adx": 20,
            "adx_f": 10,
            "ema_fast_f": 1.0,
        },
        {
            "open_time": 2,
            "open": 1.0,
            "high": 1.3,
            "low": 0.9,
            "close": 1.2,
            "volume": 1.0,
            "close_time": 3,
            "rsi": 60,
            "macd_hist": 0.1,
            "ema_fast": 1.1,
            "ema_slow": 1.0,
            "boll_mid": 1.0,
            "boll_upper": 1.2,
            "boll_lower": 0.8,
            "atr": 0.1,
            "adx": 20,
            "adx_f": 10,
            "ema_fast_f": 1.0,
        },
    ]
    df = pd.DataFrame(data)
    cfg = {
        "initial_capital": 10000,
        "fee_rate": 0.0004,
        "slippage_rate": 0.0001,
        "exec_tf": "4h",
        "filter_tf": "1d",
        "risk": {"risk_per_trade_long": 0.01, "risk_per_trade_short": 0.01, "max_notional_pct_short": 0.5},
        "v5": {
            "ema_slope_lookback": 5,
            "bear_adx_threshold": 25,
            "atr_trail_mult": 2.0,
            "atr_init_mult": 2.0,
            "tp2_invalid_min_pnl_pct_long": 0.002,
            "tp2_invalid_min_pnl_pct_short": 0.008,
            "tp2_invalid_min_hold_bars": 16,
            "tighten_to": "mid",
        },
    }
    trades, equity, result = run_backtest(df, cfg)
    counters = result["counters"]
    assert counters["entries_short"] == 0
    assert counters["entries_long"] >= 1
