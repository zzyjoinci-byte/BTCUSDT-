from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from strategy_v5 import build_signals, calculate_position_size, tp2_invalid_should_tighten


@dataclass
class Position:
    side: str
    entry_price: float
    qty: float
    stop_price: float
    entry_time: int
    hold_bars: int = 0
    tp2_tightened: bool = False
    pending_stop: Optional[float] = None


def run_backtest(
    df: pd.DataFrame,
    config: Dict[str, object],
    progress_cb: Optional[Callable[[int, int], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    df = build_signals(df)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"error": "数据为空"}

    v5 = config["v5"]
    risk = config["risk"]
    fee_rate = float(config["fee_rate"])
    slippage = float(config["slippage_rate"])
    initial_capital = float(config["initial_capital"])
    trade_mode = config.get("trade_mode", "both")
    allow_long = trade_mode in ("both", "long_only")
    allow_short = trade_mode in ("both", "short_only")

    range_filter = df["adx"] < 15
    if "adx_f" in df.columns and "ema_fast_f" in df.columns:
        slope = df["ema_fast_f"] - df["ema_fast_f"].shift(v5["ema_slope_lookback"])
        bear_pass = (df["adx_f"] >= v5["bear_adx_threshold"]) & (slope < 0)
    else:
        bear_pass = pd.Series([False] * len(df))

    counters: Dict[str, int] = {
        "d_allow_long_bars": 0,
        "d_allow_short_bars": 0,
        "d_range_blocked_bars": 0,
        "h_touch_long_events": 0,
        "h_touch_short_events": 0,
        "entries_long": 0,
        "entries_short": 0,
        "entries_total": 0,
        "candidates_long": 0,
        "candidates_short": 0,
        "tp2_invalid_blocked_by_loss": 0,
        "tp2_invalid_blocked_by_hold": 0,
        "tp2_invalid_blocked_by_profit_gate": 0,
        "tp2_invalid_tighten_count": 0,
        "tp2_invalid_triggered_exit_count": 0,
        "beargate_pass": 0,
        "beargate_fail": 0,
        "beargate_short_blocked": 0,
    }

    cash = initial_capital
    position: Optional[Position] = None
    trades: List[Dict[str, object]] = []
    equity_curve: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        if stop_flag and stop_flag():
            break

        if range_filter.iloc[i]:
            counters["d_range_blocked_bars"] += 1
        else:
            counters["d_allow_long_bars"] += 1
            if bool(bear_pass.iloc[i]):
                counters["d_allow_short_bars"] += 1

        if bool(row["touch_long"]):
            counters["h_touch_long_events"] += 1
        if bool(row["touch_short"]):
            counters["h_touch_short_events"] += 1

        if bool(bear_pass.iloc[i]):
            counters["beargate_pass"] += 1
        else:
            counters["beargate_fail"] += 1

        prev_row = df.iloc[i - 1] if i > 0 else None

        if position:
            position.hold_bars += 1
            if position.pending_stop is not None:
                position.stop_price = position.pending_stop
                position.pending_stop = None

            if prev_row is not None and pd.notna(prev_row["atr"]):
                if position.side == "LONG":
                    trail = prev_row["close"] - prev_row["atr"] * v5["atr_trail_mult"]
                    if trail > position.stop_price:
                        position.stop_price = trail
                else:
                    trail = prev_row["close"] + prev_row["atr"] * v5["atr_trail_mult"]
                    if trail < position.stop_price:
                        position.stop_price = trail

            exit_reason = None
            exit_price = None
            if position.side == "LONG" and row["low"] <= position.stop_price:
                exit_price = position.stop_price * (1 - slippage)
                exit_reason = "stop"
            if position.side == "SHORT" and row["high"] >= position.stop_price:
                exit_price = position.stop_price * (1 + slippage)
                exit_reason = "stop"

            if exit_reason and exit_price is not None:
                if position.side == "LONG":
                    pnl = (exit_price - position.entry_price) * position.qty
                else:
                    pnl = (position.entry_price - exit_price) * position.qty
                fee = exit_price * position.qty * fee_rate
                cash += pnl - fee
                realized = pnl - fee
                stop_type = "profit_stop" if realized > 0 else "loss_stop"
                if exit_reason == "stop":
                    exit_reason = "TrailStop" if realized > 0 else "Stop"
                trades.append(
                    {
                        "entry_time": position.entry_time,
                        "exit_time": int(row["open_time"]),
                        "side": position.side,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "qty": position.qty,
                        "pnl": realized,
                        "reason": exit_reason,
                        "stop_type": stop_type,
                        "hold_bars": position.hold_bars,
                    }
                )
                if position.tp2_tightened:
                    counters["tp2_invalid_triggered_exit_count"] += 1
                position = None

        if position is None:
            if bool(row["long_signal"]):
                counters["candidates_long"] += 1
                if allow_long and not range_filter.iloc[i]:
                    entry_price = row["close"] * (1 + slippage)
                    recent_low = df.loc[max(0, i - 4) : i, "low"].min()
                    atr_stop = entry_price - row["atr"] * v5["atr_init_mult"]
                    structure_stop = recent_low
                    stop_price = max(atr_stop, structure_stop)
                    qty = calculate_position_size(
                        cash,
                        risk["risk_per_trade_long"],
                        entry_price,
                        stop_price,
                        None,
                    )
                    if qty > 0:
                        fee = entry_price * qty * fee_rate
                        cash -= fee
                        position = Position(
                            side="LONG",
                            entry_price=entry_price,
                            qty=qty,
                            stop_price=stop_price,
                            entry_time=int(row["open_time"]),
                        )
                        counters["entries_long"] += 1
                        counters["entries_total"] += 1

            if position is None and bool(row["short_signal"]):
                counters["candidates_short"] += 1
                if allow_short and not range_filter.iloc[i]:
                    if not bool(bear_pass.iloc[i]):
                        counters["beargate_short_blocked"] += 1
                    else:
                        entry_price = row["close"] * (1 - slippage)
                        recent_high = df.loc[max(0, i - 4) : i, "high"].max()
                        atr_stop = entry_price + row["atr"] * v5["atr_init_mult"]
                        structure_stop = recent_high
                        stop_price = min(atr_stop, structure_stop)
                        qty = calculate_position_size(
                            cash,
                            risk["risk_per_trade_short"],
                            entry_price,
                            stop_price,
                            risk["max_notional_pct_short"],
                        )
                        if qty > 0:
                            fee = entry_price * qty * fee_rate
                            cash -= fee
                            position = Position(
                                side="SHORT",
                                entry_price=entry_price,
                                qty=qty,
                                stop_price=stop_price,
                                entry_time=int(row["open_time"]),
                            )
                            counters["entries_short"] += 1
                            counters["entries_total"] += 1

        if position:
            if i > 0:
                if position.side == "LONG":
                    pnl_pct = (row["close"] - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - row["close"]) / position.entry_price

                trigger = False
                if position.side == "LONG":
                    trigger = row["close"] < row["boll_mid"] and prev_row is not None and prev_row["close"] >= prev_row["boll_mid"]
                else:
                    trigger = row["close"] > row["boll_mid"] and prev_row is not None and prev_row["close"] <= prev_row["boll_mid"]

                if trigger:
                    allow, reason = tp2_invalid_should_tighten(
                        pnl_pct,
                        position.hold_bars,
                        v5["tp2_invalid_min_pnl_pct_long"] if position.side == "LONG" else v5["tp2_invalid_min_pnl_pct_short"],
                        v5["tp2_invalid_min_hold_bars"],
                    )
                    if not allow:
                        if reason == "loss":
                            counters["tp2_invalid_blocked_by_loss"] += 1
                        elif reason == "hold_gate":
                            counters["tp2_invalid_blocked_by_hold"] += 1
                        else:
                            counters["tp2_invalid_blocked_by_profit_gate"] += 1
                    else:
                        if v5.get("tp2_invalid_action", "tighten_stop") == "tighten_stop":
                            tighten_to = v5["tighten_to"]
                            if tighten_to == "atr_trail" and prev_row is not None:
                                if position.side == "LONG":
                                    candidate = prev_row["close"] - prev_row["atr"] * v5["atr_trail_mult"]
                                else:
                                    candidate = prev_row["close"] + prev_row["atr"] * v5["atr_trail_mult"]
                            else:
                                candidate = row["boll_mid"]
                            if position.side == "LONG":
                                new_stop = max(position.stop_price, candidate)
                            else:
                                new_stop = min(position.stop_price, candidate)
                            position.pending_stop = new_stop
                            position.tp2_tightened = True
                            counters["tp2_invalid_tighten_count"] += 1

        equity = cash
        if position:
            if position.side == "LONG":
                equity += (row["close"] - position.entry_price) * position.qty
            else:
                equity += (position.entry_price - row["close"]) * position.qty
        equity_curve.append({"open_time": int(row["open_time"]), "equity": float(equity)})

        if progress_cb:
            progress_cb(i + 1, len(df))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    result = {"counters": counters}
    return trades_df, equity_df, result
