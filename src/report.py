from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from resample import timeframe_to_minutes


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max.replace(0, np.nan)
    return float(drawdown.min())


def compute_sharpe(equity: pd.Series, timeframe: str) -> float:
    if equity.empty or len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    minutes = timeframe_to_minutes(timeframe)
    periods_per_year = (365 * 24 * 60) / minutes
    return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))


def _stop_stats(trades: pd.DataFrame, stop_type: str, bar_hours: float) -> Dict[str, float]:
    if trades.empty:
        return {
            "count": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "hold_hours_p25": 0.0,
            "hold_hours_p50": 0.0,
            "hold_hours_p75": 0.0,
        }
    if "stop_type" in trades.columns:
        subset = trades.loc[trades["stop_type"] == stop_type]
    else:
        reason_map = {"loss_stop": "Stop", "profit_stop": "TrailStop"}
        subset = trades.loc[trades["reason"] == reason_map.get(stop_type, "")]
    if subset.empty:
        return {
            "count": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "win_rate": 0.0,
            "hold_hours_p25": 0.0,
            "hold_hours_p50": 0.0,
            "hold_hours_p75": 0.0,
        }
    hold_hours = subset["hold_bars"] * bar_hours
    quantiles = hold_hours.quantile([0.25, 0.5, 0.75]).to_dict()
    total_pnl = float(subset["pnl"].sum())
    count = int(len(subset))
    return {
        "count": count,
        "total_pnl": total_pnl,
        "avg_pnl": float(total_pnl / max(count, 1)),
        "win_rate": float((subset["pnl"] > 0).mean()),
        "hold_hours_p25": float(quantiles.get(0.25, 0.0)),
        "hold_hours_p50": float(quantiles.get(0.5, 0.0)),
        "hold_hours_p75": float(quantiles.get(0.75, 0.0)),
    }


def summarize(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    counters: Dict[str, int],
    config: Dict[str, object],
    symbol: str,
) -> Dict[str, object]:
    if equity.empty:
        total_return = 0.0
        mdd = 0.0
        sharpe = 0.0
    else:
        total_return = float(equity["equity"].iloc[-1] / config["initial_capital"] - 1)
        mdd = compute_max_drawdown(equity["equity"])
        sharpe = compute_sharpe(equity["equity"], config["exec_tf"])

    trades_count = int(len(trades))
    win_rate = float((trades["pnl"] > 0).mean()) if trades_count > 0 else 0.0
    profit = trades.loc[trades["pnl"] > 0, "pnl"].sum() if trades_count > 0 else 0.0
    loss = trades.loc[trades["pnl"] <= 0, "pnl"].sum() if trades_count > 0 else 0.0
    profit_factor = float(profit / abs(loss)) if loss != 0 else 0.0
    tp2_rate = float((trades["reason"] == "tp2_tighten_stop").mean()) if trades_count > 0 else 0.0

    exit_reason = []
    exit_reason_pnl = []
    if trades_count > 0:
        grouped = trades.groupby("reason")["pnl"].agg(["count", "sum"]).sort_values("count", ascending=False)
        for reason, row in grouped.head(5).iterrows():
            exit_reason.append({"reason": reason, "count": int(row["count"]), "total_pnl": float(row["sum"])})
        for reason, row in grouped.iterrows():
            exit_reason_pnl.append(
                {
                    "reason": reason,
                    "count": int(row["count"]),
                    "total_pnl": float(row["sum"]),
                    "avg_pnl": float(row["sum"] / max(row["count"], 1)),
                }
            )

    bar_hours = timeframe_to_minutes(config["exec_tf"]) / 60
    stop_breakdown = {
        "loss_stop": _stop_stats(trades, "loss_stop", bar_hours),
        "profit_stop": _stop_stats(trades, "profit_stop", bar_hours),
    }

    side_breakdown = []
    if trades_count > 0:
        for side, group in trades.groupby("side"):
            side_profit = group.loc[group["pnl"] > 0, "pnl"].sum()
            side_loss = group.loc[group["pnl"] <= 0, "pnl"].sum()
            side_pf = float(side_profit / abs(side_loss)) if side_loss != 0 else 0.0
            side_breakdown.append(
                {
                    "side": side,
                    "count": int(len(group)),
                    "total_pnl": float(group["pnl"].sum()),
                    "pf": side_pf,
                }
            )

    report = {
        "symbol": symbol,
        "total_return": total_return,
        "mdd": mdd,
        "sharpe": sharpe,
        "trades": trades_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "tp2_rate": tp2_rate,
        "exit_reason_top5": exit_reason,
        "exit_reason_pnl": exit_reason_pnl,
        "stop_breakdown": stop_breakdown,
        "side_breakdown": side_breakdown,
        "counters": counters,
        "config": {
            "exec_tf": config["exec_tf"],
            "filter_tf": config["filter_tf"],
            "fee_rate": config["fee_rate"],
            "slippage_rate": config["slippage_rate"],
        },
    }
    return report


def export_results(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    report: Dict[str, object],
    output_prefix: str,
) -> Tuple[str, str, str, str]:
    trades_path = f"{output_prefix}_trades.csv"
    equity_path = f"{output_prefix}_equity.csv"
    report_path = f"{output_prefix}_report.json"
    png_path = f"{output_prefix}_equity.png"

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_equity_curve(equity, png_path)
    return trades_path, equity_path, report_path, png_path


def plot_equity_curve(equity: pd.DataFrame, output_path: str) -> None:
    if equity.empty:
        return
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(equity["equity"].values, linewidth=1.2)
    plt.title("资金曲线")
    plt.xlabel("Bar")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
