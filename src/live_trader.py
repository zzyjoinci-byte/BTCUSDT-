from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List

import pandas as pd

from binance_api import BinanceAPI
from resample import merge_filter_to_exec, resample_ohlcv
from strategy_v5 import build_signals, prepare_exec_frame, prepare_filter_frame


class LiveTrader:
    def __init__(
        self,
        api: BinanceAPI,
        config: Dict[str, object],
        log_cb: Callable[[str], None],
        stop_event,
    ) -> None:
        self.api = api
        self.config = config
        self.log = log_cb
        self.stop_event = stop_event
        self.error_count = 0
        self.last_order_ts = 0.0
        self.last_signal_time: int | None = None
        self.report_path = os.path.join("outputs", "live_report.json")
        self._report: List[Dict[str, object]] = self._load_report()

    def _load_report(self) -> List[Dict[str, object]]:
        if not os.path.exists(self.report_path):
            return []
        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _record(self, payload: Dict[str, object]) -> None:
        payload["ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self._report.append(payload)
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(self._report, f, ensure_ascii=False, indent=2)

    def run(self) -> Dict[str, object]:
        live_cfg = self.config.get("live", {})
        mode = str(live_cfg.get("mode", "DRY-RUN")).upper()
        environment = str(live_cfg.get("environment", "testnet"))
        poll_seconds = int(live_cfg.get("poll_seconds", 30))
        max_notional = float(live_cfg.get("max_notional_usdt", 0))
        max_position = float(live_cfg.get("max_position_usdt", 0))
        cooldown = int(live_cfg.get("cooldown_seconds", 60))
        kill_switch = int(live_cfg.get("kill_switch_max_errors", 3))

        symbol = str(self.config.get("symbol", self.config.get("default_symbol", "BTCUSDT")))
        exec_tf = str(self.config.get("exec_tf", "4h"))
        filter_tf = str(self.config.get("filter_tf", "1d"))
        trade_mode = str(self.config.get("trade_mode", "both"))
        allow_long = trade_mode in ("both", "long_only")
        allow_short = trade_mode in ("both", "short_only")
        v5 = self.config.get("v5", {})

        self.log(f"实盘启动: {symbol} {exec_tf} | 模式={mode} | 环境={environment}")
        self._record(
            {
                "action": "start",
                "symbol": symbol,
                "exec_tf": exec_tf,
                "environment": environment,
                "mode": mode,
            }
        )

        while not self.stop_event.is_set():
            try:
                self._loop_once(
                    symbol,
                    exec_tf,
                    filter_tf,
                    v5,
                    allow_long,
                    allow_short,
                    mode,
                    max_notional,
                    max_position,
                    cooldown,
                )
                self.error_count = 0
            except Exception as exc:  # noqa: BLE001
                self.error_count += 1
                self.log(f"实盘错误({self.error_count}): {exc}")
                self._record({"action": "error", "symbol": symbol, "error": str(exc)})
                if self.error_count >= kill_switch:
                    self.log("连续错误触发 kill switch，已停止")
                    self._record({"action": "kill_switch", "symbol": symbol})
                    break
            if self.stop_event.wait(poll_seconds):
                break

        self.log("实盘停止")
        self._record({"action": "stop", "symbol": symbol})
        return {"stopped": True}

    def _loop_once(
        self,
        symbol: str,
        exec_tf: str,
        filter_tf: str,
        v5: Dict[str, object],
        allow_long: bool,
        allow_short: bool,
        mode: str,
        max_notional: float,
        max_position: float,
        cooldown: int,
    ) -> None:
        klines = self.api.fetch_klines_latest(symbol, exec_tf, limit=300)
        if not klines:
            self.log("实盘: 未获取到K线")
            return
        exec_df = pd.DataFrame(klines)
        exec_df = prepare_exec_frame(exec_df, v5, exec_tf)
        if filter_tf != exec_tf:
            filter_df = resample_ohlcv(exec_df, filter_tf)
        else:
            filter_df = exec_df.copy()
        filter_df = prepare_filter_frame(filter_df, v5)
        merged = merge_filter_to_exec(exec_df, filter_df)
        merged = build_signals(merged, v5)
        if merged.empty:
            self.log("实盘: 数据不足，跳过")
            return
        last_row = merged.iloc[-1]
        if pd.isna(last_row.get("close")):
            self.log("实盘: 最新K线无效，跳过")
            return

        long_signal = bool(last_row.get("long_signal", False))
        short_signal = bool(last_row.get("short_signal", False))
        price = float(last_row["close"])
        open_time = int(last_row.get("open_time", 0))

        if self.last_signal_time != open_time:
            self.last_signal_time = open_time
            self.log(f"信号: time={open_time} long={long_signal} short={short_signal} price={price:.4f}")
            self._record(
                {
                    "action": "signal",
                    "symbol": symbol,
                    "open_time": open_time,
                    "long": long_signal,
                    "short": short_signal,
                    "price": price,
                }
            )

        pos = self.api.get_position(symbol)
        position_amt = float(pos.get("position_amt", 0.0))
        position_notional = abs(float(pos.get("notional", 0.0)))
        has_position = abs(position_amt) > 0

        now = time.time()
        if now - self.last_order_ts < cooldown:
            return

        if not has_position:
            if long_signal and allow_long:
                self._open_position(symbol, "BUY", price, mode, max_notional, max_position, position_notional)
            elif short_signal and allow_short:
                self._open_position(symbol, "SELL", price, mode, max_notional, max_position, position_notional)
        else:
            if position_amt > 0 and short_signal and allow_short:
                self._close_position(symbol, side="SELL", qty=abs(position_amt), mode=mode)
            elif position_amt < 0 and long_signal and allow_long:
                self._close_position(symbol, side="BUY", qty=abs(position_amt), mode=mode)

    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        mode: str,
        max_notional: float,
        max_position: float,
        position_notional: float,
    ) -> None:
        if max_notional <= 0:
            self.log("实盘: max_notional_usdt 未配置，跳过下单")
            return
        if max_position > 0 and position_notional + max_notional > max_position:
            self.log("实盘: 超过最大持仓名义金额，跳过")
            return
        qty = max_notional / price if price > 0 else 0.0
        if qty <= 0:
            self.log("实盘: 计算数量无效，跳过")
            return

        if mode != "LIVE":
            self.log(f"DRY-RUN: 开仓 {side} qty={qty:.6f} price={price:.4f}")
            self._record(
                {
                    "action": "open",
                    "mode": mode,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "result": "dry_run",
                }
            )
            self.last_order_ts = time.time()
            return

        result = self.api.place_order(symbol, side=side, quantity=qty, order_type="MARKET")
        self.last_order_ts = time.time()
        self.log(f"开仓成功: {side} qty={qty:.6f} orderId={result.get('orderId')}")
        self._record(
            {
                "action": "open",
                "mode": mode,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "result": result,
            }
        )

    def _close_position(self, symbol: str, side: str, qty: float, mode: str) -> None:
        if qty <= 0:
            return
        if mode != "LIVE":
            self.log(f"DRY-RUN: 平仓 {side} reduceOnly qty={qty:.6f}")
            self._record(
                {
                    "action": "close",
                    "mode": mode,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "result": "dry_run",
                }
            )
            self.last_order_ts = time.time()
            return
        result = self.api.place_order(symbol, side=side, quantity=qty, order_type="MARKET", reduce_only=True)
        self.last_order_ts = time.time()
        self.log(f"平仓成功: {side} qty={qty:.6f} orderId={result.get('orderId')}")
        self._record(
            {
                "action": "close",
                "mode": mode,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "result": result,
            }
        )
