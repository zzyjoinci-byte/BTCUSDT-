from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd
from PySide6.QtCore import QDate, QObject, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from backtest_engine import run_backtest
from binance_api import BinanceAPI, safe_api_call
from data_store import (
    calc_missing_segments,
    calc_overlap_segments,
    estimate_bars,
    ensure_schema,
    fetch_open_times,
    load_available_range,
    load_klines_df,
    normalize_segments,
    open_db,
    timeframe_ms,
    upsert_klines,
)
from report import export_results, summarize
from resample import merge_filter_to_exec, resample_ohlcv, validate_timeframe
from state import AppState
from strategy_v5 import prepare_exec_frame, prepare_filter_frame


class ApiTestWorker(QObject):
    finished = Signal(dict)

    def __init__(self, api_key: str, api_secret: str, testnet: bool) -> None:
        super().__init__()
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    def run(self) -> None:
        api = BinanceAPI(self.api_key, self.api_secret, self.testnet)
        payload, error = safe_api_call(api.test_connection)
        if error:
            self.finished.emit({"ok": False, "error": error})
        else:
            payload["ok"] = True
            self.finished.emit(payload)


class BacktestWorker(QObject):
    finished = Signal(dict)

    def __init__(self, config: Dict[str, object], api_key: str, api_secret: str, testnet: bool, state: AppState) -> None:
        super().__init__()
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.state = state
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cfg = self.config
        symbol = cfg["symbol"]
        exec_tf = cfg["exec_tf"]
        filter_tf = cfg["filter_tf"]
        start_ms = int(datetime.strptime(cfg["start_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.strptime(cfg["end_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        interval_ms = timeframe_ms(exec_tf)

        self.state.set_phase("初始化")
        self.state.set_status("准备回测...")
        self.state.set_progress(0)

        api = BinanceAPI(self.api_key, self.api_secret, self.testnet)
        conn = open_db(os.path.join("data", "market.sqlite"))
        ensure_schema(conn)

        self.state.set_phase("检查/维护本地数据")
        self.state.log(f"检查本地缓存: {symbol} {exec_tf}")
        open_times = fetch_open_times(conn, "binance", "usdtm", symbol, exec_tf, start_ms, end_ms)
        missing_segments = calc_missing_segments(start_ms, end_ms, interval_ms, open_times)
        overlap_segments = normalize_segments(calc_overlap_segments(missing_segments, interval_ms), start_ms, end_ms)
        bars_est = estimate_bars(start_ms, end_ms, interval_ms)
        self.state.log(f"预计总 bar 数: {bars_est}，缺口段: {len(missing_segments)}")
        self.state.update_detail(
            {
                "bars_est": bars_est,
                "missing_segments": len(missing_segments),
                "open_times": len(open_times),
            }
        )

        downloaded = 0
        inserted = 0
        updated = 0
        skipped = 0

        self.state.set_phase("拉取K线")
        for seg_start, seg_end in overlap_segments:
            if self._stop_event.is_set():
                self.state.set_status("已停止")
                self.finished.emit({"stopped": True})
                return
            segment_base = downloaded
            def _progress(count):
                nonlocal downloaded
                downloaded = segment_base + count
                percent = int(min(100, downloaded / max(bars_est, 1) * 100))
                self.state.set_progress(percent)
                self.state.update_detail(
                    {
                        "bars_est": bars_est,
                        "downloaded": downloaded,
                        "inserted": inserted,
                        "updated": updated,
                        "skipped": skipped,
                    }
                )

            data, error = safe_api_call(api.fetch_klines, symbol, exec_tf, seg_start, seg_end, _progress)
            if error:
                self.state.error(f"拉取K线失败: {error}")
                self.finished.emit({"error": error})
                return

            if data:
                self.state.set_phase("写库/去重/修正")
                result = upsert_klines(conn, "binance", "usdtm", symbol, exec_tf, data)
                inserted += result.inserted
                updated += result.updated
                skipped += result.skipped
                self.state.log(
                    f"写库完成: inserted={result.inserted}, updated={result.updated}, skipped={result.skipped}"
                )
                self.state.update_detail(
                    {
                        "bars_est": bars_est,
                        "downloaded": downloaded,
                        "inserted": inserted,
                        "updated": updated,
                        "skipped": skipped,
                    }
                )

        avail_start, avail_end = load_available_range(conn, "binance", "usdtm", symbol, exec_tf)
        if avail_start is None or avail_end is None:
            self.state.error("本地没有可用数据，请先拉取K线")
            self.finished.emit({"error": "no_data"})
            return
        if start_ms < avail_start or end_ms > avail_end:
            start_ms = max(start_ms, avail_start)
            end_ms = min(end_ms, avail_end)
            self.state.log("回测区间已按可用数据自动截断")
            bars_est = estimate_bars(start_ms, end_ms, interval_ms)

        self.state.set_phase("加载数据")
        exec_df = load_klines_df(conn, "binance", "usdtm", symbol, exec_tf, start_ms, end_ms)
        if exec_df.empty:
            self.state.error("K线为空，无法回测")
            self.finished.emit({"error": "empty"})
            return
        bars_actual = len(exec_df)
        ratio = max(bars_actual / max(bars_est, 1), bars_est / max(bars_actual, 1))
        self.state.update_detail({"bars_est": bars_est, "bars_actual": bars_actual, "ratio": ratio})
        if ratio > 10:
            self.state.error(f"bars_est/实际异常: bars_est={bars_est}, actual={bars_actual}")
            self.finished.emit({"error": "bars_ratio"})
            return

        ok, detail = validate_timeframe(exec_df, exec_tf)
        if not ok:
            self.state.error(f"时间周期自检失败: {detail}")
            self.finished.emit({"error": "timeframe"})
            return
        self.state.log(f"时间周期自检通过: {detail}")

        exec_df = prepare_exec_frame(exec_df, cfg["v5"], exec_tf)
        if filter_tf != exec_tf:
            filter_df = resample_ohlcv(exec_df, filter_tf)
        else:
            filter_df = exec_df.copy()
        filter_df = prepare_filter_frame(filter_df, cfg["v5"])
        merged = merge_filter_to_exec(exec_df, filter_df)

        self.state.set_phase("回测")
        self.state.log("开始回测")
        def _bt_progress(done, total):
            percent = int(done / max(total, 1) * 100)
            self.state.set_progress(percent)
            self.state.update_detail({"processed": done, "total": total})

        trades_df, equity_df, result = run_backtest(
            merged,
            cfg,
            progress_cb=_bt_progress,
            stop_flag=self._stop_event.is_set,
        )
        if self._stop_event.is_set():
            self.state.set_status("已停止")
            self.finished.emit({"stopped": True})
            return

        self.state.set_phase("生成报表/绘图")
        self.state.log("生成报表与图表")
        report = summarize(
            trades_df,
            equity_df,
            result.get("counters", {}),
            cfg,
            symbol,
            result.get("adx_4h_quantiles"),
        )
        output_prefix = os.path.join("outputs", symbol)
        export_paths = export_results(trades_df, equity_df, report, output_prefix)

        self.state.set_phase("完成")
        self.state.set_progress(100)
        self.state.set_status("回测完成")
        self.finished.emit(
            {
                "symbol": symbol,
                "trades": trades_df,
                "equity": equity_df,
                "report": report,
                "paths": export_paths,
            }
        )


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Binance USDT-M 永续合约回测工具")
        self.state = AppState()
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[BacktestWorker] = None
        self.api_thread: Optional[QThread] = None
        self.last_export_paths = None
        self.spinner_frames = ["|", "/", "-", "\\"]
        self.spinner_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self._spin)
        self._build_ui()
        self._bind_state()
        self._load_default_config()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        self.connection_group = QGroupBox("连接")
        conn_layout = QGridLayout()
        self.api_key_input = QLineEdit()
        self.api_secret_input = QLineEdit()
        self.api_secret_input.setEchoMode(QLineEdit.Password)
        self.env_combo = QComboBox()
        self.env_combo.addItems(["Mainnet", "Testnet"])
        self.test_btn = QPushButton("测试连接")
        self.conn_status = QLabel("未测试")
        self.conn_info = QLabel("-")
        conn_layout.addWidget(QLabel("API Key"), 0, 0)
        conn_layout.addWidget(self.api_key_input, 0, 1)
        conn_layout.addWidget(QLabel("Secret"), 1, 0)
        conn_layout.addWidget(self.api_secret_input, 1, 1)
        conn_layout.addWidget(QLabel("环境"), 2, 0)
        conn_layout.addWidget(self.env_combo, 2, 1)
        conn_layout.addWidget(self.test_btn, 3, 0)
        conn_layout.addWidget(self.conn_status, 3, 1)
        conn_layout.addWidget(self.conn_info, 4, 0, 1, 2)
        self.connection_group.setLayout(conn_layout)

        self.params_group = QGroupBox("回测参数")
        params_layout = QGridLayout()
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTCUSDT", "SOLUSDT"])
        self.trade_mode = QComboBox()
        self.trade_mode.addItems(["多空都做", "只做多", "只做空"])
        self.start_date = QDateEdit()
        self.end_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.end_date.setCalendarPopup(True)
        tf_items = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.exec_tf_combo = QComboBox()
        self.exec_tf_combo.addItems(tf_items)
        self.filter_tf_combo = QComboBox()
        self.filter_tf_combo.addItems(tf_items)

        self.initial_capital = self._make_spin(100, 1_000_000, 100, 0, 10000)
        self.fee_rate = self._make_spin(0, 0.01, 0.0001, 6, 0.0004)
        self.slippage_rate = self._make_spin(0, 0.01, 0.0001, 6, 0.0003)
        self.risk_long = self._make_spin(0, 0.05, 0.0005, 6, 0.006)
        self.risk_short = self._make_spin(0, 0.05, 0.0005, 6, 0.004)
        self.max_notional_short = self._make_spin(0, 1.0, 0.01, 4, 0.35)

        self.run_btn = QPushButton("运行")
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("保存配置")
        self.load_btn = QPushButton("加载配置")

        params_layout.addWidget(QLabel("Symbol"), 0, 0)
        params_layout.addWidget(self.symbol_combo, 0, 1)
        params_layout.addWidget(QLabel("交易方向"), 1, 0)
        params_layout.addWidget(self.trade_mode, 1, 1)
        params_layout.addWidget(QLabel("开始日期"), 2, 0)
        params_layout.addWidget(self.start_date, 2, 1)
        params_layout.addWidget(QLabel("结束日期"), 3, 0)
        params_layout.addWidget(self.end_date, 3, 1)
        params_layout.addWidget(QLabel("执行周期"), 4, 0)
        params_layout.addWidget(self.exec_tf_combo, 4, 1)
        params_layout.addWidget(QLabel("过滤周期"), 5, 0)
        params_layout.addWidget(self.filter_tf_combo, 5, 1)
        params_layout.addWidget(QLabel("初始资金"), 6, 0)
        params_layout.addWidget(self.initial_capital, 6, 1)
        params_layout.addWidget(QLabel("手续费率"), 7, 0)
        params_layout.addWidget(self.fee_rate, 7, 1)
        params_layout.addWidget(QLabel("滑点率"), 8, 0)
        params_layout.addWidget(self.slippage_rate, 8, 1)
        params_layout.addWidget(QLabel("多头风险"), 9, 0)
        params_layout.addWidget(self.risk_long, 9, 1)
        params_layout.addWidget(QLabel("空头风险"), 10, 0)
        params_layout.addWidget(self.risk_short, 10, 1)
        params_layout.addWidget(QLabel("空头名义上限"), 11, 0)
        params_layout.addWidget(self.max_notional_short, 11, 1)
        params_layout.addWidget(self.run_btn, 12, 0)
        params_layout.addWidget(self.stop_btn, 12, 1)
        params_layout.addWidget(self.save_btn, 13, 0)
        params_layout.addWidget(self.load_btn, 13, 1)

        v5_box = QGroupBox("v5 参数")
        v5_layout = QGridLayout()
        self.rsi_len = self._make_spin(2, 50, 1, 0, 14)
        self.macd_fast = self._make_spin(2, 50, 1, 0, 12)
        self.macd_slow = self._make_spin(5, 100, 1, 0, 26)
        self.macd_signal = self._make_spin(2, 50, 1, 0, 9)
        self.boll_len = self._make_spin(2, 60, 1, 0, 20)
        self.boll_std = self._make_spin(0.5, 5.0, 0.1, 2, 2.0)
        self.ema_fast = self._make_spin(5, 200, 1, 0, 50)
        self.ema_slow = self._make_spin(10, 400, 1, 0, 200)
        self.adx_len = self._make_spin(5, 50, 1, 0, 14)
        self.bear_adx = self._make_spin(5, 50, 1, 0, 25)
        self.ema_slope = self._make_spin(1, 20, 1, 0, 5)
        self.atr_len = self._make_spin(5, 50, 1, 0, 14)
        self.atr_init = self._make_spin(0.5, 10.0, 0.1, 2, 2.6)
        self.atr_trail = self._make_spin(0.5, 10.0, 0.1, 2, 2.8)
        self.tp2_long = self._make_spin(0, 0.02, 0.0005, 6, 0.002)
        self.tp2_short = self._make_spin(0, 0.05, 0.0005, 6, 0.008)
        self.tp2_hold = self._make_spin(1, 200, 1, 0, 16)
        self.tp2_action = QComboBox()
        self.tp2_action.addItems(["tighten_stop"])
        self.tighten_to = QComboBox()
        self.tighten_to.addItems(["mid", "atr_trail"])
        self.entry_adx_enabled = QCheckBox()
        self.entry_adx_period = self._make_spin(2, 50, 1, 0, 14)
        self.entry_adx_min_long = self._make_spin(0, 100, 1, 0, 20)
        self.entry_adx_min_short = self._make_spin(0, 100, 1, 0, 25)

        v5_items = [
            ("RSI", self.rsi_len),
            ("MACD fast", self.macd_fast),
            ("MACD slow", self.macd_slow),
            ("MACD signal", self.macd_signal),
            ("BOLL len", self.boll_len),
            ("BOLL std", self.boll_std),
            ("EMA50", self.ema_fast),
            ("EMA200", self.ema_slow),
            ("ADX", self.adx_len),
            ("Bear ADX", self.bear_adx),
            ("EMA slope", self.ema_slope),
            ("ATR len", self.atr_len),
            ("ATR init", self.atr_init),
            ("ATR trail", self.atr_trail),
            ("TP2 long", self.tp2_long),
            ("TP2 short", self.tp2_short),
            ("TP2 hold", self.tp2_hold),
            ("TP2 action", self.tp2_action),
            ("Tighten to", self.tighten_to),
            ("ADX入场过滤", self.entry_adx_enabled),
            ("ADX周期", self.entry_adx_period),
            ("ADX多头阈值", self.entry_adx_min_long),
            ("ADX空头阈值", self.entry_adx_min_short),
        ]
        for idx, (label, widget) in enumerate(v5_items):
            v5_layout.addWidget(QLabel(label), idx, 0)
            v5_layout.addWidget(widget, idx, 1)
        v5_box.setLayout(v5_layout)

        params_container = QVBoxLayout()
        params_container.addLayout(params_layout)
        params_container.addWidget(v5_box)
        self.params_group.setLayout(params_container)

        self.progress_group = QGroupBox("进度/状态")
        progress_layout = QVBoxLayout()
        self.phase_label = QLabel("初始化")
        self.spinner_label = QLabel("o")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.detail_label = QLabel("-")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        progress_layout.addWidget(self.phase_label)
        progress_layout.addWidget(self.spinner_label)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.detail_label)
        progress_layout.addWidget(self.log_box)
        self.progress_group.setLayout(progress_layout)

        self.results_group = QGroupBox("结果")
        results_layout = QVBoxLayout()
        self.metrics_label = QLabel("等待回测")
        self.metrics_label.setWordWrap(True)
        self.counters_label = QLabel("-")
        self.counters_label.setWordWrap(True)
        self.stop_breakdown_label = QLabel("-")
        self.stop_breakdown_label.setWordWrap(True)

        self.exit_table = QTableWidget(0, 3)
        self.exit_table.setHorizontalHeaderLabels(["原因", "次数", "总PnL"])
        self.side_table = QTableWidget(0, 4)
        self.side_table.setHorizontalHeaderLabels(["方向", "次数", "总PnL", "PF"])

        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)

        self.trades_table = QTableWidget(0, 8)
        self.trades_table.setHorizontalHeaderLabels(
            ["Entry", "Exit", "Side", "EntryPx", "ExitPx", "Qty", "PnL", "Reason"]
        )

        self.export_btn = QPushButton("导出结果")
        results_layout.addWidget(self.metrics_label)
        results_layout.addWidget(self.counters_label)
        results_layout.addWidget(self.stop_breakdown_label)
        results_layout.addWidget(QLabel("ExitReason Top5"))
        results_layout.addWidget(self.exit_table)
        results_layout.addWidget(QLabel("Side Breakdown"))
        results_layout.addWidget(self.side_table)
        results_layout.addWidget(self.canvas)
        results_layout.addWidget(QLabel("交易记录（前 N 笔）"))
        results_layout.addWidget(self.trades_table)
        results_layout.addWidget(self.export_btn)
        self.results_group.setLayout(results_layout)

        content_layout.addWidget(self.connection_group)
        content_layout.addWidget(self.params_group)
        content_layout.addWidget(self.progress_group)
        content_layout.addWidget(self.results_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        self.test_btn.clicked.connect(self._on_test_connection)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.save_btn.clicked.connect(self._on_save_config)
        self.load_btn.clicked.connect(self._on_load_config)
        self.symbol_combo.currentTextChanged.connect(self._on_symbol_change)
        self.export_btn.clicked.connect(self._on_export_results)

    def _bind_state(self) -> None:
        self.state.log_signal.connect(self._append_log)
        self.state.status_signal.connect(self._set_status)
        self.state.phase_signal.connect(self._set_phase)
        self.state.progress_signal.connect(self.progress_bar.setValue)
        self.state.detail_signal.connect(self._set_detail)
        self.state.result_signal.connect(self._render_results)
        self.state.error_signal.connect(self._on_error)

    def _make_spin(self, min_val, max_val, step, decimals, value) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        return spin

    def _load_default_config(self) -> None:
        path = os.path.join("config", "default_config.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.symbol_combo.setCurrentText(cfg["default_symbol"])
        self.trade_mode.setCurrentText("多空都做" if cfg.get("trade_mode", "both") == "both" else "只做多" if cfg.get("trade_mode") == "long_only" else "只做空")
        self.start_date.setDate(QDate.fromString(cfg["start_date"], "yyyy-MM-dd"))
        self.end_date.setDate(QDate.fromString(cfg["end_date"], "yyyy-MM-dd"))
        self.exec_tf_combo.setCurrentText(cfg["exec_tf"])
        self.filter_tf_combo.setCurrentText(cfg["filter_tf"])
        self.initial_capital.setValue(cfg["initial_capital"])
        self.fee_rate.setValue(cfg["fee_rate"])
        self.slippage_rate.setValue(cfg["slippage_rate"])
        self.risk_long.setValue(cfg["risk"]["risk_per_trade_long"])
        self.risk_short.setValue(cfg["risk"]["risk_per_trade_short"])
        self.max_notional_short.setValue(cfg["risk"]["max_notional_pct_short"])

        v5 = cfg["v5"]
        self.rsi_len.setValue(v5["rsi_length"])
        self.macd_fast.setValue(v5["macd_fast"])
        self.macd_slow.setValue(v5["macd_slow"])
        self.macd_signal.setValue(v5["macd_signal"])
        self.boll_len.setValue(v5["boll_length"])
        self.boll_std.setValue(v5["boll_std"])
        self.ema_fast.setValue(v5["ema_fast"])
        self.ema_slow.setValue(v5["ema_slow"])
        self.adx_len.setValue(v5["adx_length"])
        self.bear_adx.setValue(v5["bear_adx_threshold"])
        self.ema_slope.setValue(v5["ema_slope_lookback"])
        self.atr_len.setValue(v5["atr_length"])
        self.atr_init.setValue(v5["atr_init_mult"])
        self.atr_trail.setValue(v5["atr_trail_mult"])
        self.tp2_long.setValue(v5["tp2_invalid_min_pnl_pct_long"])
        self.tp2_short.setValue(v5["tp2_invalid_min_pnl_pct_short"])
        self.tp2_hold.setValue(v5["tp2_invalid_min_hold_bars"])
        self.tp2_action.setCurrentText(v5["tp2_invalid_action"])
        self.tighten_to.setCurrentText(v5["tighten_to"])
        self.entry_adx_enabled.setChecked(v5.get("entry_adx_filter_enabled", True))
        self.entry_adx_period.setValue(v5.get("entry_adx_period", 14))
        self.entry_adx_min_long.setValue(v5.get("entry_adx_min_long", 20))
        self.entry_adx_min_short.setValue(v5.get("entry_adx_min_short", 25))

    def _collect_config(self) -> Dict[str, object]:
        return {
            "env": "testnet" if self.env_combo.currentText() == "Testnet" else "mainnet",
            "symbol": self.symbol_combo.currentText(),
            "trade_mode": (
                "long_only"
                if self.trade_mode.currentText() == "只做多"
                else "short_only"
                if self.trade_mode.currentText() == "只做空"
                else "both"
            ),
            "start_date": self.start_date.date().toString("yyyy-MM-dd"),
            "end_date": self.end_date.date().toString("yyyy-MM-dd"),
            "exec_tf": self.exec_tf_combo.currentText(),
            "filter_tf": self.filter_tf_combo.currentText(),
            "initial_capital": self.initial_capital.value(),
            "fee_rate": self.fee_rate.value(),
            "slippage_rate": self.slippage_rate.value(),
            "risk": {
                "risk_per_trade_long": self.risk_long.value(),
                "risk_per_trade_short": self.risk_short.value(),
                "max_notional_pct_short": self.max_notional_short.value(),
            },
            "v5": {
                "rsi_length": int(self.rsi_len.value()),
                "macd_fast": int(self.macd_fast.value()),
                "macd_slow": int(self.macd_slow.value()),
                "macd_signal": int(self.macd_signal.value()),
                "boll_length": int(self.boll_len.value()),
                "boll_std": float(self.boll_std.value()),
                "ema_fast": int(self.ema_fast.value()),
                "ema_slow": int(self.ema_slow.value()),
                "adx_length": int(self.adx_len.value()),
                "bear_adx_threshold": int(self.bear_adx.value()),
                "ema_slope_lookback": int(self.ema_slope.value()),
                "atr_length": int(self.atr_len.value()),
                "atr_init_mult": float(self.atr_init.value()),
                "atr_trail_mult": float(self.atr_trail.value()),
                "tp2_invalid_min_pnl_pct_long": float(self.tp2_long.value()),
                "tp2_invalid_min_pnl_pct_short": float(self.tp2_short.value()),
                "tp2_invalid_min_hold_bars": int(self.tp2_hold.value()),
                "tp2_invalid_action": self.tp2_action.currentText(),
                "tighten_to": self.tighten_to.currentText(),
                "entry_adx_filter_enabled": self.entry_adx_enabled.isChecked(),
                "entry_adx_period": int(self.entry_adx_period.value()),
                "entry_adx_min_long": float(self.entry_adx_min_long.value()),
                "entry_adx_min_short": float(self.entry_adx_min_short.value()),
            },
        }

    def _on_symbol_change(self, symbol: str) -> None:
        path = os.path.join("config", "default_config.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        preset = cfg.get("slippage_presets", {}).get(symbol)
        if preset is not None:
            self.slippage_rate.setValue(preset)

    def _on_test_connection(self) -> None:
        self.conn_status.setText("测试中...")
        api_key = self.api_key_input.text().strip()
        api_secret = self.api_secret_input.text().strip()
        testnet = self.env_combo.currentText() == "Testnet"

        self.api_thread = QThread(self)
        worker = ApiTestWorker(api_key, api_secret, testnet)
        worker.moveToThread(self.api_thread)
        self.api_thread.started.connect(worker.run)
        worker.finished.connect(self._on_api_test_result)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self.api_thread.quit)
        self.api_thread.start()

    def _on_api_test_result(self, payload: Dict[str, object]) -> None:
        if not payload.get("ok"):
            self.conn_status.setText("失败")
            self.conn_info.setText(str(payload.get("error")))
            return
        server_time = payload.get("server_time", 0)
        server_dt = datetime.fromtimestamp(server_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.conn_status.setText("成功")
        self.conn_info.setText(
            f"耗时 {payload.get('elapsed_ms')} ms | 服务器时间 {server_dt} | "
            f"USDT 钱包 {payload.get('usdt_wallet')} | 可用 {payload.get('usdt_available')}"
        )

    def _on_run(self) -> None:
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.spinner_timer.start(200)
        self.log_box.clear()
        cfg = self._collect_config()
        self.worker_thread = QThread(self)
        self.worker = BacktestWorker(
            cfg,
            self.api_key_input.text().strip(),
            self.api_secret_input.text().strip(),
            self.env_combo.currentText() == "Testnet",
            self.state,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_backtest_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.start()

    def _on_stop(self) -> None:
        if self.worker:
            self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_backtest_finished(self, payload: Dict[str, object]) -> None:
        self.spinner_timer.stop()
        self.spinner_label.setText("o")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        if payload.get("stopped"):
            self._append_log("已停止")
            return
        if payload.get("error"):
            return
        self.last_export_paths = payload.get("paths")
        self.state.push_result(payload)

    def _on_save_config(self) -> None:
        cfg = self._collect_config()
        path, _ = QFileDialog.getSaveFileName(self, "保存配置", "config.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        self._append_log(f"已保存配置: {path}")

    def _on_load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载配置", "", "JSON (*.json)")
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.symbol_combo.setCurrentText(cfg.get("symbol", "SOLUSDT"))
        mode = cfg.get("trade_mode", "both")
        if mode == "long_only":
            self.trade_mode.setCurrentText("只做多")
        elif mode == "short_only":
            self.trade_mode.setCurrentText("只做空")
        else:
            self.trade_mode.setCurrentText("多空都做")
        self.start_date.setDate(QDate.fromString(cfg.get("start_date", "2022-01-01"), "yyyy-MM-dd"))
        self.end_date.setDate(QDate.fromString(cfg.get("end_date", "2026-01-01"), "yyyy-MM-dd"))
        self.exec_tf_combo.setCurrentText(cfg.get("exec_tf", "4h"))
        self.filter_tf_combo.setCurrentText(cfg.get("filter_tf", "1d"))
        self.initial_capital.setValue(cfg.get("initial_capital", 10000))
        self.fee_rate.setValue(cfg.get("fee_rate", 0.0004))
        self.slippage_rate.setValue(cfg.get("slippage_rate", 0.0003))
        risk = cfg.get("risk", {})
        self.risk_long.setValue(risk.get("risk_per_trade_long", 0.006))
        self.risk_short.setValue(risk.get("risk_per_trade_short", 0.004))
        self.max_notional_short.setValue(risk.get("max_notional_pct_short", 0.35))

        v5 = cfg.get("v5", {})
        self.rsi_len.setValue(v5.get("rsi_length", 14))
        self.macd_fast.setValue(v5.get("macd_fast", 12))
        self.macd_slow.setValue(v5.get("macd_slow", 26))
        self.macd_signal.setValue(v5.get("macd_signal", 9))
        self.boll_len.setValue(v5.get("boll_length", 20))
        self.boll_std.setValue(v5.get("boll_std", 2.0))
        self.ema_fast.setValue(v5.get("ema_fast", 50))
        self.ema_slow.setValue(v5.get("ema_slow", 200))
        self.adx_len.setValue(v5.get("adx_length", 14))
        self.bear_adx.setValue(v5.get("bear_adx_threshold", 25))
        self.ema_slope.setValue(v5.get("ema_slope_lookback", 5))
        self.atr_len.setValue(v5.get("atr_length", 14))
        self.atr_init.setValue(v5.get("atr_init_mult", 2.6))
        self.atr_trail.setValue(v5.get("atr_trail_mult", 2.8))
        self.tp2_long.setValue(v5.get("tp2_invalid_min_pnl_pct_long", 0.002))
        self.tp2_short.setValue(v5.get("tp2_invalid_min_pnl_pct_short", 0.008))
        self.tp2_hold.setValue(v5.get("tp2_invalid_min_hold_bars", 16))
        self.tp2_action.setCurrentText(v5.get("tp2_invalid_action", "tighten_stop"))
        self.tighten_to.setCurrentText(v5.get("tighten_to", "mid"))
        self.entry_adx_enabled.setChecked(v5.get("entry_adx_filter_enabled", True))
        self.entry_adx_period.setValue(v5.get("entry_adx_period", 14))
        self.entry_adx_min_long.setValue(v5.get("entry_adx_min_long", 20))
        self.entry_adx_min_short.setValue(v5.get("entry_adx_min_short", 25))

        self._append_log(f"已加载配置: {path}")

    def _on_export_results(self) -> None:
        if not self.last_export_paths:
            self._append_log("没有可导出的结果")
            return
        dir_path = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not dir_path:
            return
        for path in self.last_export_paths:
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(dir_path, os.path.basename(path)))
        self._append_log(f"导出完成: {dir_path}")

    def _append_log(self, text: str) -> None:
        self.log_box.append(text)

    def _set_status(self, text: str) -> None:
        self._append_log(text)

    def _set_phase(self, text: str) -> None:
        self.phase_label.setText(text)

    def _set_detail(self, detail: Dict[str, object]) -> None:
        self.detail_label.setText(json.dumps(detail, ensure_ascii=False))

    def _on_error(self, text: str) -> None:
        self._append_log(f"错误: {text}")
        self.spinner_timer.stop()
        self.spinner_label.setText("o")
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        if "时间周期自检失败" in text:
            self.run_btn.setEnabled(False)

    def _render_results(self, payload: Dict[str, object]) -> None:
        report = payload.get("report", {})
        trades = payload.get("trades", pd.DataFrame())
        equity = payload.get("equity", pd.DataFrame())
        self.metrics_label.setText(
            f"TotalReturn: {report.get('total_return', 0):.2%} | "
            f"MDD: {report.get('mdd', 0):.2%} | "
            f"Sharpe: {report.get('sharpe', 0):.2f} | "
            f"Trades: {report.get('trades', 0)} | "
            f"WinRate: {report.get('win_rate', 0):.2%} | "
            f"PF: {report.get('profit_factor', 0):.2f} | "
            f"TP2Rate: {report.get('tp2_rate', 0):.2%}"
        )
        self.counters_label.setText(
            "Counters: " + json.dumps(report.get("counters", {}), ensure_ascii=False)
        )
        stop_breakdown = report.get("stop_breakdown", {})
        loss_stop = stop_breakdown.get("loss_stop", {})
        profit_stop = stop_breakdown.get("profit_stop", {})
        self.stop_breakdown_label.setText(
            "LossStop: "
            f"{loss_stop.get('count', 0)} / "
            f"{loss_stop.get('total_pnl', 0):.2f} / "
            f"{loss_stop.get('avg_pnl', 0):.2f} / "
            f"{loss_stop.get('hold_hours_p50', 0):.1f}h\n"
            "TrailStop: "
            f"{profit_stop.get('count', 0)} / "
            f"{profit_stop.get('total_pnl', 0):.2f} / "
            f"{profit_stop.get('avg_pnl', 0):.2f} / "
            f"{profit_stop.get('hold_hours_p50', 0):.1f}h"
        )

        self.exit_table.setRowCount(0)
        for row in report.get("exit_reason_top5", []):
            r = self.exit_table.rowCount()
            self.exit_table.insertRow(r)
            self.exit_table.setItem(r, 0, QTableWidgetItem(str(row.get("reason"))))
            self.exit_table.setItem(r, 1, QTableWidgetItem(str(row.get("count"))))
            self.exit_table.setItem(r, 2, QTableWidgetItem(f"{row.get('total_pnl', 0):.2f}"))

        self.side_table.setRowCount(0)
        for row in report.get("side_breakdown", []):
            r = self.side_table.rowCount()
            self.side_table.insertRow(r)
            self.side_table.setItem(r, 0, QTableWidgetItem(str(row.get("side"))))
            self.side_table.setItem(r, 1, QTableWidgetItem(str(row.get("count"))))
            self.side_table.setItem(r, 2, QTableWidgetItem(f"{row.get('total_pnl', 0):.2f}"))
            self.side_table.setItem(r, 3, QTableWidgetItem(f"{row.get('pf', 0):.2f}"))

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not equity.empty:
            ax.plot(equity["equity"].values, linewidth=1.2)
        ax.set_title("资金曲线")
        self.canvas.draw()

        self.trades_table.setRowCount(0)
        if not trades.empty:
            show = trades.head(20)
            for _, row in show.iterrows():
                r = self.trades_table.rowCount()
                self.trades_table.insertRow(r)
                self.trades_table.setItem(r, 0, QTableWidgetItem(str(row["entry_time"])))
                self.trades_table.setItem(r, 1, QTableWidgetItem(str(row["exit_time"])))
                self.trades_table.setItem(r, 2, QTableWidgetItem(str(row["side"])))
                self.trades_table.setItem(r, 3, QTableWidgetItem(f"{row['entry_price']:.4f}"))
                self.trades_table.setItem(r, 4, QTableWidgetItem(f"{row['exit_price']:.4f}"))
                self.trades_table.setItem(r, 5, QTableWidgetItem(f"{row['qty']:.4f}"))
                self.trades_table.setItem(r, 6, QTableWidgetItem(f"{row['pnl']:.2f}"))
                self.trades_table.setItem(r, 7, QTableWidgetItem(str(row["reason"])))

    def _spin(self) -> None:
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        self.spinner_label.setText(self.spinner_frames[self.spinner_index])
