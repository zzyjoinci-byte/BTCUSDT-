# Binance USDT-M 永续合约回测工具（带 GUI）

这是一个从零创建的、可直接运行的桌面回测工具。支持 Binance USDT-M Futures 的 API 连通性测试、K 线本地缓存与增量维护、GUI 进度展示、报告与图表导出，并实现 v5 策略规则（含 BearGate、风险预算仓位、TP2_invalid 盈利保护、TrailStop）。

## 运行环境
- Python 3.11+
- macOS / Windows / Linux 均可

## 安装与运行
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/app.py
```

## 重要假设（已写入代码与默认配置）
- 标的为 USDT-M 线性合约，数量单位为 base 数量（如 SOL 数量）。
- 手续费与滑点按市价成交近似处理：
  - 买入价格 = close * (1 + slippage)
  - 卖出价格 = close * (1 - slippage)
  - 手续费按成交名义金额 * fee_rate
- 结构止损使用近 5 根的高/低点；初始止损取“更保守”：
  - 多头：max(ATR 止损, 结构止损)
  - 空头：min(ATR 止损, 结构止损)
- 过滤级别（filter_tf=1d）只对短线 BearGate 与日线趋势/区间过滤生效；不会使用未来数据（merge_asof 向后对齐）。
- 数据不足时自动截断回测区间，并在日志中提示。

## v5 策略要点（简述）
- 同时允许做多/做空；做空必须通过 BearGate。
- BearGate：日线 ADX 达标 + EMA50 斜率为负，仅作用于空头。
- 风险预算仓位：qty = 风险预算金额 / 止损距离；空头受 max_notional_pct_short 限制。
- TrailStop：逐 bar 收紧 ATR 轨迹止损。
- TP2_invalid 仅做盈利保护：
  - 仅在浮盈 > 0、达到最小盈利门槛、持仓 bar 数 >= min_hold_bars 才触发。
  - 默认动作：收紧止损到中位（mid）或 ATR trail。

## 目录结构
```
/src
  app.py
  ui_main.py
  state.py
  binance_api.py
  data_store.py
  resample.py
  indicators.py
  strategy_v5.py
  backtest_engine.py
  report.py
/config
  default_config.json
/data
  market.sqlite
/outputs
  <SYMBOL>_trades.csv
  <SYMBOL>_equity.csv
  <SYMBOL>_report.json
  <SYMBOL>_equity.png
  app.log
/tests
  test_data_store.py
  test_timeframe_validation.py
  test_risk_position.py
  test_tp2_invalid.py
  test_beargate.py
  test_gap_segments.py
```

## 输出说明
回测完成后，默认输出到 `outputs/`：
- `<SYMBOL>_trades.csv`
- `<SYMBOL>_equity.csv`
- `<SYMBOL>_report.json`
- `<SYMBOL>_equity.png`
- `app.log`（GUI 日志）

## 运行测试
```bash
pytest -q
```

## 备注
- 如果你没有 API Key，也可以直接运行回测：程序会优先使用本地缓存；没有缓存时会提示无法拉取。
- Binance API 权限需要启用 Futures 读取权限（只读即可）。
