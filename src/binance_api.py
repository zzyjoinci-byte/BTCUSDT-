from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from resample import timeframe_to_minutes


class BinanceAPI:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: str = "mainnet",
        timeout: int = 10,
        retries: int = 1,
    ) -> None:
        self.environment = environment
        self.retries = max(0, retries)
        self.client = Client(api_key, api_secret, requests_params={"timeout": timeout})
        if environment.lower() == "testnet":
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

    def test_connection(self) -> Dict[str, object]:
        start = time.time()
        server_time = self._call(self.client.futures_time)
        balances = self._call(self.client.futures_account_balance)
        elapsed_ms = int((time.time() - start) * 1000)
        usdt_wallet = 0.0
        usdt_available = 0.0
        for item in balances:
            if item.get("asset") == "USDT":
                usdt_wallet = float(item.get("balance", 0))
                usdt_available = float(item.get("availableBalance", 0))
                break
        return {
            "server_time": int(server_time.get("serverTime", 0)),
            "elapsed_ms": elapsed_ms,
            "usdt_wallet": usdt_wallet,
            "usdt_available": usdt_available,
        }

    def get_account_summary(self) -> Dict[str, object]:
        info = self._call(self.client.futures_account)
        wallet = float(info.get("totalWalletBalance", info.get("walletBalance", 0)))
        available = float(info.get("availableBalance", 0))
        margin = float(info.get("totalMarginBalance", info.get("marginBalance", 0)))
        unrealized = float(info.get("totalUnrealizedProfit", info.get("unrealizedProfit", 0)))
        return {
            "walletBalance": wallet,
            "availableBalance": available,
            "marginBalance": margin,
            "unrealizedProfit": unrealized,
        }

    def fetch_klines_latest(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 200,
    ) -> List[Dict[str, object]]:
        interval = self._tf_to_interval(timeframe)
        batch = self._call(
            self.client.futures_klines,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        klines: List[Dict[str, object]] = []
        for item in batch:
            klines.append(
                {
                    "open_time": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": int(item[6]),
                }
            )
        return klines

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        reduce_only: bool = False,
    ) -> Dict[str, object]:
        return self._call(
            self.client.futures_create_order,
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity,
            reduceOnly=reduce_only,
        )

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, object]:
        return self._call(self.client.futures_cancel_order, symbol=symbol, orderId=order_id)

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, object]:
        return self._call(self.client.futures_change_leverage, symbol=symbol, leverage=leverage)

    def get_exchange_info(self, symbol: str) -> Dict[str, object]:
        info = self._call(self.client.futures_exchange_info)
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                lot_size = filters.get("LOT_SIZE", {})
                min_notional = filters.get("MIN_NOTIONAL", {})
                return {
                    "min_qty": float(lot_size.get("minQty", 0)),
                    "max_qty": float(lot_size.get("maxQty", 0)),
                    "step_size": float(lot_size.get("stepSize", 0)),
                    "min_notional": float(min_notional.get("notional", 0)),
                    "precision": int(s.get("quantityPrecision", 8)),
                }
        return {"min_qty": 0.0, "max_qty": 0.0, "step_size": 0.0, "min_notional": 0.0, "precision": 8}

    def get_position(self, symbol: str) -> Dict[str, object]:
        positions = self._call(self.client.futures_position_information, symbol=symbol)
        if not positions:
            return {"position_amt": 0.0, "entry_price": 0.0, "mark_price": 0.0, "notional": 0.0}
        pos = positions[0]
        position_amt = float(pos.get("positionAmt", 0))
        entry_price = float(pos.get("entryPrice", 0))
        mark_price = float(pos.get("markPrice", 0))
        notional = position_amt * mark_price
        return {
            "position_amt": position_amt,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "notional": notional,
        }

    def fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        progress_cb: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, object]]:
        interval = self._tf_to_interval(timeframe)
        interval_ms = timeframe_to_minutes(timeframe) * 60_000
        limit = 1500
        klines: List[Dict[str, object]] = []
        cursor = start_ms
        while cursor <= end_ms:
            batch = self._call(
                self.client.futures_klines,
                symbol=symbol,
                interval=interval,
                startTime=cursor,
                endTime=end_ms,
                limit=limit,
            )
            if not batch:
                break
            for item in batch:
                klines.append(
                    {
                        "open_time": int(item[0]),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "close_time": int(item[6]),
                    }
                )
            last_open = int(batch[-1][0])
            cursor = last_open + interval_ms
            if progress_cb:
                progress_cb(len(klines))
            if len(batch) < limit:
                break
        return klines

    @staticmethod
    def _tf_to_interval(timeframe: str) -> str:
        tf = timeframe.lower()
        if tf == "1m":
            return Client.KLINE_INTERVAL_1MINUTE
        if tf == "5m":
            return Client.KLINE_INTERVAL_5MINUTE
        if tf == "15m":
            return Client.KLINE_INTERVAL_15MINUTE
        if tf == "1h":
            return Client.KLINE_INTERVAL_1HOUR
        if tf == "4h":
            return Client.KLINE_INTERVAL_4HOUR
        if tf == "1d":
            return Client.KLINE_INTERVAL_1DAY
        raise ValueError(f"不支持的时间周期: {timeframe}")

    def _call(self, func, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return func(*args, **kwargs)
            except (BinanceAPIException, BinanceRequestException, Exception) as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(0.5)
        raise RuntimeError(self._format_error(last_exc))

    @staticmethod
    def _format_error(exc: Exception | None) -> str:
        if exc is None:
            return "未知错误"
        if isinstance(exc, BinanceAPIException):
            code = getattr(exc, "code", None)
            msg = getattr(exc, "message", str(exc))
            if code in (-2014, -2015):
                return "鉴权失败或权限不足，请检查 API Key 权限"
            if code == -1021:
                return "时间戳错误，请检查系统时间"
            if code == -1003:
                return "请求过于频繁，请稍后再试"
            if code == -1000:
                return f"API错误(-1000): {msg}。可能原因：数量精度不符合要求、数量低于最小值、名义金额不足"
            if code == -1111:
                return f"数量精度错误: {msg}。请检查币对的最小数量和步长"
            if code == -4164:
                return f"最小名义金额不足: {msg}"
            return f"API错误({code}): {msg}"
        if isinstance(exc, BinanceRequestException):
            return f"网络请求错误: {exc}"
        return f"未知错误: {exc}"


def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except (BinanceAPIException, BinanceRequestException, Exception) as exc:  # noqa: BLE001
        return None, str(exc)
