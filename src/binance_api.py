from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from resample import timeframe_to_minutes


class BinanceAPI:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False) -> None:
        self.client = Client(api_key, api_secret)
        if testnet:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

    def test_connection(self) -> Dict[str, object]:
        start = time.time()
        server_time = self.client.futures_time()
        balances = self.client.futures_account_balance()
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
            batch = self.client.futures_klines(
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


def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except (BinanceAPIException, BinanceRequestException, Exception) as exc:  # noqa: BLE001
        return None, str(exc)
