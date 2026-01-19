import sqlite3

from data_store import ensure_schema, upsert_klines


def test_upsert_dedup_and_update():
    conn = sqlite3.connect(":memory:")
    ensure_schema(conn)
    row = {
        "open_time": 1,
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "volume": 100.0,
        "close_time": 2,
    }
    result = upsert_klines(conn, "binance", "usdtm", "BTCUSDT", "4h", [row])
    assert result.inserted == 1
    assert result.updated == 0

    result2 = upsert_klines(conn, "binance", "usdtm", "BTCUSDT", "4h", [row])
    assert result2.skipped == 1

    row2 = dict(row)
    row2["close"] = 10.8
    result3 = upsert_klines(conn, "binance", "usdtm", "BTCUSDT", "4h", [row2])
    assert result3.updated == 1
