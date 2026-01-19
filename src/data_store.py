from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from resample import timeframe_to_minutes


@dataclass
class UpsertResult:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            exchange TEXT NOT NULL,
            market_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            PRIMARY KEY (exchange, market_type, symbol, timeframe, open_time)
        )
        """
    )
    conn.commit()


def fetch_open_times(
    conn: sqlite3.Connection,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> List[int]:
    cursor = conn.execute(
        """
        SELECT open_time
        FROM klines
        WHERE exchange=? AND market_type=? AND symbol=? AND timeframe=?
          AND open_time BETWEEN ? AND ?
        ORDER BY open_time ASC
        """,
        (exchange, market_type, symbol, timeframe, start_ms, end_ms),
    )
    return [int(row[0]) for row in cursor.fetchall()]


def calc_missing_segments(
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    open_times: Sequence[int],
) -> List[Tuple[int, int]]:
    if start_ms > end_ms:
        return []
    if not open_times:
        return [(start_ms, end_ms)]

    segments: List[Tuple[int, int]] = []
    if open_times[0] > start_ms:
        segments.append((start_ms, open_times[0] - interval_ms))

    for prev_t, next_t in zip(open_times[:-1], open_times[1:]):
        if next_t - prev_t > interval_ms:
            seg_start = prev_t + interval_ms
            seg_end = next_t - interval_ms
            if seg_start <= seg_end:
                segments.append((seg_start, seg_end))

    if open_times[-1] < end_ms:
        segments.append((open_times[-1] + interval_ms, end_ms))

    return segments


def estimate_bars(start_ms: int, end_ms: int, interval_ms: int) -> int:
    if end_ms < start_ms:
        return 0
    return int((end_ms - start_ms) // interval_ms) + 1


def upsert_klines(
    conn: sqlite3.Connection,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    klines: Iterable[Dict[str, object]],
) -> UpsertResult:
    result = UpsertResult()
    cursor = conn.cursor()
    for row in klines:
        open_time = int(row["open_time"])
        open_v = float(row["open"])
        high_v = float(row["high"])
        low_v = float(row["low"])
        close_v = float(row["close"])
        volume_v = float(row["volume"])
        close_time = int(row["close_time"])
        existing = cursor.execute(
            """
            SELECT open, high, low, close, volume, close_time
            FROM klines
            WHERE exchange=? AND market_type=? AND symbol=? AND timeframe=? AND open_time=?
            """,
            (exchange, market_type, symbol, timeframe, open_time),
        ).fetchone()
        if existing is None:
            cursor.execute(
                """
                INSERT INTO klines
                (exchange, market_type, symbol, timeframe, open_time,
                 open, high, low, close, volume, close_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exchange,
                    market_type,
                    symbol,
                    timeframe,
                    open_time,
                    open_v,
                    high_v,
                    low_v,
                    close_v,
                    volume_v,
                    close_time,
                ),
            )
            result.inserted += 1
        else:
            if (
                float(existing[0]) == open_v
                and float(existing[1]) == high_v
                and float(existing[2]) == low_v
                and float(existing[3]) == close_v
                and float(existing[4]) == volume_v
                and int(existing[5]) == close_time
            ):
                result.skipped += 1
            else:
                cursor.execute(
                    """
                    UPDATE klines
                    SET open=?, high=?, low=?, close=?, volume=?, close_time=?
                    WHERE exchange=? AND market_type=? AND symbol=? AND timeframe=? AND open_time=?
                    """,
                    (
                        open_v,
                        high_v,
                        low_v,
                        close_v,
                        volume_v,
                        close_time,
                        exchange,
                        market_type,
                        symbol,
                        timeframe,
                        open_time,
                    ),
                )
                result.updated += 1
    conn.commit()
    return result


def load_klines_df(
    conn: sqlite3.Connection,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    cursor = conn.execute(
        """
        SELECT open_time, open, high, low, close, volume, close_time
        FROM klines
        WHERE exchange=? AND market_type=? AND symbol=? AND timeframe=?
          AND open_time BETWEEN ? AND ?
        ORDER BY open_time ASC
        """,
        (exchange, market_type, symbol, timeframe, start_ms, end_ms),
    )
    rows = cursor.fetchall()
    if not rows:
        return pd.DataFrame(
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
            ]
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
        ],
    )
    return df


def load_available_range(
    conn: sqlite3.Connection,
    exchange: str,
    market_type: str,
    symbol: str,
    timeframe: str,
) -> Tuple[int | None, int | None]:
    cursor = conn.execute(
        """
        SELECT MIN(open_time), MAX(open_time)
        FROM klines
        WHERE exchange=? AND market_type=? AND symbol=? AND timeframe=?
        """,
        (exchange, market_type, symbol, timeframe),
    )
    row = cursor.fetchone()
    if not row or row[0] is None:
        return None, None
    return int(row[0]), int(row[1])


def calc_overlap_segments(segments: Sequence[Tuple[int, int]], interval_ms: int) -> List[Tuple[int, int]]:
    expanded: List[Tuple[int, int]] = []
    for start_ms, end_ms in segments:
        expanded.append((start_ms - interval_ms, end_ms + interval_ms))
    return expanded


def normalize_segments(segments: Sequence[Tuple[int, int]], start_ms: int, end_ms: int) -> List[Tuple[int, int]]:
    normalized = []
    for seg_start, seg_end in segments:
        seg_start = max(seg_start, start_ms)
        seg_end = min(seg_end, end_ms)
        if seg_start <= seg_end:
            normalized.append((seg_start, seg_end))
    return normalized


def timeframe_ms(timeframe: str) -> int:
    return timeframe_to_minutes(timeframe) * 60_000
