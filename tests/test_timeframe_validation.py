import pandas as pd

from resample import validate_timeframe


def test_validate_timeframe_4h_ok():
    interval_ms = 4 * 60 * 60 * 1000
    times = [i * interval_ms for i in range(10)]
    df = pd.DataFrame({"open_time": times})
    ok, detail = validate_timeframe(df, "4h")
    assert ok is True


def test_validate_timeframe_fail():
    interval_ms = 3 * 60 * 60 * 1000
    times = [i * interval_ms for i in range(10)]
    df = pd.DataFrame({"open_time": times})
    ok, detail = validate_timeframe(df, "4h")
    assert ok is False
