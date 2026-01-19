from data_store import calc_missing_segments


def test_calc_missing_segments_basic():
    start = 0
    end = 9
    interval = 1
    open_times = [0, 1, 2, 5, 6, 9]
    segments = calc_missing_segments(start, end, interval, open_times)
    assert segments == [(3, 4), (7, 8)]


def test_calc_missing_segments_empty():
    segments = calc_missing_segments(0, 9, 1, [])
    assert segments == [(0, 9)]
