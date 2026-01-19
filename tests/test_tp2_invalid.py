from strategy_v5 import tp2_invalid_should_tighten


def test_tp2_invalid_loss_gate():
    ok, reason = tp2_invalid_should_tighten(-0.01, 20, 0.002, 16)
    assert ok is False
    assert reason == "loss"


def test_tp2_invalid_profit_gate():
    ok, reason = tp2_invalid_should_tighten(0.001, 20, 0.002, 16)
    assert ok is False
    assert reason == "profit_gate"


def test_tp2_invalid_hold_gate():
    ok, reason = tp2_invalid_should_tighten(0.01, 10, 0.002, 16)
    assert ok is False
    assert reason == "hold_gate"


def test_tp2_invalid_allow():
    ok, reason = tp2_invalid_should_tighten(0.01, 20, 0.002, 16)
    assert ok is True
    assert reason == "allow"
