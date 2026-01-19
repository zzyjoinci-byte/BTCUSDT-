from strategy_v5 import calculate_position_size


def test_risk_position_with_cap():
    qty = calculate_position_size(
        equity=10000,
        risk_pct=0.01,
        entry_price=100,
        stop_price=90,
        max_notional_pct=0.05,
    )
    assert qty == 5


def test_risk_position_no_cap():
    qty = calculate_position_size(
        equity=10000,
        risk_pct=0.01,
        entry_price=100,
        stop_price=90,
        max_notional_pct=None,
    )
    assert qty == 10
