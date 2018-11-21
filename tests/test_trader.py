from src.trading.trader import *

# Example
# def inc(x):
#     return x + 1
#
# def test_answer():
#     assert inc(3) == 5

def test_decision():
    tr = Trader("null")
    series = [1.2, 1.3, 1.4, 1.5, 1.6]
    assert tr.decide(series) == "buy"
