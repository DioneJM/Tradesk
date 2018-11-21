from src.trading.tradingapi import *

# Example
# def inc(x):
#     return x + 1
#
# def test_answer():
#     assert inc(3) == 5

def test_get_instruments():
    api = TradingAPI()
    assert len(api.get_instruments()) == 61

