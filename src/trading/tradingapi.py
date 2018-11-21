import os
import fxcmpy

FILE_LOC = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUTPUT_LOC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

class TradingAPI:

    def __init__(self):
        self.con = fxcmpy.fxcmpy(config_file=os.path.join(FILE_LOC, "fxcm.cfg"), server="demo")
        print("Connected to FXCM Server")
        self.instruments = []

    def get_instruments(self):
        """
        :return: List of instruments
        """
        if not self.instruments:
            self.instruments = self.con.get_instruments()
        return self.instruments

    def get_historical_data(self, investment, period, number):
        """
        :return: Pandas DataFrame
        """
        df = self.con.get_candles(investment, period=period, number=number)
        df['date'] = df.index
        df = df[['date', 'bidclose']]
        return df

    def subscribe(self, investment):
        try:
            self.con.subscribe_market_data(investment)
        except:
            raise

    def subscribe_all(self, investments):
        if any(investments not in self.con.get_subscribed_symbols()):
            try:
                for i in investments:
                    self.con.subscribe_market_data(i)
            except:
                raise

    def get_live_data(self, investment, max_prices):
        self.con.set_max_prices(max_prices)
        df = self.con.get_prices(investment)
        return df

    def get_all_live_data(self, investments, max_price):
        """
        :param investments: List of Instruments
        :param max_price: max size of df
        :return: df containing each of the investments
        """
        pass

    def buy_at_market_price(self, investment, size):
        return self.con.create_market_buy_order(investment, size)

    def sell_at_market_price(self, investment, size):
        return self.con.create_market_sell_order(investment, size)

    def open_trade(self, investment, size, stop, limit):
        order_id = self.con.open_trade(investment, amount=size, stop=stop, limit=limit, time_in_force="GTC",
                                       is_buy=True, order_type="AtMarket")
        return order_id

    def close_trade(self, investment, size, stop, limit):
        order_id = self.con.open_trade(investment, amount=size, stop=stop, limit=limit, is_buy=False, time_in_force="GTC")
        return order_id

    def close_all_trade(self):
        pass

    def get_account_snapshot(self):
        return self.con.get_accounts_summary()

    def get_orders_snapshot(self):
        return self.con.get_orders()

    def get_open_positions_snapshot(self):
        return self.con.get_open_positions_summary()

    def get_closed_positions_snapshot(self):
        return self.con.get_closed_positions_summary()

    def shutdown(self):
        self.con.close()

    def batch_generate_csv(self):
        print("Generating csv files...")
        self.get_account_snapshot().to_csv(os.path.join(OUTPUT_LOC, "account_snapshot.csv"), index=False)
        self.get_orders_snapshot().to_csv(os.path.join(OUTPUT_LOC, "orders_snapshot.csv"), index=False)
        self.get_open_positions_snapshot().to_csv(os.path.join(OUTPUT_LOC, "open_snapshot.csv"), index=False)
        self.get_closed_positions_snapshot().to_csv(os.path.join(OUTPUT_LOC, "close_snapshot.csv"), index=False)
        print("CSV generation complete.")

    def has_money(self):
        return True if self.get_account_snapshot()['UsableMargin'] >= 450 else False
