from src.auxilliaries import *
import tensorflow as tf
from src.trading.tradingapi import TradingAPI
from src.trading.trader import Trader
import random
import os

TRADING_INTERVAL = 5

def random_floats(low, high, size):
    return [random.uniform(low, high) for _ in range(size)]

if __name__ == "__main__":

    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(server, port, username, password)

    # index = create_client()

    while True:

        #TODO: Check with Benny for the format of this data
        data = api.get_historical_data(number=10000)

        # We work with the average of the highs and low of the stock price.
        mids = get_mid_prices(data)

        # Split into training and testing. About 90 and 10% used for the former and latter respectively
        training, testing = split_data(mids)

        # Scale data to the range [0,1] due to varying magnitude fluctuations of the stock as a function of time.
        training, testing = scale_data(training, testing)

        # Smooth the data in order to rid ourselves of the inherent raggedness associated with variations in stock
        # prices. For obvious reasons this is only performed upon the training data
        training = exponential_smoothing(training)

        mid_stock_prices = gather_data(training, testing)

        # TODO: Plot training data after smoothing and rescaling (Include image in final presentation perhaps)
        # If the directory exists then this means a model exists. We proceed with making real-time financial predictions
        # and trading based on these predictions
        # In the instance previous sessions were run the graph needs to be reset
        tf.reset_default_graph()

        start_time = time.time()

        predictions, testing_losses, x_axis_values = machine_learn(training, mid_stock_prices)

        finish_time = time.time()

        print (finish_time - start_time)

        #time.sleep(5 - (finish_time - start_time))

        previous_data = get_mid_prices((api.get_historical_data(number=sequence_size)))
        preds = (get_predictions(previous_data))

        output_to_csv(preds, time.time())

        #scp bash function in python
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(csv_file_name, put_address)
        print ('CSV HAS SENT YO!!!!')

        # save_data(predictions, testing_losses, x_axis_values)

