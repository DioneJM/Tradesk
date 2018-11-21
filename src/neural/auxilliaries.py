# Aux functions defined here
import csv
import math
import numpy as np
from src.neural.globals import *
import tensorflow as tf
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.python.ops.rnn import dynamic_rnn
#import splunklib.client as client
import subprocess
from paramiko import SSHClient
from scp import SCPClient


def get_mid_prices(data):
    return data.loc[:, 'bidclose'].as_matrix()
    # highs = data.loc[:, 'High'].as_matrix()
    # lows = data.loc[:, 'Low'].as_matrix()
    # return (highs + lows) / 2.0

def split_data(data):
    return data[:train], data[train:]


def scale_data(training_data, testing_data):
    # First reshape as the sklearn module prefers tensors of Rank 2
    # The reshape simply serves to add an extra dimension to the tensor
    training_data = training_data.reshape(-1, 1)
    testing_data = testing_data.reshape(-1, 1)

    scaler = MinMaxScaler()

    for i in range(0, len(training_data), window):
        scaler.fit(training_data[i:i + window, :])
        training_data[i:i + window, :] = scaler.transform(training_data[i:i + window, :])

    # i = max(range(0, 10000, window))
    #
    # # Not forgetting to scale the left-over data
    # scaler.fit(training_data[(i + window):, :])
    # training_data[(i + window):, :] = scaler.transform(training_data[(i + window):, :])

    # Normalize the testing data with respect to the training data as the testing data is "unseen" at this stage
    return training_data.reshape(-1), scaler.transform(testing_data).reshape(-1)


def exponential_smoothing(training_data):
    # Exponential Moving Average
    ema = 0.0

    for t in range(train):
        ema = gamma * training_data[t] + (1 - gamma) * ema
        training_data[t] = ema

    return training_data


def gather_data(training, testing):
    return np.concatenate((training, testing), axis=0)


def save_data(predictions, testing_losses, x_axis_values):
    np.save('predictions.npy', predictions)
    np.save('testing_losses.npy', testing_losses)
    np.save('x_axis_values.npy', x_axis_values)


def machine_learn(train_data, mid_stock_prices):

    # ===================== Defining inputs and outputs ========================

    # Input data.
    train_inputs, train_outputs = [], []

    # The tensors train_inputs and train_outputs below are place-holders for the training and labelled data given to
    # the model. Right now, they are list of 50 tensors with shape (500, 1)
    for ui in range(sequence_size):
        train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size, dimension], name='train_inputs_%d' % ui))
        train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size, dimension], name='train_outputs_%d' % ui))

    # ===================== Defining memory network ========================

    lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],  state_is_tuple=True,
                                          initializer=tf.contrib.layers.xavier_initializer())
                  for li in range(number_layers)]

    # Adding dropout circumvents over-fitting
    drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1.0, output_keep_prob=1.0 - dropout,
                                                     state_keep_prob=1.0 - dropout) for lstm in lstm_cells]

    # The cells defined above are fused together as such
    drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    # ===================== Defining Regression Layer ========================

    w = tf.get_variable('w', shape=[num_nodes[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', initializer=tf.random_uniform([1], -0.1, 0.1))

    # Create cell state and hidden state variables to maintain the state of the LSTM
    c, h = [], []
    initial_state = []
    for li in range(number_layers):
        c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
        h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
        initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

    # Recall, the inputs, along with their corresponding labels, are lists of 2D tensors. The line below transforms this
    # these lists to Rank 3 Tensors with shape (50, 500, 1) as required by the dynamic_rnn function below
    all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)

    all_lstm_outputs, state = dynamic_rnn(drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
                                          time_major=True, dtype=tf.float32)

    # all_lstm_outputs has shape [seq_length, batch_size, num_nodes[-1]] above. We reshape it below to a Rank 2 Tensor
    # prior to inputting it to the Regression Layer.
    all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * sequence_size, num_nodes[-1]])

    # all_outputs has shape (sequence_size * batch_size, 1)
    all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)

    # Split the outputs above to obtain the output for each batch
    split_outputs = tf.split(all_outputs, sequence_size, axis=0)

    # ===================== Optimization & Loss ========================

    print('Defining training Loss')
    loss = 0.0
    # The initial state of the cell for each subsequent prediction in the training phase is given by the previous
    # final and hidden state for each respective layer
    with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(number_layers)] +
                                 [tf.assign(h[li], state[li][1]) for li in range(number_layers)]):
        for ui in range(sequence_size):
            loss += tf.reduce_mean(0.5 * (split_outputs[ui] - train_outputs[ui]) ** 2)

    print('Learning rate decay operations')
    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step, global_step + 1)
    tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
    tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

    learning_rate = tf.maximum(
        tf.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
        tf_min_learning_rate)

    # Optimizer.
    print('TF Optimization operations')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, clip)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v))

    print('\tAll done')

    print('Defining prediction related TF functions')

    # ===================== Prediction functions ========================
    # Name included to retrieve this placeholder when restoring the model
    sample_inputs = tf.placeholder(tf.float32, shape=[1, dimension], name='sample_inputs')

    # Maintaining LSTM state for prediction stage
    sample_c, sample_h, initial_sample_state = [], [], []
    for li in range(number_layers):
        sample_c.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
        sample_h.append(tf.Variable(tf.zeros([1, num_nodes[li]]), trainable=False))
        initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li], sample_h[li]))

    reset_sample_states = tf.group(*[tf.assign(sample_c[li], tf.zeros([1, num_nodes[li]]))
                                     for li in range(number_layers)],
                                   *[tf.assign(sample_h[li], tf.zeros([1, num_nodes[li]]))
                                     for li in range(number_layers)], name='reset_sample_states')

    sample_outputs, sample_state = dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                               initial_state=tuple(initial_sample_state), time_major=True,
                                               dtype=tf.float32)

    # The final cell and hidden states are again recycled in the testing phase for each subsequent prediction given
    # a test point. That being said, when transitioning to the next testing point, the cell state is reset for obvious
    # reasons. (It defeats the purpose of the testing phase)
    with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(number_layers)] +
                                 [tf.assign(sample_h[li], sample_state[li][1]) for li in range(number_layers)]):
        # Name included to retrieve this placeholder when restoring the model
        # Recall the bias variable is just a scalar. The operation below is thus acceptable
        sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs, [1, -1]), w, b, name="sample_prediction")

    print('\tAll done')

    # ===================== Training ========================

    testing_interval = 1  # Interval you make test predictions

    train_seq_length = train_data.size  # Full length of the training data

    # Store training losses here (training loss per epoch)
    train_mse_ot = []

    # Store testing losses here
    test_mse_ot = []

    # Store predictions here
    predictions_over_time = []

    # Comment out if using GPU

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allocator_type = 'BFC'
    session = tf.Session(config=config)

    #Comment out below if using CPU only
    # session = tf.Session()

    with session.as_default():
        tf.global_variables_initializer().run()

        # Saver for saving the Tensor-flow model
        # The save_relative_paths needs to be set to True in order to restore the model
        saver = tf.train.Saver(save_relative_paths=True)

        # To keep tabs on the epoch pertaining to the lowest loss
        lowest_loss = math.inf

        # Used for decaying learning rate
        loss_non_decrease_count = 0

        # If the test error hasn't increased in this many steps, decrease learning rate
        loss_non_decrease_threshold = 2

        print('Initialized')
        average_loss = 0

        # Define data generator
        data_gen = DataGenerator(train_data, batch_size, sequence_size)

        x_axis_seq = []

        # Points you start our test predictions from
        test_points_seq = np.arange(train, train + testing_range, sequence_size).tolist()

        for ep in range(epochs):

            for step in range(train_seq_length // batch_size):

                # Get sequence_size batches of training and labelled data
                u_data, u_labels = data_gen.unroll_batches()

                feed_dict = {}
                for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
                    # The place-holders train_inputs, train_outputs, tf_learning_rate and tf_learning_rate_min are defined
                    # here prior to obtaining the predictions
                    feed_dict[train_inputs[ui]] = dat.reshape(-1, 1)
                    feed_dict[train_outputs[ui]] = lbl.reshape(-1, 1)

                feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate: 0.000001})

                # Obtain the loss for a single iteration for a given epoch given the training and labelled data and
                # learning rates. Although we are primarily interested in the loss we need to also compute the optimizer
                # otherwise the gradient updates to the variables will not be applied
                _, dl = session.run([optimizer, loss], feed_dict=feed_dict)

                average_loss += dl

            if (ep + 1) % testing_interval == 0:

                # Get the average loss over all iterations for a given epoch
                average_loss = average_loss / (testing_interval * (train_seq_length // batch_size))

                # The average loss
                if (ep + 1) % testing_interval == 0:
                    print('Average loss at step %d: %f' % (ep + 1, average_loss))

                # Add the average loss of this epoch to the container storing the training losses
                train_mse_ot.append(average_loss)

                # Reset the loss in preparation for the next epoch
                average_loss = 0

                # For each test point in the testing sequence, we compute the next sequence_size predictions given in
                # our_predictions below. Once our_predictions is exhausted we append it here. This contains the predictions
                # for all testing points then
                predictions_seq = []

                # The average loss associated with each test point predictions is stored here
                mse_test_loss_seq = []

                # Loop through each and every test point after completion of an epoch
                for w_i in test_points_seq:

                    # The loss associated with each prediction for a given test point is accumulated here
                    mse_test_loss = 0.0

                    # Given a test sample point this container contains the predictions for the next sequence_size points
                    our_predictions = []

                    if (ep + 1) - testing_interval == 0:
                        # Only calculate x_axis values in the first validation epoch
                        x_axis = []

                    # Feed in the recent past behavior of stock prices to make predictions from that point onwards given
                    # our model thus far (feed_dict is not reset at this stage..)
                    for tr_i in range(w_i - sequence_size + 1, w_i - 1):
                        current_price = mid_stock_prices[tr_i]
                        feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)
                        _ = session.run(sample_prediction, feed_dict=feed_dict)

                    feed_dict = {}

                    current_price = mid_stock_prices[w_i - 1]

                    feed_dict[sample_inputs] = np.array(current_price).reshape(1, 1)

                    # Make predictions for this many steps where each prediction uses the previous prediction as the input
                    for pred_i in range(sequence_size):

                        pred = session.run(sample_prediction, feed_dict=feed_dict)

                        # Add to the predictions for each test sample point here
                        our_predictions.append(np.asscalar(pred))

                        feed_dict[sample_inputs] = np.asarray(pred).reshape(-1, 1)

                        if (ep + 1) - testing_interval == 0:
                            # Only calculate x_axis values in the first validation epoch
                            x_axis.append(w_i + pred_i)

                        # Accumulate the predicted loss for each test point here
                        mse_test_loss += 0.5 * (pred - mid_stock_prices[w_i + pred_i]) ** 2

                    # Prior to moving on to the next point to sample we reset the cell and hidden states for obvious reasons
                    session.run(reset_sample_states)

                    # Recall predictions_seq contains the predictions for all testing points. our_predictions,
                    # at this stage, contains the sequence_size predictions for a given test point
                    predictions_seq.append(np.array(our_predictions))

                    # Get the average loss for the predictions
                    mse_test_loss /= sequence_size

                    # This will contain the average loss for each test point
                    mse_test_loss_seq.append(mse_test_loss)

                    if (ep + 1) - testing_interval == 0:
                        x_axis_seq.append(x_axis)

                # Get the mean of the (average) loss for all test points
                current_test_mse = np.mean(mse_test_loss_seq)

                if current_test_mse < lowest_loss:
                    lowest_loss = current_test_mse
                    print("Encountered a lower loss. (Re)saving model")
                    saver.save(session, './Model', global_step=None, meta_graph_suffix="meta",
                               write_meta_graph=True, write_state=True)

                # Learning rate decay logic
                if len(test_mse_ot) > 0 and current_test_mse > min(test_mse_ot):
                    loss_non_decrease_count += 1
                else:
                    loss_non_decrease_count = 0

                if loss_non_decrease_count > loss_non_decrease_threshold:
                    session.run(inc_gstep)
                    loss_non_decrease_count = 0
                    print('\tDecreasing learning rate by 0.5')

                # Store the mean of the average losses incurred for each testing point and store in this container
                test_mse_ot.append(current_test_mse)
                print('\tTest MSE: %.5f' % np.mean(mse_test_loss_seq))
                predictions_over_time.append(predictions_seq)
                print('\tFinished Predictions')

            print(ep, epochs - ep)
        session.close()
        return predictions_over_time, test_mse_ot, x_axis_seq


def visualize_data(data, mid_stock_prices, predictions, mse_test_losses, x_axis_seq):
    plt.figure(figsize=(18, 18))
    plt.subplot(2, 1, 1)
    plt.plot(range(data.shape[0]), mid_stock_prices, color='b')

    # Plotting how the predictions change over time
    # Plot older predictions with low alpha and newer predictions with high alpha
    start_alpha = 0.25
    alpha = np.arange(start_alpha, 1.1, (1.0 - start_alpha) / len(predictions[::3]))
    for p_i, p in enumerate(predictions[::3]):
        for xval, yval in zip(x_axis_seq, p):
            plt.plot(xval, yval, color='r', alpha=alpha[p_i])

    plt.title('Evolution of Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(11000, 12500)

    plt.subplot(2, 1, 2)

    # Predicting the best test prediction you got
    plt.plot(range(data.shape[0]), mid_stock_prices, color='b')
    for xval, yval in zip(x_axis_seq, predictions[np.argmin(mse_test_losses)]):
        plt.plot(xval, yval, color='r')

    plt.title('Best Test Predictions Over Time', fontsize=18)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.xlim(11000, 12500)
    plt.show()


def load():
    return np.load('predictions.npy', allow_pickle=True), np.load('testing_losses.npy', allow_pickle=True),\
           np.load('x_axis_values.npy', allow_pickle=True)


def get_predictions(live_data):

    # Initialize a Tensor-flow session in order to load the model

    # Comment out if using GPU

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allocator_type = 'BFC'
    session = tf.Session(config=config)

    #Comment out below if using CPU only
    # session = tf.Session()

    restorer = tf.train.import_meta_graph(model_directory_path + '.meta')
    restorer.restore(session, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    # Get the required placeholder in order to update the feed dictionary
    sample_inputs = graph.get_tensor_by_name('sample_inputs:0')

    # The required operations
    sample_prediction = graph.get_tensor_by_name('sample_prediction:0')

    feed_dictionary = dict()
    for i in live_data[:-1]:
        feed_dictionary[sample_inputs] = i.reshape(1, dimension)
        _ = session.run(fetches=sample_prediction, feed_dict=feed_dictionary)

    feed_dictionary.clear()
    feed_dictionary[sample_inputs] = live_data[-1].reshape(1, dimension)

    predictions = list()

    for _ in range(number_of_future): 
        # Make predictions for this many steps where each prediction uses the previous prediction as the input
        pred = session.run(sample_prediction, feed_dict=feed_dictionary)

        # Add to the predictions for each test sample point here
        predictions.append(np.asscalar(pred))

        feed_dictionary[sample_inputs] = np.asarray(pred).reshape(-1, 1)

    session.close()
    # Call the reset_state operation at this stage
    return predictions
    pass


def output_to_csv(predictions, time, visualize=False):
    with open(csv_file_name, mode='w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        # for [time, predict]
        writer.writerow(["Time", "1st Prediction", "2nd Prediction", "3rd Prediction", "4th Prediction", "5th Prediction", "6th Prediction"])
        #writer.writerow(["Time"] + )
        writer.writerow([time] + predictions)

        # writer.writerow(["Value", 'chicken'])
        # for prediction in predictions:
        #         writer.writerow([prediction, 'chicken'])

            # The argument to writerow must be an iterable type
            # writer.writerow([prediction])

    csvfile.close()
    # index_name.upload(index_directory)


    # Finally plot the predictions instead of manually looking through the array prior to making financial decision
    if visualize:
        plt.plot(np.arange(start=0, stop=number_of_future, step=1), predictions, color='r')
        plt.title('Future predictions', fontsize=18)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Predictions', fontsize=18)
        plt.ylim(0, 1)
        plt.show()

    # TODO: Possibly store previous predictions and update the plot of the previous predictions along with the current
    # TODO: predictions in order to get a better overall picture. I am just plotting the current predictions below

    # Not sure if resetting the cell states is necessary at this stage as we are re-loading the model every-time we make
    # a series of predictions. Investigate this later. (Comment for Marwan)

# def output_to_txt(predictions):
#     baconFile = open('predictions.xml', 'w')
#     baconFile.write('Value\n')
#     for num in range(len(predictions)):
#         baconFile.write(str(predictions[num]))
#         baconFile.write('\n')
#     baconFile.close()


def create_client():
    host = "localhost"
    port = 8000
    username = admin
    password = "Whereisthelove54!"

    service = client.connect(host = host, port = port, username = username, password = password)

    for app in service.apps:
        print(app.name)

    index = service.indexes.create("index")
    return index


    



