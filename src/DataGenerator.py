import numpy as np


class DataGenerator(object):

    # Class variable
    extent = 5

    def __init__(self, stock_prices, batch_size, sequence_size):

        # This is the full data sequence
        self.stock_prices = stock_prices

        # Size of each element in the sequence
        self.batch_size = batch_size

        # The size of the sequence given to the memory layer.
        self.sequence_size = sequence_size

        # Length of the training sequence
        self.length = len(self.stock_prices) - self.sequence_size
        self.segment_length = self.length // self.batch_size
        self.pointer = [offset * self.segment_length for offset in range(self.batch_size)]

    def next_batch(self):

        # A grand total of self.sequence_size sequential data is passed to each memory layer
        sequential_data = np.zeros(self.batch_size, dtype=np.float32)
        # The labels associated with each input
        sequential_labels = np.zeros(self.batch_size, dtype=np.float32)

        for b in range(self.batch_size):
            if self.pointer[b] + 1 >= self.length:
                self.pointer[b] = np.random.randint(0, (b + 1) * self.segment_length)

            sequential_data[b] = self.stock_prices[self.pointer[b]]
            sequential_labels[b] = self.stock_prices[self.pointer[b] + np.random.randint(1, self.extent)]

            self.pointer[b] = (self.pointer[b] + 1) % self.length

        return sequential_data, sequential_labels

    def unroll_batches(self):

        unrolled_data = list()
        unrolled_labels = list()

        for ui in range(self.sequence_size):

            data, labels = self.next_batch()
            unrolled_data.append(data)
            unrolled_labels.append(labels)

        return unrolled_data, unrolled_labels

    def reset_indices(self):
        for b in range(self.batch_size):
            self.pointer[b] = np.random.randint(b * self.segment_length, min((b + 1) * self.segment_length,
                                                                             self.length - 1))

