import numpy as np

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self):
        self.batch_size = 0  # batch_size indicates the number of places to be trained
        # num_samples_per_class indicates the number of sample in each class
        self.num_samples_per_class = 0
        # numbers of indicators in each batch
        self.num_indicators = FLAGS.num_indicators
        self.train_file_path = FLAGS.train_csv_file  # the data file path
        self.test_file_path = FLAGS.test_csv_file  # the test file path
        self.time_seires_length = FLAGS.time_series_length  # the length of time series

    def generate_time_series_batch(self, train=True):
        if self.train_file_path is None:  # no datafile input
            return False
        else:
            file = self.train_file_path if train == True else self.test_file_path
            with open(file, 'r', encoding='utf-8') as file:
                context = file.read()  # context is the string input of data
                # row_list is a list with each row containing each line of
                # data
                row_list = context.split("\n")
                # the number of rows in data(each row indicate an indicator in
                # one place)
                #row_list.pop()
                num_rows = len(row_list)
                # the rows of data must be integral multiple of indicators
                assert num_rows % self.num_indicators is 0
                self.batch_size = int(num_rows / self.num_indicators)
                self.num_samples_per_class = len(row_list[0].split(','))
                if train is False:
                    assert self.batch_size == 1
                init_inputs = np.zeros(
                    [self.batch_size, self.num_indicators, self.num_samples_per_class - 1])
                init_lables = np.zeros(
                    [
                        self.batch_size,
                        self.num_indicators,
                        self.num_samples_per_class -
                        self.time_seires_length - 1])
                # declare input
                for batch_index in range(self.batch_size):
                    batch_data = []  # for each batch, initial batch data
                    batch_label = []  # the labels of each batch
                    for i in range(self.num_indicators):
                        # each row is a list of time series of each input
                        row = row_list[batch_index + i].split(',')
                        assert len(row) == self.num_samples_per_class
                        row.pop(0)
                        # the time series length must be identical for each
                        # indicator in each place
                        batch_data.append(row)
                        labelbatch = row[self.time_seires_length:]
                        batch_label.append(labelbatch)

                    init_inputs[batch_index] = np.array([batch_data])
                    init_lables[batch_index] = np.array([batch_label])


        return init_inputs, init_lables


