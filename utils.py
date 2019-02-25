import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.platform import flags
import math

from math import cos, asin, sqrt

FLAGS = flags.FLAGS
tfd = tfp.distributions


def distance(place_one, place_two):
    # calculate the distance between two place, the inputs are the location of
    # two places
    lat1, lon1 = float(place_one[0]), float(place_one[1])
    lat2, lon2 = float(place_two[0]), float(place_two[1])
    p = 0.017453292519943295     # Pi/180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * \
        cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return -12742 * asin(sqrt(a))

def distance_rou(place_one,place_two):
    lat1, lon1 = float(place_one[0]), float(place_one[1])
    lat2, lon2 = float(place_two[0]), float(place_two[1])
    return math.exp(-((lat1-lat2)**2+(lon1-lon2)**2)**(1/2))


def get_matrix_distance():
    path = FLAGS.location_csv_file
    with open(path, 'r') as file:
        context = file.read()
        location_list = context.split('\n')
        num_locations = len(location_list)
        for row in range(num_locations):
            location_list[row] = location_list[row].split(',')
    distance_matrix = np.zeros([num_locations, num_locations],dtype=np.float32)
    for i in range(len(location_list)):
        for j in range(len(location_list)):
            distance_matrix[i][j] = distance_rou(
                location_list[i], location_list[j])
    return distance_matrix, num_locations


def loss_function_l1(pred, lables):
    return tf.reduce_sum(tf.square(pred - lables), [0, 1])
def abs_lose(pred, lables):
    return tf.reduce_sum(tf.abs(pred-lables),[0,1])

def loss_function_l3(pred, lables):
    width = FLAGS.window_width
    num_sample = pred.shape[1]
    gaussion_sum = tf.constant([0], dtype=tf.float32)
    for i in range(num_sample):
        for j in range(num_sample):
            dist1 = tfd.Normal(lables[:, j], 2 * (width**2))
            dist2 = tfd.Normal(pred[:, j], 2 * (width**2))
            gaussion_sum = gaussion_sum + \
                tf.reduce_sum(dist1.prob(pred[:, i]) - dist2.prob(pred[:, i]))
    return gaussion_sum / (gaussion_sum ** 2)


def loss_function_l2(all_weights):
    a, num_locations = get_matrix_distance()
    d = np.zeros([num_locations, num_locations],dtype=np.float32)
    sum_a = np.sum(a, axis=1)
    for i in range(num_locations):
        d[i][i] = sum_a[i]
    middle_matrix = tf.constant(d - a)
    return tf.trace(tf.matmul(
        all_weights,
        tf.matmul(
            middle_matrix,
            tf.transpose(all_weights))))

def get_pred(input, weight, bias,reuse = False):
    length = FLAGS.time_series_length
    input_length = FLAGS.feed_length
    pred = []
    biasl = []
    weight = tf.reshape(weight, [FLAGS.time_series_length, 1])
    for i in range(input_length - length):
        pred.append(tf.matmul(input[:, i:i + length], weight)[:])
        biasl.append(bias)
    pred = tf.transpose(tf.convert_to_tensor(pred))[0]
    pred = pred + tf.convert_to_tensor(biasl)
    return pred





def inside_loss(pred, lables):
    return loss_function_l1(pred, lables) #+ loss_function_l3(pred, lables)


def outside_loss(weights):
    return loss_function_l2(weights)
