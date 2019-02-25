import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from tensorflow.python.platform import flags
from data_generator import DataGenerator
from maml import MAML

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_indicators',
    56,
    "number of indicators in each location")
flags.DEFINE_integer('time_series_length', 2, "length of time series")
flags.DEFINE_string(
    'train_csv_file',
    "foursquare.csv",
    "the csv file path of training data")
flags.DEFINE_string(
    'test_csv_file',
    "testfoursquare.csv",
    "the csv file path of test data")
flags.DEFINE_string(
    'location_csv_file',
    "location.csv",
    "the csv file path of location")
flags.DEFINE_float("window_width", 1, "the Parzan window width in ")
flags.DEFINE_bool('train', False, 'True to train, False to test')
flags.DEFINE_float("l2_update", 1e-10, "step size alpha for loss function l2")
flags.DEFINE_float("l3_update", 1e-10, "step size lambda for loss function l3")
flags.DEFINE_float('meta_lr', 1e-12, 'the base learning rate of the generator')
flags.DEFINE_integer(
    'pretrain_iterations',
    5600,
    'number of pre-training iterations.')
flags.DEFINE_integer(
    'metatrain_iterations',
    1,
    'number of metatraining iterations.')
flags.DEFINE_integer('feed_length', 3, 'length of days to feed')
flags.DEFINE_string(
    'logdir',
    '',
    'directory for summaries and checkpoints.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_boolean('resume', False,'resume or not')
flags.DEFINE_integer('num_updates',5,'number of updates')


def train(model, saver, sess, exp_string, data_generator, resume_itr):
    SUMMARY_INTERVAL = 500
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 500
    TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
    print('Done initializing')
    prelosses, postlosses = [], []
    inputs, lables = data_generator.generate_time_series_batch(train=True)
    for itr in range(
            resume_itr,
            FLAGS.pretrain_iterations +
            FLAGS.metatrain_iterations):
        feed_dict = {}
        update_index = itr % int(
            data_generator.num_samples_per_class - FLAGS.feed_length)

        inputa = inputs[:, :, update_index:update_index + FLAGS.feed_length]
        labela = lables[:, :, update_index:update_index + FLAGS.feed_length - FLAGS.time_series_length]
        inputb = inputs[:, :, update_index:update_index + FLAGS.feed_length]
        labelb = lables[:, :, update_index:update_index + FLAGS.feed_length - FLAGS.time_series_length]
        feed_dict = {
            model.inputa: inputa,
            model.inputb: inputb,
            model.labela: labela,
            model.labelb: labelb,
            model.meta_lr: 1e-3}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            input_tensors = [model.metatrain_op]
        if itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0:
            input_tensors.extend(
                [model.all_weights, model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])
        result = sess.run(input_tensors, feed_dict)
        print(sess.run(model.weights,feed_dict))
        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            postlosses.append(result[-1])
        #print(sess.run(model.test, feed_dict))
        if (itr != 0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + \
                ', ' + str(np.mean(postlosses))
            #print(sess.run(model.test, feed_dict))
            print(print_str)
            prelosses, postlosses = [], []

            if (itr != 0) and itr % SAVE_INTERVAL == 0:
                print(FLAGS.logdir + '/' +exp_string)
                saver.save(
                    sess,
                    FLAGS.logdir +
                    '/' +
                    exp_string +
                    '/model' +
                    str(itr))



def test(model, sess, exp_string, data_generator):
    np.random.seed(1)
    random.seed(1)
    metaval_accuracies = []

    inputs, lables = data_generator.generate_time_series_batch(train=False)
    for itr in range(data_generator.num_samples_per_class - FLAGS.feed_length - 2):
        feed_dict = {}
        update_index = itr

        inputa = inputs[:, :, update_index:update_index + FLAGS.feed_length]
        labela = lables[:, :, update_index:update_index + FLAGS.feed_length - FLAGS.time_series_length]
        inputb = inputs[:, :, update_index:update_index + FLAGS.feed_length]
        labelb = lables[:, :, update_index:update_index + FLAGS.feed_length - FLAGS.time_series_length]
        feed_dict = {
            model.inputa: inputa,
            model.inputb: inputb,
            model.labela: labela,
            model.labelb: labelb,
            model.meta_lr: 0.001}
        input_tensors = [model.metatrain_op,model.total_losses2[FLAGS.num_updates - 1]]
        #input_tensors = [model.total_losses2[FLAGS.num_updates - 1]]
        result = sess.run(input_tensors, feed_dict)
        print(result[-1])
        if itr > 0.9*(data_generator.num_samples_per_class - FLAGS.feed_length - 2):
            metaval_accuracies.append(result[-1])
    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(data_generator.num_samples_per_class -
                                 data_generator.time_seires_length)
    print('Mean validation accuracy/loss and confidence intervals')
    print((means, ci95))

    out_filename = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + '_stepsize' + '.csv'
    out_pkl = FLAGS.logdir + '/' + exp_string + '/' + 'test_ubs' + \
        '_stepsize' + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    '''
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update' + str(i) for i in range(len(means))])
        writer.writerow(['update' + str(1)])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)
    '''


def main():
    if FLAGS.train:
        test_num_updates = FLAGS.num_updates
    else:
        test_num_updates = 5
    data_generator = DataGenerator()
    data_generator.generate_time_series_batch(train=FLAGS.train)
    model = MAML(data_generator.batch_size, test_num_updates)
    model.construct_model(input_tensors=None, prefix='metatrain_')
    model.summ_op = tf.summary.merge_all()
    saver = loader = tf.train.Saver(
        tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES),
        max_to_keep=10)
    sess = tf.InteractiveSession()

    exp_string = FLAGS.train_csv_file + '.numstep' + str(test_num_updates) + '.updatelr' + str(FLAGS.meta_lr)


    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(
            FLAGS.logdir + '/' + exp_string)
        print(model_file)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index(
                'model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1 + 5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, sess, exp_string, data_generator)


if __name__ == "__main__":
    main()
