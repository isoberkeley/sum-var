import tensorflow as tf
from tensorflow.python.platform import flags
from utils import inside_loss, outside_loss, get_pred
import utils
FLAGS = flags.FLAGS

class MAML:
    def __init__(self, batch_size , test_num_update = 5):
        self.dim_input = [FLAGS.num_indicators, FLAGS.time_series_length]
        self.dim_output = FLAGS.num_indicators
        self.l2_update = FLAGS.l2_update
        self.l3_update = FLAGS.l3_update
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr,())
        self.test_num_updates = test_num_update
        self.inside_lose = inside_loss
        self.outside_lose = outside_loss
        self.forward = get_pred
        self.construct_weights = self.construct_fc_weight
        self.batch_size = batch_size

    def construct_model(self, input_tensors = None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(dtype=tf.float32)
            self.inputb = tf.placeholder(dtype=tf.float32)
            self.labela = tf.placeholder(dtype=tf.float32)
            self.labelb = tf.placeholder(dtype=tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reise_variables()
                weights = self.weights
            else:
                self.weights, self.bias = weights, bias = self.construct_weights()

            lossesa, outputas, lossesb, outputbs = [], [], [], []
            num_updates = self.test_num_updates
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates

            def tast_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                task_outputa = self.forward(inputa, weights, bias, reuse=reuse)
                task_lossa = self.inside_lose(task_outputa, labela)

                grads = tf.gradients(task_lossa, weights)
                gradsb = tf.gradients(task_lossa, bias)
                cal_grads = []
                calb_grads = []
                for h in grads:
                    cal_grads.append(FLAGS.meta_lr*h)
                for h in gradsb:
                    calb_grads.append(FLAGS.meta_lr*h)

                out_weights = fast_weights = weights - cal_grads
                out_bias = fast_bias = bias - calb_grads
                fast_weights = tf.reshape(fast_weights, [FLAGS.time_series_length, 1])
                fast_bias = tf.reshape(fast_bias, [FLAGS.num_indicators, 1])
                output = self.forward(inputb, fast_weights, fast_bias, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(utils.abs_lose(output, labelb))

                for j in range(num_updates - 1):
                    cal_grads = []
                    calb_grads = []
                    loss = self.inside_lose(self.forward(inputa, fast_weights, fast_bias, reuse=True), labela)
                    grads = tf.gradients(loss, weights)
                    gradsb = tf.gradients(task_lossa, bias)
                    for h in grads:
                        cal_grads.append(FLAGS.meta_lr * h)
                    for h in gradsb:
                        calb_grads.append(FLAGS.meta_lr * h)
                    out_weights = fast_weights = fast_weights - cal_grads
                    fast_bias = bias - calb_grads
                    fast_weights = tf.reshape(fast_weights, [FLAGS.time_series_length, 1])
                    out_bias = fast_bias = tf.reshape(fast_bias, [FLAGS.num_indicators, 1])
                    output = self.forward(inputb, fast_weights, fast_bias,reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(utils.abs_lose(output, labelb))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, tf.reshape(out_weights, [FLAGS.time_series_length,1]), out_bias]
                return task_output

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates, tf.float32,tf.float32]
            result = tf.map_fn(tast_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=self.batch_size)
            outputas, outputbs, lossesa, lossesb, all_weights, all_bias = result

        if 'train' in prefix:

            self.all_weights = all_weights = tf.transpose(tf.squeeze(all_weights, axis=2))
            self.all_bias = all_bias = tf.transpose(tf.squeeze(all_bias, axis=2))
            self.all_parameters= all_parameters = tf.concat([all_weights,all_bias], 0)
            #self.total_loss1 = total_loss1 = (tf.reduce_sum(lossesa) + outside_loss(all_weights))/tf.to_float(self.batch_size)
            self.total_loss1 = total_loss1 = (tf.reduce_sum(lossesa)) / tf.to_float(
                self.batch_size)/tf.to_float(FLAGS.num_indicators)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / (tf.to_float(self.batch_size))/tf.to_float(FLAGS.num_indicators) for j
                                                  in range(num_updates)]
            #self.test = self.total_losses2
            self.outputas, self.outputbs = outputas, outputbs
            self.outputas, self.outputbs = outputas, outputbs
            #self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1+utils.loss_function_l2(all_parameters))
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[num_updates-1])
                self.metatrain_op = optimizer.apply_gradients(gvs)
            else:
                self.metaval_total_loss1 = total_loss1 = (tf.reduce_sum(lossesa))/tf.to_float(self.batch_size)
                self.metaval_total_loss2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(self.batch_size) for j
                                                      in range(num_updates)]

            tf.summary.scalar(prefix+'pre-update loss', total_loss1)
            for j in range(num_updates):
                tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])

    def forward_fc(self, inp, weight, reuse=False):
        return tf.matmul(inp, weight, reuse=reuse)

    def construct_fc_weight(self):
        weights = tf.Variable(tf.truncated_normal([FLAGS.time_series_length, 1], stddev=0.01))
        bias = tf.Variable(tf.truncated_normal([FLAGS.num_indicators, 1], stddev=0.01))
        return weights, bias
