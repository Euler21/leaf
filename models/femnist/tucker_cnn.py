import tensorflow as tf
import t3f

from model import Model
import numpy as np


IMAGE_SIZE = 28


class TuckerLayer(tf.keras.layers.Layer):
  def __init__(self, input_shape, output_shape):
    super(TuckerLayer, self).__init__()

    initializer = tf.keras.initializers.glorot_normal()
    # self.factors = [tf.Variable(initializer(shape=[input_shape[i], output_shape[i]])) for i in range(len(input_shape))]
    self.f0 = tf.Variable(initializer(shape=[input_shape[0], output_shape[0]]))
    self.f1 = tf.Variable(initializer(shape=[input_shape[1], output_shape[1]]))
    self.f2 = tf.Variable(initializer(shape=[input_shape[2], output_shape[2]]))



  # def build(self, input_shape):
  #  self.kernel = self.add_weight("kernel",
  #                                shape=[int(input_shape[-1]),
  #                                       self.num_outputs])

  def call(self, itensor):
        x1 = tf.tensordot(itensor, self.f0, axes=[[1], [0]])
        x2 = tf.tensordot(x1, self.f1, axes=[[1],[0]]) 
        x3 = tf.tensordot(x2, self.f2, axes=[[1], [0]])
        return tf.nn.relu(x3)      


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, tucker_rank):
        self.num_classes = num_classes
        self.tucker_rank = tucker_rank
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        print("Creating TT-CNN!")
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense layer replacement with a Tucker-inspired layer 
        tucker = TuckerLayer([7, 7, 64], [5, 5, 40]) 
        t_out = tucker(pool2) 
        t_out_flat = tf.reshape(t_out, [-1, 5 * 5 * 40])

        # denseLayer = t3f.nn.KerasDense([14, 14, 16], [16, 16, 8], activation=tf.nn.relu, tt_rank=self.dense_rank)
        logits = tf.layers.dense(inputs=t_out_flat, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # TODO: Confirm that opt initialized once is ok?
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
