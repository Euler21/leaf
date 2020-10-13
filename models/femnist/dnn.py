import numpy as np
import tensorflow as tf

from model import Model
from model.utils.tf_utils import get_dtype_with_precision


IMAGE_SIZE = 28

class ClientModel(Model):
    def __init__(self, seed, lr, num_classes, model_precision='float32', **kwargs):
        self.num_classes = num_classes
        self.dtype = get_dtype_with_precision(model_precision)
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            self.dtype, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        dense1 = tf.layers.dense(inputs=features, units=64, activation=tf.nn.sigmoid)
        dense2 = tf.layers.dense(inputs=dense1, units=64, activation=tf.nn.sigmoid)
        logits = tf.layers.dense(inputs=dense2, units=self.num_classes)
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
