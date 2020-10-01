# OtterTune - nn_tf.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Sep 16, 2019
@author: Bohan Zhang
'''

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .util import get_analysis_logger

LOG = get_analysis_logger(__name__)


class NeuralNetResult(object):
    def __init__(self, minl=None, minl_conf=None):
        self.minl = minl
        self.minl_conf = minl_conf


class NeuralNet(object):

    def __init__(self,
                 n_input,
                 include_context=False,
                 n_context=None,
                 learning_rate=0.01,
                 debug=False,
                 debug_interval=100,
                 batch_size=1,
                 explore_iters=500,
                 noise_scale_begin=0.1,
                 noise_scale_end=0,
                 reset_seed=False):

        self.history = None
        self.recommend_iters = 0
        self.n_input = n_input
        self.include_context = include_context
        self.n_context = n_context
        self.debug = debug
        self.debug_interval = debug_interval
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.explore_iters = explore_iters
        self.noise_scale_begin = noise_scale_begin
        self.noise_scale_end = noise_scale_end
        self.vars = {}
        self.ops = {}

        tf.reset_default_graph()
        if reset_seed:
            tf.set_random_seed(0)
        self.session = tf.Session()
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():   # pylint: disable=not-context-manager
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                # input X is placeholder, weights are variables.
                if self.include_context:
                    knobs_in = keras.layers.Input(shape=(n_input, ))
                    knobs_out = keras.layers.Dense(32, activation=tf.nn.relu)(knobs_in)
                    context_in = keras.layers.Input(shape=(n_context, ))
                    context_out = keras.layers.Dense(32, activation=tf.nn.relu)(context_in)
                    merged = keras.layers.concatenate([knobs_out, context_out])
                    l1_out = keras.layers.Dropout(0.5)(merged)
                    l2_out = keras.layers.Dense(64, activation=tf.nn.relu)(l1_out)
                    l3_out = keras.layers.Dense(1)(l2_out)
                    self.model = keras.models.Model([knobs_in, context_in], l3_out)
                else:
                    self.model = keras.Sequential([
                        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[n_input]),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(64, activation=tf.nn.relu),
                        keras.layers.Dense(1)
                    ])
                self.model.compile(loss='mean_squared_error',
                                   optimizer=self.optimizer,
                                   metrics=['mean_squared_error', 'mean_absolute_error'])
        self._build_graph()

    def save_weights_file(self, weights_file):
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                self.model.save_weights(weights_file)

    def load_weights_file(self, weights_file):
        try:
            with self.graph.as_default():
                with self.session.as_default():  # pylint: disable=not-context-manager
                    self.model.load_weights(weights_file)
            if self.debug:
                LOG.info('Neural Network Model weights file exists, load weights from the file')
        except Exception:  # pylint: disable=broad-except
            LOG.info('Weights file does not match neural network model, train model from scratch')

    def get_weights_bin(self):
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                weights = self.model.get_weights()
                return pickle.dumps(weights)

    def set_weights_bin(self, weights):
        try:
            with self.graph.as_default():
                with self.session.as_default():  # pylint: disable=not-context-manager
                    self.model.set_weights(pickle.loads(weights))
            if self.debug:
                LOG.info('Neural Network Model weights exists, load the existing weights')
        except Exception:  # pylint: disable=broad-except
            LOG.info('Weights does not match neural network model, train model from scratch')

    # Build same neural network as self.model, But input X is variables,
    # weights are placedholders. Find optimial X using gradient descent.
    def _build_graph(self):
        batch_size = self.batch_size
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                x = tf.Variable(tf.ones([batch_size, self.n_input]))
                X_min = tf.placeholder(tf.float32, [self.n_input])
                X_max = tf.placeholder(tf.float32, [self.n_input])
                x_bounded = tf.minimum(x, X_max)
                x_bounded = tf.maximum(x_bounded, X_min)
                x_bounded = tf.cast(x_bounded, tf.float32)

                if self.include_context:
                    # knob input
                    w1 = tf.placeholder(tf.float32, [self.n_input, 32])
                    b1 = tf.placeholder(tf.float32, [32])
                    l11 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

                    # context input
                    xc = tf.placeholder(tf.float32, [batch_size, self.n_context])  # placeholder
                    wc = tf.placeholder(tf.float32, [self.n_context, 32])
                    bc = tf.placeholder(tf.float32, [32])
                    l12 = tf.nn.relu(tf.add(tf.matmul(xc, wc), bc))

                    # merged
                    l1 = tf.concat([l11, l12], 1)

                    self.vars['xc'] = xc
                    self.vars['wc'] = wc
                    self.vars['bc'] = bc

                else:
                    w1 = tf.placeholder(tf.float32, [self.n_input, 64])
                    b1 = tf.placeholder(tf.float32, [64])
                    l1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))

                w2 = tf.placeholder(tf.float32, [64, 64])
                b2 = tf.placeholder(tf.float32, [64])
                w3 = tf.placeholder(tf.float32, [64, 1])
                b3 = tf.placeholder(tf.float32, [1])
                l2 = tf.nn.relu(tf.add(tf.matmul(l1, w2), b2))
                y = tf.add(tf.matmul(l2, w3), b3)
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train = optimizer.minimize(y)

                self.vars['x'] = x
                self.vars['y'] = y
                self.vars['w1'] = w1
                self.vars['w2'] = w2
                self.vars['w3'] = w3
                self.vars['b1'] = b1
                self.vars['b2'] = b2
                self.vars['b3'] = b3
                self.vars['X_min'] = X_min
                self.vars['X_max'] = X_max
                self.vars['x_bounded'] = x_bounded
                self.ops['train'] = train

    def _fit_with_context(self, X_train, y_train, fit_epochs, X_context):
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                self.history = self.model.fit(
                    [X_train, X_context], y_train, epochs=fit_epochs, verbose=0)
                if self.debug:
                    mse = self.history.history['mean_squared_error']
                    i = 0
                    size = len(mse)
                    while(i < size):
                        LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                                 i, mse[i])
                        i += self.debug_interval
                    LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                             size - 1, mse[size - 1])

    def _fit_without_context(self, X_train, y_train, fit_epochs):
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                self.history = self.model.fit(
                    X_train, y_train, epochs=fit_epochs, verbose=0)
                if self.debug:
                    mse = self.history.history['mean_squared_error']
                    i = 0
                    size = len(mse)
                    while(i < size):
                        LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                                 i, mse[i])
                        i += self.debug_interval
                    LOG.info("Neural network training phase, epoch %d: mean_squared_error %f",
                             size - 1, mse[size - 1])

    def fit(self, X_train, y_train, fit_epochs=500, X_context=None):
        if self.include_context:
            self._fit_with_context(X_train, y_train, fit_epochs, X_context)
        else:
            self._fit_without_context(X_train, y_train, fit_epochs)

    def predict(self, X_pred, X_context=None):
        with self.graph.as_default():
            with self.session.as_default():  # pylint: disable=not-context-manager
                if self.include_context:
                    return self.model.predict([X_pred, X_context])
                return self.model.predict(X_pred)

    # Reference: Parameter Space Noise for Exploration.ICLR 2018, https://arxiv.org/abs/1706.01905
    def _add_noise(self, weights):
        scale = self._adaptive_noise_scale()
        size = weights.shape[-1]
        noise = scale * np.random.normal(size=size)
        return weights + noise

    def _adaptive_noise_scale(self):
        if self.recommend_iters > self.explore_iters:
            scale = self.noise_scale_end
        else:
            scale = self.noise_scale_begin - (self.noise_scale_begin - self.noise_scale_end) \
                * 1.0 * self.recommend_iters / self.explore_iters
        return scale

    def _recommend_with_context(self, X_start, X_min, X_max, recommend_epochs, explore, X_context):
        batch_size = len(X_start)
        assert(batch_size == self.batch_size)
        assert(batch_size == len(X_context))
        if X_min is None:
            X_min = np.tile([-np.infty], self.n_input)
        if X_max is None:
            X_max = np.tile([np.infty], self.n_input)

        with self.graph.as_default():
            with self.session.as_default() as sess:  # pylint: disable=not-context-manager
                w1, b1 = self.model.get_layer(index=2).get_weights()
                wc, bc = self.model.get_layer(index=3).get_weights()
                w2, b2 = self.model.get_layer(index=6).get_weights()
                w3, b3 = self.model.get_layer(index=7).get_weights()

                if explore is True:
                    w1 = self._add_noise(w1)
                    b1 = self._add_noise(b1)
                    wc = self._add_noise(wc)
                    bc = self._add_noise(bc)
                    w2 = self._add_noise(w2)
                    b2 = self._add_noise(b2)
                    w3 = self._add_noise(w3)
                    b3 = self._add_noise(b3)

                y_predict = self.predict(X_pred=X_start, X_context=X_context)
                if self.debug:
                    LOG.info("Recommend phase, y prediction: min %f, max %f, mean %f",
                             np.min(y_predict), np.max(y_predict), np.mean(y_predict))

                init = tf.global_variables_initializer()
                sess.run(init)
                assign_x_op = self.vars['x'].assign(X_start)
                sess.run(assign_x_op)
                y_before = sess.run(self.vars['y'],
                                    feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                               self.vars['w3']: w3, self.vars['b1']: b1,
                                               self.vars['b2']: b2, self.vars['b3']: b3,
                                               self.vars['xc']: X_context,
                                               self.vars['wc']: wc, self.vars['bc']: bc,
                                               self.vars['X_max']: X_max,
                                               self.vars['X_min']: X_min})

                if self.debug:
                    LOG.info("Recommend phase, y before gradient descent: min %f, max %f, mean %f",
                             np.min(y_before), np.max(y_before), np.mean(y_before))

                for i in range(recommend_epochs):
                    sess.run(self.ops['train'],
                             feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                        self.vars['w3']: w3, self.vars['b1']: b1,
                                        self.vars['b2']: b2, self.vars['b3']: b3,
                                        self.vars['xc']: X_context,
                                        self.vars['wc']: wc, self.vars['bc']: bc,
                                        self.vars['X_max']: X_max, self.vars['X_min']: X_min})

                    if self.debug and i % self.debug_interval == 0:
                        y_train = sess.run(self.vars['y'],
                                           feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                                      self.vars['w3']: w3, self.vars['b1']: b1,
                                                      self.vars['b2']: b2, self.vars['b3']: b3,
                                                      self.vars['xc']: X_context,
                                                      self.vars['wc']: wc, self.vars['bc']: bc,
                                                      self.vars['X_max']: X_max,
                                                      self.vars['X_min']: X_min})
                        LOG.info("Recommend phase, epoch %d, y: min %f, max %f, mean %f",
                                 i, np.min(y_train), np.max(y_train), np.mean(y_train))

                y_recommend = sess.run(self.vars['y'],
                                       feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                                  self.vars['w3']: w3, self.vars['b1']: b1,
                                                  self.vars['b2']: b2, self.vars['b3']: b3,
                                                  self.vars['xc']: X_context,
                                                  self.vars['wc']: wc, self.vars['bc']: bc,
                                                  self.vars['X_max']: X_max,
                                                  self.vars['X_min']: X_min})

                X_recommend = sess.run(self.vars['x_bounded'],
                                       feed_dict={self.vars['X_max']: X_max,
                                                  self.vars['X_min']: X_min})
                res = NeuralNetResult(minl=y_recommend, minl_conf=X_recommend)

                if self.debug:
                    LOG.info("Recommend phase, epoch %d, y after gradient descent: \
                             min %f, max %f, mean %f", recommend_epochs, np.min(y_recommend),
                             np.max(y_recommend), np.mean(y_recommend))

                self.recommend_iters += 1
                return res

    def _recommend_without_context(self, X_start, X_min, X_max, recommend_epochs, explore):
        batch_size = len(X_start)
        assert(batch_size == self.batch_size)
        if X_min is None:
            X_min = np.tile([-np.infty], self.n_input)
        if X_max is None:
            X_max = np.tile([np.infty], self.n_input)

        with self.graph.as_default():
            with self.session.as_default() as sess:  # pylint: disable=not-context-manager
                w1, b1 = self.model.get_layer(index=0).get_weights()
                w2, b2 = self.model.get_layer(index=2).get_weights()
                w3, b3 = self.model.get_layer(index=3).get_weights()

                if explore is True:
                    w1 = self._add_noise(w1)
                    b1 = self._add_noise(b1)
                    w2 = self._add_noise(w2)
                    b2 = self._add_noise(b2)
                    w3 = self._add_noise(w3)
                    b3 = self._add_noise(b3)

                y_predict = self.predict(X_start)
                if self.debug:
                    LOG.info("Recommend phase, y prediction: min %f, max %f, mean %f",
                             np.min(y_predict), np.max(y_predict), np.mean(y_predict))

                init = tf.global_variables_initializer()
                sess.run(init)
                assign_x_op = self.vars['x'].assign(X_start)
                sess.run(assign_x_op)
                y_before = sess.run(self.vars['y'],
                                    feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                               self.vars['w3']: w3, self.vars['b1']: b1,
                                               self.vars['b2']: b2, self.vars['b3']: b3,
                                               self.vars['X_max']: X_max,
                                               self.vars['X_min']: X_min})
                if self.debug:
                    LOG.info("Recommend phase, y before gradient descent: min %f, max %f, mean %f",
                             np.min(y_before), np.max(y_before), np.mean(y_before))

                for i in range(recommend_epochs):
                    sess.run(self.ops['train'],
                             feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                        self.vars['w3']: w3, self.vars['b1']: b1,
                                        self.vars['b2']: b2, self.vars['b3']: b3,
                                        self.vars['X_max']: X_max, self.vars['X_min']: X_min})

                    if self.debug and i % self.debug_interval == 0:
                        y_train = sess.run(self.vars['y'],
                                           feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                                      self.vars['w3']: w3, self.vars['b1']: b1,
                                                      self.vars['b2']: b2, self.vars['b3']: b3,
                                                      self.vars['X_max']: X_max,
                                                      self.vars['X_min']: X_min})
                        LOG.info("Recommend phase, epoch %d, y: min %f, max %f, mean %f",
                                 i, np.min(y_train), np.max(y_train), np.mean(y_train))

                y_recommend = sess.run(self.vars['y'],
                                       feed_dict={self.vars['w1']: w1, self.vars['w2']: w2,
                                                  self.vars['w3']: w3, self.vars['b1']: b1,
                                                  self.vars['b2']: b2, self.vars['b3']: b3,
                                                  self.vars['X_max']: X_max,
                                                  self.vars['X_min']: X_min})
                X_recommend = sess.run(self.vars['x_bounded'],
                                       feed_dict={self.vars['X_max']: X_max,
                                                  self.vars['X_min']: X_min})
                res = NeuralNetResult(minl=y_recommend, minl_conf=X_recommend)

                if self.debug:
                    LOG.info("Recommend phase, epoch %d, y after gradient descent: \
                             min %f, max %f, mean %f", recommend_epochs, np.min(y_recommend),
                             np.max(y_recommend), np.mean(y_recommend))

                self.recommend_iters += 1
                return res

    def recommend(self, X_start, X_min=None, X_max=None, recommend_epochs=500,
                  explore=False, X_context=None):
        if self.include_context:
            return self._recommend_with_context(X_start, X_min, X_max, recommend_epochs,
                                                explore, X_context)
        return self._recommend_without_context(X_start, X_min, X_max, recommend_epochs, explore)
