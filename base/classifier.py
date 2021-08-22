import warnings

import tensorflow as tf
import tensorflow as tf
import numpy as np
import time

from sklearn.base import BaseEstimator
from tensorflow import keras

from base.losses import MSE, LogLoss, SoftmaxCrossEntropy
from base.model import Model, AdditiveModel, Sigmoid, Tanh, Ident, Relu

# tf.enable_eager_execution()
#print(tf.executing_eagerly())
# tfe = tf.contrib.eager




class BaseFcmClassifier(object):
  def __init__(self, **kwargs):
    """self.act is deprecated"""
    self.act = kwargs.get('act',None)
    self.activation = kwargs.get('activation','sigmoid')
    self.activation_m = kwargs.get('activation_m',1.0)


    self.depth = kwargs.get('depth', 5)
    self.batch_size = kwargs.get('batch_size', 10)
    self.epochs = kwargs.get('epochs', 100)
    self.buffer_size = kwargs.get('buffer_size', 1000)
    self.random_state = kwargs.get('random_state', None)
    self.optimizer = kwargs.get('optimizer', tf.compat.v1.train.RMSPropOptimizer(0.001))
    self.learning_rate = kwargs.get('learning_rate', 0.001)
    if self.optimizer=='rmsprop':
      self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate)
    if self.optimizer=='adam':
      self.optimizer = tf.compat.v1.train.AdamOptimizer()


    self.history = []
    self.training_time = 0
    self.model_type = kwargs.get('model_type', 'basic')  # or 'additive'
    self.init_method = kwargs.get('init_method', None)
    self.model = None
    self.current_gradient = None

    self.traing_hook = kwargs.get('training_hook', None)
    if self.traing_hook is not None and not hasattr(self.traing_hook, '__iter__'):
      self.traing_hook = [self.traing_hook]

    self.training_loss = kwargs.get('training_loss', MSE())
    if self.training_loss=='mse':
      self.training_loss=MSE()
    if self.training_loss=='logloss':
      self.training_loss=LogLoss()
    if self.training_loss=='softmax':
      self.training_loss=SoftmaxCrossEntropy()

    self.init_y_method = kwargs.get('init_y_method', 'uniform')
    self.dropout_p = kwargs.get('dropout_p', 0)
    if self.dropout_p < 0:
      self.dropout_p = 0
    if self.dropout_p > 1:
      self.dropout_p = 1
    self.apply_dropout = False
    if self.dropout_p != 0:
      self.apply_dropout = True
      raise NotImplementedError
    # self.report_epoch_freq = kwargs.get('report_epoch_freq', -1)

  def get_init_y(self, X, Y):
    if self.init_y_method == 'equal':
      init_y = tf.constant(np.ones((self.batch_size, Y.shape[1])) * .5)
    elif self.init_y_method == 'frequency':
      init_y = np.sum(Y, axis=0) / Y.shape[0]
      init_y = np.broadcast_to(init_y, (self.batch_size, Y.shape[1]))
      init_y = tf.constant(init_y)
    else:
      init_y = tf.constant(np.random.rand(self.batch_size, Y.shape[1]))
    return init_y


  def _get_activation_function(self):
    if self.act is not None:
      return self.act

    if self.activation == 'sigmoid':
      return Sigmoid(self.activation_m)
    elif self.activation == 'tanh':
      return Tanh()
    elif self.activation == 'ident':
      return Ident()
    elif self.activation == 'relu':
      return Relu()
    else:
      raise ValueError('Not recognized activation function: '+self.activation)


  def fit(self, X, Y):
    if self.random_state is not None:
      tf.random.set_seed(self.random_state)
      np.random.seed(self.random_state)

    if self.batch_size == -1:
      self.batch_size = X.shape[0]
    if self.batch_size > X.shape[0]:
      self.batch_size = X.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(self.buffer_size)
    dataset = dataset.batch(self.batch_size, drop_remainder=True)
    init_y = self.get_init_y(X, Y)
    #    self.model = Model(dim = X.shape[1]+Y.shape[1],act = self.g,depth=self.depth)
    if self.model_type == 'basic':
      self.model = Model(dim=X.shape[1] + Y.shape[1], act=self._get_activation_function(), depth=self.depth,init_method=self.init_method)
    elif self.model_type == 'additive':
      self.model = AdditiveModel(dim=X.shape[1] + Y.shape[1], act=self._get_activation_function(), depth=self.depth,init_method=self.init_method)
    else:
      raise ValueError('Unknown model type: {}'.format(self.model_type))
    self.train(self.model, dataset, init_y)


  def train(self, model, dataset, init_y):

    self.history.clear()
    start = time.time()
    loss = 0
    for epoch in range(self.epochs):
      total_epoch_loss = 0
      for (batch, (inp, target)) in enumerate(dataset):
        #        print('.')
        if self.apply_dropout:
          pass
          # tf.set_random_seed(123)

        with tf.GradientTape() as tape:
          #          print(model.act)
          preds = model(inp, init_y)

          loss = self.training_loss(target, preds, model)
          # loss = loss/self.batch_size
          warnings.warn('2018.12.16 The instruction  \'loss = loss/self.batch_size\' was commented out. Due to increased gradient values check the optimizer learning rate ')
          self.history.append(loss)
          total_epoch_loss = total_epoch_loss + loss
          variables = (model.W, model.b)
          self.current_gradient = tape.gradient(loss, variables)
          if self.apply_dropout:
            pass
          if self.traing_hook is not None:
            for f in self.traing_hook:
              f(epoch, batch, self, loss)
          self.optimizer.apply_gradients(zip(self.current_gradient, variables))
          self.current_gradient=None
          # print(grads[0][0][0])

          #?????????
      total_epoch_loss = total_epoch_loss / (batch + 1)

          # if self.report_epoch_freq > 0 and (epoch + 1) % self.report_epoch_freq == 0:
          #   print('Epoch {} Loss {:.4f} (last batch {: 4f})'.format(epoch + 1, total_epoch_loss, loss))
      if self.traing_hook is not None:
        for f in self.traing_hook:
          f(epoch, -1, self, total_epoch_loss)

    print('Learning time {} sec\n'.format(time.time() - start))
    return {'history': np.array(self.history)}

  def predict_ranks(self, X):
    init_y = np.ones((X.shape[0], self.model.dim - X.shape[1])) * .5
    return self.model(X, init_y).numpy()

  def predict(self, X):
    ranks = self.predict_ranks(X)
    return np.where(ranks < 0.5, 0.0, 1.0)

  def score(self, X, Y):
    init_y = self.get_init_y(X, Y)
    preds = self.model(X, init_y)
    loss = self.training_loss(Y, preds, self.model)
    return loss * -1.0

  def transform_features(self,X,feature_index=-2):
    # TODO get_init_y
    init_y = np.ones((X.shape[0], self.model.dim - X.shape[1])) * .5
    trj = self.model.trajectory(X, init_y)
    Z = trj[feature_index]
    return Z

