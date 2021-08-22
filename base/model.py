import tensorflow as tf
import numpy as np
import time

from sklearn.base import BaseEstimator
from tensorflow import keras


# tf.enable_eager_execution()
#print(tf.executing_eagerly())
# tfe = tf.contrib.eager


class Model(object):
  """Fuzzy Cognitive Maps model

  """

  def __init__(self, **kwargs):
    # print(kwargs)
    self.dim = kwargs.get('dim')
    self.act = kwargs.get('act')
    self.depth = kwargs.get('depth', 5)
    self.init_method = kwargs.get('init_method', None)

    random_state = kwargs.get('random_state')
    if random_state is not None:
      np.random.seed(random_state)

    weights = kwargs.get('weights')

    if weights is None:
      w,b = self._generate_inital_weights()
    else:
      w=weights[0]
      b=weights[1]
      self.dim = w.shape[0]


    if (w.shape[0] != w.shape[1]):
      raise ValueError('Expecting square matrix W, while the shape is (%d,%d)' % w.shape)
    if (w.shape[0] != b.shape[0]):
      raise ValueError('W and b shapes should be equal')


    self.W = tf.Variable(w)
    self.b = tf.Variable(b.reshape(b.size, 1))

  def _generate_inital_weights(self):
    if self.init_method == 'random':
      w = np.random.randn(self.dim, self.dim) * 0.01
      b = np.random.randn(self.dim) * 0.01
    if self.init_method == 'xavier':
      w = np.random.randn(self.dim, self.dim) * np.sqrt(1 / self.dim)
      b = np.random.randn(self.dim) * np.sqrt(1 / self.dim)
    if self.init_method == 'ho':
      w = np.random.randn(self.dim, self.dim) * np.sqrt(2 / self.dim)
      b = np.random.randn(self.dim) * np.sqrt(2 / self.dim)
    if self.init_method == 'depth':
        w = np.random.randn(self.dim, self.dim) * (1 / self.dim)**(1/self.depth)
        b = np.random.randn(self.dim) * (1 / self.dim)**(1/self.depth)
    else:
      w = np.random.rand(self.dim, self.dim) * 2 - 1
      b = np.random.rand(self.dim) * 2 - 1
    return w,b


  def __call__(self, X, Y, return_X=False):
    Z = tf.concat([X, Y], axis=1)
    for i in range(self.depth):
      Z = tf.matmul(self.W, Z, transpose_b=True) + self.b
      Z = self.act(Z)
      Z = tf.transpose(Z)
    Z = Z[:, X.shape[1]:]
    #         Z=tf.exp(Z)
    #         Z=Z/tf.reduce_sum(Z,axis=1,keepdims=True)
    if return_X:
      return Z[:, :X.shape[1]], Z
    else:
      return Z

  def trajectory(self, X, Y):
    trajectory = []
    Z = tf.concat([X, Y], axis=1)
    trajectory.append(Z.numpy())
    for i in range(self.depth):
      Z = tf.matmul(self.W, Z, transpose_b=True) + self.b
      Z = self.act(Z)
      Z = tf.transpose(Z)
      trajectory.append(Z.numpy())
    return trajectory


class AdditiveModel(object):
  """Fuzzy Cognitive Maps model

  """

  def __init__(self, **kwargs):
    self.dim = kwargs.get('dim')
    self.act = kwargs.get('act')
    self.depth = kwargs.get('depth', 5)
    w = kwargs.get('weights')
    b = kwargs.get('biases')
    if w is None:
      w = np.random.rand(self.dim, self.dim) * 2 - 1
      self.dim = w.shape[0]

    if b is None:
      b = np.random.rand(self.dim) * 2 - 1

    if (w.shape[0] != w.shape[1]):
      raise ValueError('Expecting square matrix W, while the shape is (%d,%d)' % w.shape)
    if (w.shape[0] != b.shape[0]):
      raise ValueError('W and b shapes should be equal')

    random_state = kwargs.get('random_state')
    #         if random_state is not None:
    np.random.seed(random_state)

    self.W = tfe.Variable(w)
    self.b = tfe.Variable(b.reshape(b.size, 1))
    raise ValueError('Not to be used')

  def __call__(self, X, Y):
    Z = tf.concat([X, Y], axis=1)
    for i in range(self.depth):
      Z_c = tf.matmul(self.W, Z, transpose_b=True) + self.b
      #          Z_c = self.g(Z_c)
      Z_c = tf.transpose(Z_c)
      Z = Z + Z_c
      Z = self.act(Z)
    Z = Z[:, X.shape[1]:]
    #         Z=tf.exp(Z)
    #         Z=Z/tf.reduce_sum(Z,axis=1,keepdims=True)
    return Z

  def trajectory(self, X, Y):
    trajectory = []
    Z = tf.concat([X, Y], axis=1)
    for i in range(self.depth):
      Z_c = tf.matmul(self.W, Z, transpose_b=True) + self.b
      #          Z_c = self.g(Z_c)
      Z_c = tf.transpose(Z_c)
      Z = Z + Z_c
      Z = self.act(Z)
      trajectory.append(Z.numpy())
    return trajectory


# Acivation functions

class Ident:
  def __call__(self, X):
    return X


class Relu:
  def __call__(self, X):
    return tf.nn.relu(X)


class Tanh:
  """Tanh scaled to map 0.5->0.5
  """

  def __call__(self, X):
    return tf.tanh(X) / 2 + .5


class Sigmoid:
  """Sigmoid scaled to map 0.5->0.5
     Attributes:
         m - slope constant
  """
  def __init__(self, m=1.0):
    self.m = m

  def __repr__(self):
    return super().__repr__()+'[m={}]'.format(self.m)

  def __call__(self, X):
    return 1 / (1 + tf.exp(-self.m * (X - 0.5)))


class SinExpWavelet:
  def __init__(self,a=1.0,b=1.0):
    self.a = a
    self.b = b
    self.w = 1
    x=np.linspace(-5,5,1000)
    y=self(x)
    y_min = np.min(y)
    y_max = np.max(y)
    # imax = np.argmax(y)
    # imin = np.argmin(y)
    # print('min = {} at {}'.format(y_min,x[imin]))
    # print('max = {} at {}'.format(y_max,x[imax]))
    self.w=(1/(y_max-y_min))
    # print(self.w)

  def __repr__(self):
    return super().__repr__()+'[a={} b={}]'.format(self.a,self.b)

  def __call__(self, X):
    return self.w*tf.sin(self.a*2*np.pi*(X-0.5))*tf.exp(-self.b*2*np.pi*(X-0.5)**2)+0.5
