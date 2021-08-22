from sklearn.base import BaseEstimator, ClassifierMixin
from base.classifier import BaseFcmClassifier
from base.losses import MSE, SoftmaxCrossEntropy, LogLoss
from base.model import Sigmoid
import tensorflow as tf
import sklearn.preprocessing
import numpy as np


class FcmBinaryClassifier(BaseEstimator, ClassifierMixin):
  def __init__(self,
               act=None,
               activation='sigmoid',
               activation_m=1,
               depth=5,
               epochs=100,
               batch_size=10,
               buffer_size=1000,
               training_loss='logloss',
               optimizer='rmsprop',
               learning_rate=0.001,
               model_type='basic',
               init_method = None,
               init_y_method = 'uniform',
               dropout_p=0.0,
               training_hook=[],
               # report_epoch_freq=-1,
               random_state=None):
    self.act = act
    self.activation = activation
    self.activation_m = activation_m
    self.depth = depth
    self.batch_size = batch_size
    self.epochs = epochs
    self.buffer_size = buffer_size
    self.training_loss = training_loss
    self.optimizer = optimizer
    self.learning_rate=learning_rate
    self.model_type = model_type
    self.init_method = init_method
    self.init_y_method = init_y_method
    self.training_hook = training_hook
    self.dropout_p = dropout_p
    # self.report_epoch_freq = report_epoch_freq
    self.random_state=random_state

    self.base_fcm = None

  def fit(self, X, y, sample_weight=None):
    n_labels = np.max(y) + 1
    if n_labels >2:
      raise ValueError('This is a binary classifier. Use a multiclass version')
    for i in range(5):
      print('-----------------------------------------')
    print(self.get_params())
    print('-----------------------------------------')
    self.base_fcm = BaseFcmClassifier(**self.get_params())
    self.base_fcm.fit(X, y.reshape(y.size,1))
    return self

  def fit_transform(self,X,y):
    self.fit(X,y)
    return self.base_fcm.transform_features(X)

  def transform(self,X,y=None):
    return self.base_fcm.transform_features(X)

  def predict_proba(self, X):
    y_pred = self.base_fcm.predict_ranks(X)
    return y_pred

  def predict(self, X):
    r = self.base_fcm.predict_ranks(X)
    y = np.where(r>=0.5,1,0)
    return y


  def score(self, X, y, sample_weight=None):
    return self.base_fcm.score(X, y)



# ----------------------------------------------------------------------------------------------------------------------
# Some utilities
# ----------------------------------------------------------------------------------------------------------------------


