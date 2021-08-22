import tensorflow as tf
import numpy as np

#Loss function

class MSE:
  def __call__(self, real,preds,model=None,X=None):
        return tf.losses.mean_squared_error(labels=real, predictions=preds)


# niewydajne
class LogReg:
  def __call__(self, real,preds,model=None,X=None):
#        ll =  real*tf.log(preds + 1e5) + (1-real)*tf.log(1-preds + 1e5)
        ll =  tf.reduce_mean(real*tf.log(preds) + (1-real)*tf.log(1-preds + 1e5))
        return -ll
#        return tf.losses.mean_squared_error(labels=real, predictions=preds)

#bardzo niewydajne
class SigmoidCrossEntropy:
  def __call__(self, real,preds,model=None,X=None):
#        ll =  real*tf.log(preds + 1e5) + (1-real)*tf.log(1-preds + 1e5)
#        real = tf.reshape(real,[tf.size(real),1])
#        preds = tf.reshape(preds,[tf.size(preds),1])
        ll =  tf.losses.sigmoid_cross_entropy(multi_class_labels=real, logits=preds)
        return ll

class LogLoss:
  def __call__(self, real,preds,model=None,X=None):
#        ll =  real*tf.log(preds + 1e5) + (1-real)*tf.log(1-preds + 1e5)
#        real = tf.reshape(real,[tf.size(real),1])
#        preds = tf.reshape(preds,[tf.size(preds),1])
#         ll =  tf.losses.log_loss(labels=real, predictions=preds)
        ll =  tf.compat.v1.losses.log_loss(labels=real, predictions=preds)
        return ll

class SoftmaxCrossEntropy:
  def __call__(self, real,preds,model=None,X=None):
#        ll =  real*tf.log(preds + 1e5) + (1-real)*tf.log(1-preds + 1e5)
#        real = tf.reshape(real,[tf.size(real),1])
#        preds = tf.reshape(preds,[tf.size(preds),1])
        # FIXED
        # https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/softmax_cross_entropy
        # ll = tf.losses.softmax_cross_entropy(onehot_labels=real, logits=preds)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        ll = loss_fn(y_true=real, y_pred=preds)
        return ll
