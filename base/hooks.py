import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------------------------------------------------
# Training hooks
# ----------------------------------------------------------------------------------------------------------------------
from sklearn.utils import Bunch


class TrainingHook(object):
  def __init__(self, **kwargs):
    self.batch_interval = kwargs.get('batch_interval', -1)
    self.epoch_interval = kwargs.get('epoch_interval', -1)

  def test_execute(self, epoch, batch, classifier, training_loss):
    if batch != -1 and self.batch_interval != -1 and batch % self.batch_interval == 0:
      return True
    if batch == -1 and self.epoch_interval != -1 and epoch % self.epoch_interval == 0:
      return True
    return False

  def test_stop(self, epoch, batch, classifier, training_loss):
    return False


# ----------------------------------------------------------------------------------------------------------------------


class ReportTrainingLoss(TrainingHook):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    #    self.batch_interval = kwargs.get('batch_interval',-1)
    #    self.epoch_interval = kwargs.get('epoch_interval',-1)
    self.store_loss_hist = kwargs.get('store_loss_hist', False)
    self.hist = []

  def __call__(self, epoch, batch, classifier, training_loss):
    if self.test_execute(epoch, batch, classifier, training_loss):
      #      pass
      if batch == -1:
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, training_loss))
      else:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch + 1, training_loss))
      if self.store_loss_hist:
        self.hist.append(training_loss)


# ----------------------------------------------------------------------------------------------------------------------

class CollectLossesOnTestSet(TrainingHook):
  def __init__(self, X_test, Y_test, **kwargs):
    super().__init__(**kwargs)
    self.X_test = X_test
    self.Y_test = Y_test
    #    self.batch_interval = kwargs.get('batch_interval',-1)
    #    self.epoch_interval = kwargs.get('epoch_interval',-1)
    self.loss_fun = kwargs.get('loss', -1)
    self.store_vars_hist = kwargs.get('store_vars_hist', False)
    self.hist = []
    self.Y_init = np.ones(Y_test.shape) * 0.5
    self.w_hist = []
    self.b_hist = []

  def __call__(self, epoch, batch, classifier, training_loss):
    if self.test_execute(epoch, batch, classifier, training_loss):
      Y_pred = classifier.model(self.X_test, self.Y_init)
      loss = self.loss_fun(self.Y_test, Y_pred, classifier.model)
      self.hist.append(loss)
      if self.store_vars_hist:
        self.w_hist.append(classifier.model.W.numpy())
        self.b_hist.append(classifier.model.b.numpy())


# ----------------------------------------------------------------------------------------------------------------------

class GradientBooster(TrainingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.epochs=kwargs.get('epochs')
        self.p_threshold = kwargs.get('p_threshold',0.05)
        self.boost_factor = kwargs.get('boost_factor',100)

    def __call__(self, epoch, batch, classifier, training_loss):
        if classifier.current_gradient is None:
            return
        if self.test_execute(epoch, batch, classifier, training_loss):
            p = np.random.random_sample()
            if p<self.p_threshold:
                return
            # classifier.current_gradient=(classifier.current_gradient[0]*self.boost_factor,classifier.current_gradient[1]*self.boost_factor)
            classifier.current_gradient=(classifier.current_gradient[0]*self.boost_factor,classifier.current_gradient[1])


# ----------------------------------------------------------------------------------------------------------------------

class CollectWeightsAndGradients(TrainingHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.collect_weights = kwargs.get('collect_weights', True)
        self.collect_gradients = kwargs.get('collect_gradients', True)
        self.collect_losses = kwargs.get('collect_losses', True)

        self.w_hist = []
        self.b_hist = []
        self.loss_hist = []
        self.w_gradient_hist = []
        self.b_gradient_hist = []

    def __call__(self, epoch, batch, classifier, training_loss):
        if not self.test_execute(epoch, batch, classifier, training_loss):
            return
        if batch == -1:
            return
        if self.collect_weights:
            w = tf.reduce_mean(classifier.model.W**2)
            self.w_hist.append(w.numpy())
            b = tf.reduce_mean(classifier.model.b**2)
            self.b_hist.append(w.numpy())
        if self.collect_losses:
            self.loss_hist.append(training_loss.numpy())
        if self.collect_gradients:
            # print(gradients[0])
            # exit(0)
            g = tf.reduce_mean(classifier.current_gradient[0]**2)
            self.w_gradient_hist.append(g.numpy())
            g = tf.reduce_mean(classifier.current_gradient[1]**2)
            self.b_gradient_hist.append(g.numpy())


    def get_results(self):
        b = Bunch()
        b.losses = np.array(self.loss_hist)
        b.w_weights = np.array(self.w_hist)
        b.b_weights = np.array(self.b_hist)
        b.w_gradients = np.array(self.w_gradient_hist)
        b.b_gradients = np.array(self.b_gradient_hist)
        return b

    def plot_results(self,sel=['losses']):
        r = self.get_results()
        plt.figure()
        for k in r:
            if k in sel:
                x = np.arange(r[k].size) + 1
                plt.plot(x, r[k], label=k)
        plt.xlabel('iterations')
        plt.ylabel('scores')
        plt.legend()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
