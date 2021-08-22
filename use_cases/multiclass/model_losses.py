"""
This module was created to make functions picklable
https://github.com/scikit-learn/scikit-learn/issues/10054
"""
import tensorflow as tf

def model_loss_l2_w(fcm_classifier):
    return tf.reduce_mean(fcm_classifier.base_fcm.model.W**2).numpy()/2

def model_loss_l2_b(fcm_classifier):
    return tf.reduce_mean(fcm_classifier.base_fcm.model.b**2).numpy()/2

def model_loss_l1_w(fcm_classifier):
    return tf.reduce_mean(tf.abs(fcm_classifier.base_fcm.model.W)).numpy()

def model_loss_l1_b(fcm_classifier):
    return tf.reduce_mean(tf.abs(fcm_classifier.base_fcm.model.b)).numpy()

def model_loss_maxabs_w(fcm_classifier):
    return tf.reduce_max(tf.abs(fcm_classifier.base_fcm.model.W)).numpy()

def model_loss_maxabs_b(fcm_classifier):
    return tf.reduce_max(tf.abs(fcm_classifier.base_fcm.model.b)).numpy()
