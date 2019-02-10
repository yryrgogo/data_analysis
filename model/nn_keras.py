import os
import time
import datetime
import sys
import warnings
from contextlib import contextmanager
import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import class_weight

#========================================================================
# Keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import backend as K

#  import functools
#  import tensorflow as tf
#========================================================================


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s\n".format(title, time.time() - t0))


#========================================================================
# For Elo Competition
def RMSE(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


#========================================================================
# Corporación Favorita Grocery Sales Forecasting 1st Place NN Model
# https://www.kaggle.com/shixw125/1st-place-nn-model-public-0-507-private-0-513
def elo_build_NN(input_rows, input_cols):
    model = Sequential()
    model.add(LSTM(512, input_shape=(input_rows, input_cols)))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    # original
    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(16))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(1))

    return model


def elo_build_linear_NN(input_cols, first_neuron=1024, drop_rate=0.7):

    #the model is just a sequence of fully connected layers, batch normalization and dropout using ELUs as activation functions
    model = Sequential()
    model.add(Dense(first_neuron, input_dim=input_cols, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(first_neuron*2, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_rate))
    model.add(Dense(first_neuron*2, activation='elu'))
    model.add(Dense(first_neuron, activation='elu'))
    model.add(Dense(1, activation='linear'))
    return model



def basic_build():
    models = Sequential()
    models.add(Dense(output_dim=1024, input_dim=input_d, init='lecun_uniform')) 
    models.add(Activation('relu'))
    models.add(BatchNormalization())
    models.add(Dropout(0.5))
    models.add(Dense(512, activation='relu',init='lecun_uniform'))
    models.add(Activation('relu'))
    models.add(BatchNormalization())
    models.add(Dropout(0.4))
    models.add(Dense(2, init='lecun_uniform'))
    models.add(Activation('softmax'))
    opt = optimizers.Adam(lr=learning_rate)
    models.compile(loss='binary_crossentropy', optimizer=opt)

#========================================================================


#  def as_keras_metric(method):
#          @functools.wraps(method)
#          def wrapper(self, args, **kwargs):
#              """ Wrapper for turning tensorflow metrics into keras metrics """
#              value, update_op = method(self, args, **kwargs)
#              K.get_session().run(tf.local_variables_initializer())
#              with tf.control_dependencies([update_op]):
#                  value = tf.identity(value)
#              return value
#          return wrapper
#  #  auc_roc = as_keras_metric(tf.metrics.auc)


#  class AucCallback(Callback):  # inherits from Callback

#      def __init__(self, validation_data=(), patience=20, is_regression=True, best_model_name='best_keras.mdl', feval='roc_auc_score', batch_size=1024*8):
#          super(Callback, self).__init__()

#          self.patience = patience
#          self.X_val, self.y_val = validation_data  # tuple of validation X and y
#          self.best = -np.inf
#          self.wait = 0  # counter for patience
#          self.best_model = None
#          self.best_model_name = best_model_name
#          self.is_regression = is_regression
#          self.y_val = self.y_val  # .astype(np.int)
#          self.feval = feval
#          self.batch_size = batch_size

#      def on_epoch_end(self, epoch, logs={}):
#          p = self.model.predict(
#              self.X_val, batch_size=self.batch_size, verbose=0)  # .ravel()
#          if self.feval == 'roc_auc_score':
#              current = roc_auc_score(self.y_val, p)

#          if current > self.best:
#              self.best = current
#              self.wait = 0
#              self.model.save_weights(self.best_model_name, overwrite=True)

#          else:
#              if self.wait >= self.patience:
#                  self.model.stop_training = True
#                  print('Epoch %05d: early stopping' % (epoch))

#              self.wait += 1  # incremental the number of times without improvement
#          print('Epoch %d Auc: %f | Best Auc: %f \n' %
#                (epoch, current, self.best))


#  def redhat_keras(x_train, y_train, x_val, y_val, x_test, dim):

#      # モデル定義
#      model = Sequential()

#      #  Embedding
#      model.add(Embedding(
#          output_dim = embedding_size,
#          input_dim = dim,
#          input_length=dim,
#          dropout=0.2,
#          embeddings_regularizer=None
#      ))
#      model.add(Dropout(0.1))

#      model.add(Dense(128, activation='relu', input_shape=(dim,)))
#      model.add(Dropout(0.1))
#      model.add(Dense(128, activation='relu'))
#      model.add(Dropout(0.1))
#      model.add(Flatten())
#      model.add(Dense(1, activation='sigmoid'))

#      model.compile(optimizer='adam',
#                    loss='binary_crossentropy',
#                    metrics=[auc_roc])

#      model.fit(x_train,
#                y_train,
#                epochs=epoch_size,
#                batch_size=batch_size,
#                validation_data=(x_val, y_val)
#                )
#      y_pred = model.predict(x_val, batch_size=batch_size)
#      sc_score = roc_auc_score(y_val, y_preds)
#      logger.info(f'auc: {sc_score}')
#      test_pred = model.predict(x_test, batch_size=2048)

#      return y_pred, sc_score, test_pred


#  def dense_bn_block(x, size, act='elu'):  # LeakyReLU(alpha=0.01)):
#      x = Dense(size, kernel_initializer='he_normal')(x)
#      x = BatchNormalization()(x)
#      x = AlphaDropout(0.3)(x)
#      if isinstance(act, str):
#          z = Activation(act)(x)
#      else:
#          z = act(x)
#      return z


#  def build_classifier(input_shape):
#      model_input = Input(shape=input_shape)

#      model = GaussianNoise(stddev=0.001)(model_input)

#      model = dense_bn_block(model, 1024)
#      model = dense_bn_block(model, 1024)
#      model = dense_bn_block(model, 1024)
#      model = dense_bn_block(model, 256)
#      model = dense_bn_block(model, 32)

#      model_output = Dense(1, activation='sigmoid')(model)
#      return Model(inputs=[model_input], outputs=[model_output])


#  def keras_ANN(x_train, y_train, x_val, y_val, x_test, dim):

#      with timer("calculating class weights"):

#          class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
#          print('sklearn class weights: {}'.format(class_weights))


#      with timer("creating classifier"):
#          input_shape = (x_train.shape[-1],)
#          model = build_classifier(input_shape)

#      # saved model variables
#      best_model_path = './best_model.h5'

#      # auc variables
#      best_auc = 0

#      min_delta = 0.0003

#      # early stop variables
#      max_es_rounds = 10
#      current_es_round = 0

#      # reduce lr variables
#      max_lr_rounds = 4
#      current_lr_round = 0
#      reduce_lr_rate = (1 / 3)
#      current_lr = 1.0

#      # increase batch size
#      max_bs_rounds = 6
#      current_bs_round = 0
#      increase_bs_rate = 2
#      current_bs = 256
#      max_bs = 2048

#      current_epoch = 1
#      while True:
#          with timer("epoch #{}".format(current_epoch)):
#              # epochs are random stratified minibatch
#              sgd = SGD(lr=current_lr, momentum=0.85, decay=0.0, nesterov=True)
#              auc_roc = as_keras_metric(tf.metrics.auc)
#              model.compile(
#                  optimizer=sgd, loss='binary_crossentropy', metrics=[auc_roc])
#              print('current lr: {}'.format(current_lr))

#              # train model
#              model.fit(x_train, y_train, batch_size=current_bs,
#                        epochs=1, class_weight=class_weights)

#              # get validation score
#              #  y_pred = model.predict(x_val, batch_size=2048)
#              ' 2次元配列になってるのでflattenする '
#              y_pred = model.predict(x_val, batch_size=2048).flatten().clip(0, 1)

#              try:
#                  auc = roc_auc_score(y_val, y_pred)
#              except ValueError:
#                  logger.info(f'y_val: {y_val}')
#                  logger.info(f'y_pred: {y_pred}')
#                  logger.info(f'{y_pred[y_pred==np.nan]}')
#                  logger.info('ValueError')
#              logger.info(f'auc: {auc}')

#              # save model and check early stop
#              if (auc - best_auc) > min_delta:
#                  print('validation ROC_AUC improved from {} to {}!'.format(
#                      best_auc, auc))
#                  model.save(best_model_path)
#                  print('model_saved to {}'.format(best_model_path))

#                  # deal with es variables
#                  current_es_round = 0

#                  # deal with lr variables
#                  current_lr_round = 0

#                  # deal with bs variables
#                  current_bs_round = 0

#                  best_auc = auc
#              else:
#                  current_es_round += 1
#                  current_lr_round += 1
#                  current_bs_round += 1
#                  print('validation ROC_AUC did not improve from {} for {} rounds'.format(
#                      best_auc, current_es_round))

#              # stop if potato
#              if current_es_round >= max_es_rounds:
#                  print('Stopping due to validation AUC_ROC not increasing for {} rounds'.format(
#                      max_es_rounds))
#                  break

#              # reduce lr if potato
#              if current_lr_round >= max_lr_rounds:
#                  print('reducing learning rate from {} to {}'.format(
#                      current_lr, current_lr * reduce_lr_rate))
#                  current_lr *= reduce_lr_rate
#                  current_lr_round = 0

#              # increase batch size if potato
#              if current_bs_round >= max_bs_rounds:
#                  if int(current_bs * increase_bs_rate) <= max_bs:
#                      print('increasing batch size from {} to {}'.format(
#                          current_bs, int(current_bs * increase_bs_rate)))
#                      current_bs = int(current_bs * increase_bs_rate)
#                  current_bs_round = 0

#          current_epoch += 1

#      ' 2次元になるのでflatten '
#      test_pred = model.predict(x_test, batch_size=2048).flatten().clip(0,1)

#      return y_pred, auc, test_pred
