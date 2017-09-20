import os
import logging
import time
from keras.callbacks import Callback
from util.metric import categorical_metric


class metricHistory(Callback):

    def __init__(self, X_val=None, y_val=None, saveDir=None, save=True):
        super(metricHistory, self).__init__()
        self.metric_history = []
        self.X_val = X_val
        self.y_val = y_val
        self.model = None
        self.params = {}
        if saveDir is None:
            dirName = time.strftime("%m-%d_%H:%M", time.localtime())
            self.dir = 'h5/' + dirName + '/'
        else:
            self.dir = 'h5/' + saveDir + '/'
        self.save = save

    def set_params(self, params):
        self.params.update(params)

    def on_train_begin(self, logs=None):
        os.mkdir(self.dir)

    def on_epoch_end(self, epoch, logs={}):
        y_predict = self.model.predict(self.X_val)
        y_predict = y_predict.argmax(axis=-1)
        metric = categorical_metric(self.y_val, y_predict)
        if self.save and self.metric_history and metric > max(self.metric_history):
            self.model.save('{}epoch{}_acc{}.h5'.format(self.dir, epoch, metric))
        self.metric_history.append(metric)
        logging.info('Metric: ' + str(metric))
