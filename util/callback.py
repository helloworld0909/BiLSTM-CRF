import logging
from keras.callbacks import Callback
from util.metric import categorical_metric


class metricHistory(Callback):

    def __init__(self, X_val=None, y_val=None):
        super(metricHistory, self).__init__()
        self.metric_history = []
        self.X_val = X_val
        self.y_val = y_val
        self.model = None
        self.params = {}

    def set_params(self, params):
        self.params.update(params)

    def on_epoch_end(self, epoch, logs={}):
        y_predict = self.model.predict(self.X_val)
        y_predict = y_predict.argmax(axis=-1)
        metric = categorical_metric(self.y_val, y_predict)
        self.metric_history.append(metric)
        logging.info(str(metric))
        self.model.save('h5/epoch{}_acc{}.h5'.format(epoch, metric))