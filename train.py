import logging
import random
from sklearn.model_selection import train_test_split
from neuralnets.BiLSTMCRF import BiLSTMCRF
from neuralnets.BiLSTMCRF import load_model
from neuralnets.keraslayers.ChainCRF import create_custom_objects
from util.data import Data
from util.metric import categorical_metric
from util.callback import metricHistory


# :: Logging level ::
logger = logging.getLogger()
logger.setLevel(logging.INFO)

inputPath = 'data/artist/artist.train'


data = Data(inputPathList=[inputPath])
data.loadCoNLL(inputPath)

X = data.sentences
y = data.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

modelWrapper = BiLSTMCRF(data)
model = modelWrapper.buildModel()

history = metricHistory(X_test, y_test)
history.set_model(model)
history.set_params(params={'label2idx': data.label2idx})
model.fit(X_train, y_train, epochs=20, batch_size=128, shuffle=True, callbacks=[history])
model.save('model.h5')

y_predict = model.predict(X_test)
y_predict = y_predict.argmax(axis=-1)

print(categorical_metric(y_test, y_predict, label2idx=data.label2idx))
