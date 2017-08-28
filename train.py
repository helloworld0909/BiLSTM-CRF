import logging
import random
from sklearn.model_selection import train_test_split
from neuralnets.BiLSTMCRF import BiLSTMCRF
from neuralnets.BiLSTMCRF import load_model
from neuralnets.keraslayers.ChainCRF import create_custom_objects
from util.data import Data
from util.metric import categorical_metric


# :: Logging level ::
logger = logging.getLogger()
logger.setLevel(logging.INFO)

inputPath = 'data/class/train.txt'


data = Data(inputPathList=[inputPath])
data.loadCoNLL(inputPath)

X = data.sentences
y = data.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

modelWrapper = BiLSTMCRF(data)
model = modelWrapper.buildModel()

model.fit(X_train, y_train, epochs=10, validation_split=0.1, shuffle=True)
model.save('model.h5')

y_predict = model.predict(X_test)
y_predict = y_predict.argmax(axis=-1)
print(y_predict)

print(data.label2idx)
print(categorical_metric(y_test, y_predict, label2idx=data.label2idx))
