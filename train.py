import logging
import random
import sys
from sklearn.model_selection import train_test_split
from neuralnets.BiLSTMCRF import BiLSTMCRF
from neuralnets.BiLSTMCRF import load_model
from neuralnets.keraslayers.ChainCRF import create_custom_objects
from util.data import Data
from util.metric import categorical_metric
from util.callback import metricHistory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

if len(sys.argv) >= 2:
    dataIdx = '{:0>2}'.format(sys.argv[1])
else:
    dataIdx = 'CoNLL'

trainPath = 'data/normal/en_train_{}.txt'.format(dataIdx)
testPath = 'data/normal/en_test_CoNLL.txt'


data = Data(inputPathList=[trainPath], testPath=testPath)
return_data = data.loadCoNLL(trainPath, loadFeatures=True)

split_data = train_test_split(*return_data, test_size=0.1, random_state=0)
X_train = split_data[:-2:2]
X_val = split_data[1:-2:2]
y_train, y_val = split_data[-2:]

modelWrapper = BiLSTMCRF(data)
model = modelWrapper.buildModel(feature2idx=data.feature2idx)

history = metricHistory(X_val, y_val, saveDir=dataIdx)
history.set_model(model)
model.fit(X_train, y_train, epochs=50, batch_size=64, shuffle=True, callbacks=[history])
model.save('model.h5')

y_predict = model.predict(X_val)
y_predict = y_predict.argmax(axis=-1)

print(categorical_metric(y_val, y_predict))
