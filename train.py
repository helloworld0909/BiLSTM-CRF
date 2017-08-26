import logging
from keras.models import load_model
from neuralnets.BiLSTMCRF import BiLSTMCRF
from neuralnets.keraslayers.ChainCRF import create_custom_objects
from util.data import Data

# :: Logging level ::
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

inputPath = 'data/family/train.txt'


data = Data(inputPathList=[inputPath])
data.loadCoNLL(inputPath)

X1 = data.sentences
X2 = data.charSentences
y = data.labels

modelWrapper = BiLSTMCRF(data)
model = modelWrapper.buildModel()

model.fit(X1, y)
model.save('model.h5')

model = load_model('model.h5', custom_objects=create_custom_objects())
model.summary()
