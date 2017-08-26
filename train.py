import logging
from neuralnets.BiLSTMCRF import BiLSTMCRF
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
