import sys
import logging
from neuralnets.BiLSTMCRF import load_model
from util.data import Data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

if len(sys.argv) >= 3:
    dataIdx = '{:0>2}'.format(sys.argv[1])
else:
    dataIdx = 'CoNLL'

trainPath = 'data/normal/en_train_{}.txt'.format(dataIdx)
testPath = 'data/normal/en_test_CoNLL.txt'

data = Data(inputPathList=[trainPath], testPath=testPath)
model = load_model('h5/' + dataIdx + '/' +  sys.argv[-1])

data.predict(model, testPath='data/normal/en_test.csv', outputPath='en_test_enrich.txt')