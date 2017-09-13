import sys
import logging
from neuralnets.BiLSTMCRF import load_model
from util.data import Data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
)

trainPath = 'data/normal/en_train_CoNLL.txt'
testPath = 'data/normal/en_test_CoNLL.txt'

data = Data(inputPathList=[trainPath], testPath=testPath)
model = load_model('h5/' + sys.argv[1])

data.predict(model, testPath='data/normal/en_test.csv', outputPath='en_test_enrich.txt')