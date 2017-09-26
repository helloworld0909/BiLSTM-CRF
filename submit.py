import sys
import logging
import re
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
X_test = data.loadCoNLL(testPath, loadFeatures=True, mode='test')
model = load_model('h5/' + dataIdx + '/' +  sys.argv[-1])

acc = float(re.search(r'acc([\d.]+)\d', sys.argv[-1]).group(1))
data.predictWithFeature(model, X_test, '%s_enrich_%.5f.txt' %(dataIdx, acc))