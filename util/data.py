import os
import logging
from collections import defaultdict
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from util import preprocess


class Data(object):

    token2idx = {'PADDING': 0, 'UNKNOWN': 1}
    feature2idx = defaultdict(lambda : {'PADDING': 0})
    label2idx = {'PADDING': 0}
    tokenIdx2charVector = []

    sentences = None
    charSentences = None
    labels = None
    features = None

    def __init__(self, inputPathList):

        tokenFreq = preprocess.tokenFrequency(inputPathList)
        for token in tokenFreq.keys():
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
        self.feature2idx, self.label2idx = preprocess.featureLabelIndex(inputPathList)
        self.char2idx = preprocess.getChar2idx()
        self.maxTokenLen = len(max(self.token2idx.keys(), key=len))
        logging.info('Max token length: ' + str(self.maxTokenLen))

        self.maxSentenceLen = 0

        self.vocabSize = len(self.token2idx)
        logging.info('Vocabulary size: ' + str(self.vocabSize))

        self.labelDim = len(self.label2idx)
        logging.info('Label dim: ' + str(self.labelDim))

        self.initToken2charVector()

    def initToken2charVector(self):
        tokenIdx2charVector = []
        for token, idx in sorted(self.token2idx.items(), key=lambda kv: kv[1]):
            if idx != 0:
                charVector = list(map(lambda c: self.char2idx.get(c, 1), token))    # 1 for UNKNOWN char
            else:
                charVector = [0]  # PADDING
            tokenIdx2charVector.append(charVector)

        self.tokenIdx2charVector = np.asarray(pad_sequences(tokenIdx2charVector))
        logging.debug(self.tokenIdx2charVector.shape)




    def loadCoNLL(self, filePath):

        sentences = []
        charSentences = []
        features = defaultdict(list) #TODO: load features
        labels = []

        with open(filePath, 'r', encoding='utf-8') as inputFile:
            sentenceTmp = []
            charSentenceTmp = []
            labelTmp = []
            for line in inputFile:
                line = line.strip()
                if not line:
                    sentences.append(sentenceTmp)
                    labels.append(labelTmp)

                    sentenceTmp = []
                    labelTmp = []
                else:
                    data_tuple = line.split('\t')

                    token = data_tuple[0]
                    tokenIdx = self.token2idx[token]
                    sentenceTmp.append(tokenIdx)


                    labelIdx = self.label2idx[data_tuple[-1]]
                    labelTmp.append(labelIdx)

            sentences.append(sentenceTmp)
            labels.append(labelTmp)

        # Pad sentence to the longest length
        self.maxSentenceLen = len(max(sentences, key=len))
        self.sentences = pad_sequences(sentences, maxlen=self.maxSentenceLen)

        # Pad char to the length of longest word
        for idx, seq in enumerate(charSentences):
            charSentences[idx] = pad_sequences(seq)
        self.charSentences = charSentences

        # Transform labels to one hot encoding
        self.labels = []
        for seq in pad_sequences(labels):
            self.labels.append(to_categorical(seq))
        self.labels = np.asarray(self.labels)





