import keras
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import Model

import os
import sys
import random
import time
import math
import numpy as np
import logging

from .keraslayers.ChainCRF import ChainCRF
from util import preprocess

class BiLSTMCRF(object):

    params = {'wordEmbeddingDim': 100, 'charEmbeddingDim': 10, 'lstmOutDim': 100}

    vocabSize = 0
    labelDim = 0
    maxTokenLen = 0
    maxSentenceLen = 0

    charSet = []
    char2idx = {}

    charEmbedding = []
    tokenIdx2charVector = []


    def __init__(self, data, params=None):
        self.initCharEmbedding()
        self.setLookup(data)
        if params is not None:
            self.params.update(params)


    def initCharEmbedding(self):

        self.charSet = preprocess.getCharSet()
        self.char2idx = preprocess.getChar2idx()

        charEmbeddingDim = self.params['charEmbeddingDim']
        for _ in self.charSet:
            limit = math.sqrt(3.0 / charEmbeddingDim)
            vector = np.random.uniform(-limit, limit, charEmbeddingDim)
            self.charEmbedding.append(vector)

    def setLookup(self, data):
        self.vocabSize = data.vocabSize
        self.labelDim = data.labelDim
        self.maxTokenLen = data.maxTokenLen
        self.maxSentenceLen = data.maxSentenceLen
        self.tokenIdx2charVector = data.tokenIdx2charVector

    def buildModel(self):

        word_input = Input((self.maxSentenceLen, ), name='word_input')
        word = Embedding(
            input_dim=self.vocabSize,
            output_dim=self.params['wordEmbeddingDim'],
            input_length=self.maxSentenceLen,
            trainable=True,
            name='word_embedding'
        )(word_input)


        char_input = Embedding(
            input_dim=self.tokenIdx2charVector.shape[0],
            output_dim=self.tokenIdx2charVector.shape[1],
            input_length=self.maxSentenceLen,
            weights=[self.tokenIdx2charVector],
            trainable=False,
            name='word_to_char'
        )(word_input)

        char = TimeDistributed(Embedding(
                input_dim=len(self.charSet),
                output_dim=self.params['charEmbeddingDim'],
                weights=[self.charEmbedding],
                trainable=True,
                mask_zero=True
            ),
            name='char_embedding'
        )(char_input)

        char = TimeDistributed(Conv1D(filters=25, kernel_size=3, border_mode='same'), name='Conv1D')(char)
        char = TimeDistributed(GlobalMaxPooling1D(), name='MaxPooling')(char)

        merge_layer = concatenate([word, char])

        bilstm = Bidirectional(LSTM(self.params['lstmOutDim'], return_sequences=True), name='BiLSTM')(merge_layer)

        hidden = TimeDistributed(Dense(self.labelDim, activation=None), name='hidden_layer')(bilstm)


        crf = ChainCRF()
        output = crf(hidden)
        loss = crf.loss

        model = Model(inputs=word_input, outputs=output)
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        model.summary()

        return model