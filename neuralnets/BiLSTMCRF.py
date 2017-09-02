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
from .keraslayers.ChainCRF import create_custom_objects
from util import preprocess

class BiLSTMCRF(object):

    params = {'wordEmbeddingDim': 100, 'charEmbeddingDim': 25, 'lstmOutDim': 100, 'filters': 15}

    vocabSize = 0
    labelDim = 0
    maxTokenLen = 0
    maxSentenceLen = 0

    charSet = []
    char2idx = {}

    charEmbedding = []
    tokenIdx2charVector = []
    wordEmbedding = []


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
        self.wordEmbedding = data.wordEmbedding

    def buildModel(self):

        word_input = Input((self.maxSentenceLen, ), name='word_input')
        word_input_masking = Masking(mask_value=0, input_shape=(self.maxSentenceLen, ))(word_input)
        word = Embedding(
            input_dim=self.vocabSize,
            output_dim=self.params['wordEmbeddingDim'],
            input_length=self.maxSentenceLen,
            weights=[self.wordEmbedding],
            trainable=False,
            name='word_embedding'
        )(word_input_masking)


        char_input = Embedding(
            input_dim=self.tokenIdx2charVector.shape[0],
            output_dim=self.tokenIdx2charVector.shape[1],
            input_length=self.maxSentenceLen,
            weights=[self.tokenIdx2charVector],
            trainable=False,
            name='word_to_char'
        )(word_input_masking)

        char = TimeDistributed(Embedding(
                input_dim=len(self.charSet),
                output_dim=self.params['charEmbeddingDim'],
                weights=[self.charEmbedding],
                trainable=True,
            ),
            name='char_embedding'
        )(char_input)

        char = TimeDistributed(LSTM(10, return_sequences=False), name='charLSTM')(char)

        merge_layer = concatenate([word, char])

        bilstm = Bidirectional(LSTM(self.params['lstmOutDim'], return_sequences=True), name='BiLSTM')(merge_layer)

        hidden = TimeDistributed(Dense(self.labelDim, activation=None), name='hidden_layer')(bilstm)


        crf = ChainCRF()
        output = crf(hidden)

        model = Model(inputs=word_input, outputs=output)
        model.compile(optimizer='adam', loss=crf.sparse_loss, metrics=['sparse_categorical_accuracy'])
        model.summary()

        return model

def load_model(filepath):
    return keras.models.load_model(filepath, custom_objects=create_custom_objects())