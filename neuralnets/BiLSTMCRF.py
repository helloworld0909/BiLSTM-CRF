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

    params = {'wordEmbeddingDim': 100, 'charEmbeddingDim': 25, 'lstmOutDim': 100, 'featureEmbeddingDim': 20}

    vocabSize = 0
    labelDim = 0
    maxTokenLen = 0
    maxSentenceLen = 0

    charSet = []
    char2idx = {}
    casing2idx = {}

    charEmbedding = []
    tokenIdx2charVector = []
    tokenIdx2casingVector = []
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
        self.casing2idx = data.casing2idx
        self.tokenIdx2casingVector = data.tokenIdx2casingVector

    def buildModel(self, feature2idx=None):

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

        casing = Embedding(
            input_dim=self.vocabSize,
            output_dim=len(self.casing2idx),
            input_length=self.maxSentenceLen,
            weights=[self.tokenIdx2casingVector],
            trainable=False,
            name='casing_embedding'
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

        char = TimeDistributed(Bidirectional(LSTM(25, return_sequences=False)), name='charLSTM')(char)

        defaultLayers = [word, char, casing]
        inputLayers = [word_input]

        if feature2idx is not None:
            for featureIdx in feature2idx.values():
                feature_input = Input((self.maxSentenceLen,))
                feature = Embedding(
                    input_dim=len(featureIdx),
                    output_dim=self.params['featureEmbeddingDim'],
                    trainable=True,
                )(feature_input)
                defaultLayers.append(feature)
                inputLayers.append(feature_input)

        merge_layer = concatenate(defaultLayers)

        bilstm = Bidirectional(LSTM(self.params['lstmOutDim'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2), name='BiLSTM')(merge_layer)

        hidden = TimeDistributed(Dense(self.params['lstmOutDim'], activation='elu'), name='hidden_1')(bilstm)
        hidden = TimeDistributed(Dense(self.params['lstmOutDim'], activation='elu'), name='hidden_2')(hidden)
        hidden = TimeDistributed(Dense(self.params['lstmOutDim'], activation='elu'), name='hidden_3')(hidden)

        hidden = TimeDistributed(Dense(self.labelDim, activation=None), name='hidden_layer')(hidden)


        crf = ChainCRF()
        output = crf(hidden)

        model = Model(inputs=inputLayers, outputs=output)
        model.compile(optimizer='adam', loss=crf.sparse_loss)
        model.summary()

        return model

def load_model(filepath):
    return keras.models.load_model(filepath, custom_objects=create_custom_objects())