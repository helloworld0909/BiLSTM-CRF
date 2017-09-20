import os
import logging
from collections import defaultdict
import numpy as np


def getCharSet():
    charStr = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"
    return ['PADDING', 'UNKNOWN'] + list(charStr)

def getChar2idx():
    charSet = getCharSet()
    return {v:k for k,v in enumerate(charSet)}

def sentenceLengthDistribution(filePathList):
    sentenceLength = 0
    distribution = defaultdict(int)

    for filePath in filePathList:
        with open(filePath, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                line = line.strip('\n')
                if line:
                    sentenceLength += 1
                else:
                    distribution[sentenceLength] += 1
                    sentenceLength = 0
            if sentenceLength != 0:
                distribution[sentenceLength] += 1

    return distribution

def tokenLengthDistribution(token2idx):
    distribution = defaultdict(int)
    for token in token2idx.keys():
        tokenLength = len(token)
        distribution[tokenLength] += 1
    return distribution


def tokenFrequency(filePathList):
    tokenFreq = defaultdict(int)

    for filePath in filePathList:
        with open(filePath, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                line = line.strip('\n')
                if not line:
                    continue
                else:
                    token = line.split('\t')[0]
                    tokenFreq[token] += 1
    return tokenFreq

def featureLabelIndex(filePathList):
    feature2idx = defaultdict(lambda : {'PADDING': 0})
    label2idx = {'PADDING': 0}

    for filePath in filePathList:
        with open(filePath, 'r', encoding='utf-8') as inputFile:
            for line in inputFile:
                line = line.strip('\n')
                if not line:
                    continue
                else:
                    data_tuple = line.split('\t')
                    features = data_tuple[1:-1]
                    for idx, feature in enumerate(features):
                        if feature not in feature2idx[idx]:
                            feature2idx[idx][feature] = len(feature2idx[idx])
                    label = data_tuple[-1]
                    if label not in label2idx:
                        label2idx[label] = len(label2idx)

    return feature2idx, label2idx

def selectPaddingLength(lengthDistribution, ratio=0.99):
    totalCount = sum(lengthDistribution.values())
    threshold = int(totalCount * ratio)
    countSum = 0
    selectedLength = 0
    for length, count in sorted(lengthDistribution.items(), key=lambda kv: kv[0]):
        countSum += count
        selectedLength = length
        if countSum >= threshold:
            break
    return selectedLength

def loadWordEmbedding(filepath, dim=100):
    word2vector = {'PADDING': np.zeros(dim), 'UNKNOWN': np.random.uniform(-0.25, 0.25, dim)}
    with open(filepath, 'r', encoding='utf-8') as embeddingFile:
        for line in embeddingFile:
            data_tuple = line.rstrip('\n').split(' ')
            token = data_tuple[0]
            vector = data_tuple[1:]
            word2vector[token] = vector
    return word2vector
