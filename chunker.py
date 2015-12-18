
from __future__ import division
import optparse, sys, codecs, re, logging, os
from collections import Counter, defaultdict

import sys
import json
import datetime
import nltk
import pickle
import util
from os import listdir
from os.path import isfile, join

from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import treebank

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
st = LancasterStemmer()

#example sentence
sentence = 'Mr. Vinken is chairman of Elsevier N.V., the Dutch publishing group.'

tokens = tokenizer.tokenize(sentence)

    
def parse_training_data(path,dirs):
    training_data,label_sequence = [],[]
    for single_file in dirs:
        with open(path+single_file,'r') as f:
            sentence,labels = [],[]
            for line in f:
                if line == "" or line == " " or line=='\n':
                    if sentence:
                        training_data.append(sentence)
                        sentence = []
                        label_sequence.append(labels)
                        labels = []
                        continue
                else:
                    try:
                        if line[0]=='[':
                            tagged_words = line[1:-2].strip().lower().split(' ')
                            for tagged_word in tagged_words:
                                pair = tagged_word.split('/')
                                sentence.append((pair[0],pair[1]))
                                labels.append('NP')
                        else:
                            tagged_words = line.strip().split(' ')
                            for tagged_word in tagged_words:
                                pair = tagged_word.split('/')
                                if isinstance(pair,list):
        						  sentence.append((pair[0],pair[1]))
                                else:
                                    sentence.append((pair,'/'))
                                labels.append('-')
                    except:
                        pass
            if sentence:
                training_data.append(sentence)
                label_sequence.append(labels)
    return training_data,label_sequence

mypath = 'tagged/'


onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
train_data,label_sequence = parse_training_data(mypath,onlyfiles)

print len(train_data)

def normalize(weights):
    z = sum(weights)
    return [1.0 * w / z for w in weights]


def train_model(sequences,label_sequence,featureExtractor):
    eta = 0.1
    weights = {}
    for sequence,labels in zip(sequences,label_sequence):
        tag_1 = '_BEGIN_'
        tag_2 = '_BEGIN_'
        tags = []
        #print sequence
        for idx,pair in enumerate(sequence):
            #print pair
            word,tag = pair
            if labels[idx]=='NP':
                y = 1
            else:
                y = -1

            x = (sequence,idx,tags)
            phi = featureExtractor(x)

            if util.dotProduct(weights,phi)*y<1:
                util.increment(weights,eta*y,phi)
            tags.append(labels[idx])

    #print weights
    return weights


def test_model(sequences,featureExtractor,weights):
    for sequence in sequences:
    	predictions = []
        tag_1 = '_BEGIN_'
        tag_2 = '_BEGIN_'
        tags = []
        #print sequence
        for idx,pair in enumerate(sequence):
            word,tag = pair
            x = (sequence,idx,tags)
            phi = featureExtractor(x)

            if util.dotProduct(weights,phi)>0:
                label = 1
            else:
                label = 0		

            predictions.append(label)
        print predictions
def featureExtractor(state):
    sequence,i,tags = state
    output = defaultdict(lambda: 0)
    current_word,current_tag = sequence[i]
    feature_inputs = [(current_word,current_tag),(current_word,sequence[i][1]),(current_word,sequence[i][1])]

    for feature_input in feature_inputs:
        output[feature_input]+=1

    return output    

def learnPredictor(trainExamples, featureExtractor):
    
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    numIters = 1#15#13
    eta = 0.1#0.3
    for i in xrange(numIters):
        for pair in trainExamples:
            x,y=pair
            phi = featureExtractor(x)
            if dotProduct(weights,phi)*y<1:
                increment(weights,eta*y,phi)
        print "Iteration:",i+1
            #"trainloss",evaluatePredictor(trainExamples,lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1)),\
            #"testloss",evaluatePredictor(testExamples,lambda(x) : (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
    return weights

#print train_data
weights = train_model(train_data,label_sequence,featureExtractor)

test_model(train_data,featureExtractor,weights)
#print label_sequence[0]
#print label_sequence[1]


