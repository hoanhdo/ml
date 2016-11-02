# -*- coding: utf-8 -*-
from numpy import *


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
                    'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', \
                    'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', \
                    'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
                    'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 1 is abusive, 0 not
    classVec = [0, 1, 0, 1, 0, 1] 
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # union 2 set
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s i nos in my vcabulary !" % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom
    return p0Vect, p1Vect, pAbusive


def main():
    # listOPosts, listClasses = loadDataSet()
    # print listOPosts[0]
    # print listOPosts[3]
    # myVocabList = createVocabList(listOPosts)
    # print myVocabList
    # print setOfWords2Vec(myVocabList, listOPosts[0])
    # print setOfWords2Vec(myVocabList, listOPosts[3])
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    print trainMat
    print listClasses
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print p0V
    # print p1V
    # print pAb


if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
