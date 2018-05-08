import json
import numpy as np
from CRF import conditional_random_field
from collections import Counter
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

class DataV:
    def __init__(self, fname):
        self.fname = fname

    def readdata(self):
        self.data = json.load(open(self.fname))

    def printdata(self):
        nwords = 0
        nchars = 0
        nfolds = 0
        for fold in self.data:
            for word in self.data[fold]:
                for character in word:
                    nchars += 1
                nwords += 1
            nfolds += 1
        print "nwords:", nwords
        print "nchars:", nchars
        print "nfolds:", nfolds

    def statistics(self):
        listdict = []
        foldchar = np.zeros(shape=(10, 26))
        for fold in self.data:
            chars = []
            for word in self.data[fold]:
                for character in word:
                    chars.append(character['letter'])
                    foldchar[ord(fold)-ord('0')][ord(character['letter'])-ord('a')] += 1
            listdict.append(chars)

        np.savetxt("foldchar.csv", foldchar)

    def makesplit(self, trainp, devp, testp):
        nfolds = 10
        trfolds = trainp/nfolds
        devfolds = devp/nfolds
        tefolds = testp/nfolds
        
        self.trainfolds = {}
        self.develfolds = {}
        self.testfolds = {}
        for fold in self.data:
            if trfolds != 0:
                self.trainfolds[fold] = self.data[fold]
                trfolds -= 1
            elif devfolds != 0:
                self.develfolds[fold] = self.data[fold]
                devfolds -= 1
            else:
                self.testfolds[fold] = self.data[fold]
        del self.data
        return self.trainfolds, self.develfolds, self.testfolds

    def makemaps(self):
        self.inputtoindex = {}
        idx = 0
        for fold in self.trainfolds:
            for word in self.trainfolds[fold]:
                for character in word:
                    tstr = ''.join(str(e) for e in character["inputs"])
                    if tstr not in self.inputtoindex:
                        self.inputtoindex[tstr] = idx
                        idx = idx+1
        return self.inputtoindex

datav = DataV("../Data/letter.data.json")
datav.readdata()
trainfolds, develfolds, testfolds = datav.makesplit(20, 20, 20)

W = np.zeros((26, 128))
T = np.zeros((26, 26))

features = {}

cnt = 0
for fold in trainfolds:
    for word in trainfolds[fold]:
        X = np.zeros((len(word), 128))
        Y = np.zeros((len(word)), dtype=np.uint64)
        WD = []
        for i,character in enumerate(word):
            X[i] = character['inputs']
            Y[i] = ord(character['letter'])-ord('a')
        features[cnt] = (X, Y)
        cnt += 1

print len(features)
CRF = conditional_random_field(features)
CRF.optimize()
