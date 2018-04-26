import json
import numpy as np
from naivebayes import naivebayes
from collections import Counter
import matplotlib.pyplot as plt

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

datav = DataV("Data/letter.data.json")
datav.readdata()
datav.printdata()
datav.statistics()
trainfolds, develfolds, testfolds = datav.makesplit(60, 20, 20)
inputotindex = datav.makemaps()

NB = naivebayes()
NB.train(trainfolds)

tchars = 0
ncorrect = 0
for fold in develfolds:
    for word in develfolds[fold]:
        for character in word:
            p = NB.predict(character["inputs"])
            if p == character["letter"]:
                ncorrect += 1
            tchars += 1

print "Accuracy = ", float(ncorrect)/float(tchars)
