import numpy as np
import string
from collections import Counter

np.set_printoptions(threshold=np.nan)

class naivebayes:
    def train(self, data):
        self.piprob = np.zeros(shape=(26, 128))
        self.classprob = np.zeros(shape=(26))
        tchars = 0
        charcount = Counter()
        for fold in data:
            for word in data[fold]:
                for character in word:
                    idx = ord(character['letter'])-ord('a')
                    for pindex, pixel in enumerate(character['inputs']):
                        self.piprob[idx][pindex] += pixel
                    charcount[character['letter']] += 1
                    tchars += 1

        for rowidx, row in enumerate(self.piprob):
            for colidx, col in enumerate(row):
                self.piprob[rowidx][colidx] += 1
                self.piprob[rowidx][colidx] /= charcount[chr(ord('a')+rowidx)]
        print "*******Char count*********"
        for character in charcount:
            self.classprob[ord(character)-ord('a')] = float(charcount[character])/tchars
        print self.classprob

    def predict(self, pixels):
        arange = list(string.ascii_lowercase)
        tprobs = np.zeros(shape=(26))

        for c in arange:
            tp = 0
            cidx = ord(c)-ord('a')
            for pindex, pixel in enumerate(pixels):
                if pixel != 0:
                    tp += np.log(self.piprob[cidx][pindex])
                else:
                    tp += np.log(1-self.piprob[cidx][pindex])
            tp += np.log(self.classprob[cidx])
            tprobs[cidx] = tp
        return chr(ord('a')+np.argmax(tprobs))
