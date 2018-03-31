import numpy as np
import operator
np.set_printoptions(threshold=np.nan)
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
'''
Reduce rank of a vector U, by replacing N
elements which are close to zero with zeros
'''
def reduce_rank(U, N):
    Uprime = U.flatten()
    Uprime.sort()
    return np.where(U < Uprime[-N], 0, U)

TypetoIndex = {}
TypeFreq = Counter()

with open("toy_data", "rb") as f:
    sentences = pickle.load(f)

index = 0
for sentence in sentences:
    for word in sentence:
        if (word['text'] == '__SEQ__') or (word['text'].isalpha()):
            TypeFreq[word['text']] += 1

it = 0
for wtype in TypeFreq.most_common():
    if wtype[0] not in TypetoIndex:
        TypetoIndex[wtype[0]] = index
        index = index+1
    it = it+1

'''
We are interested in only top 10 most frequent words
'''
w1 = 10

'''
print most common w1 words
'''
print "Most common words"
print TypeFreq.most_common()[0:w1]
Lcontext =  np.zeros((len(TypetoIndex), w1))
Rcontext =  np.zeros((len(TypetoIndex), w1))

PrevToken = '__SEQ__' 
for sentence in sentences:
    for word in sentence:
        print word['text'],
        if word['text'] != '__SEQ__' and not word['text'].isalpha():
            continue
        pretindex = TypetoIndex[PrevToken]
        curindex = TypetoIndex[word['text']]
        if curindex < w1:
            Rcontext[pretindex][curindex] += 1
        if pretindex < w1:
            Lcontext[curindex][pretindex] += 1
        PrevToken = word['text']
    print "\n--------------------------------------------------\n"

#for i in range(len(Lcontext)):
#    print "Word=", (TypeFreq.most_common())[i]," ", Lcontext[i]
#print Lcontext
#print "********************************"
#print Rcontext
plt.imshow(Lcontext, cmap='hot', interpolation='nearest')
plt.show()
'''
print "Lcontext shape=", Lcontext.shape
UL, SL, VL = np.linalg.svd(Lcontext, full_matrices=True)

SL = reduce_rank(SL, 100)
SLstar = SL*np.identity(len(SL))
print matrix_rank(SLstar)
Lstar = UL*SLstar
plt.imshow(Lcontext, cmap='hot', interpolation='nearest')
plt.show()

plt.imshow(Lstar, cmap='hot', interpolation='nearest')
plt.show()
'''
