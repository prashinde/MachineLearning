import json
import numpy as np
np.set_printoptions(threshold=np.nan)
import operator
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import matrix_rank
import seaborn as sns
sns.set()

'''
Normalizes each row of a given matrix to unit length
'''
def normalize_row(C):
    l2 = np.atleast_1d(np.linalg.norm(C, ord=2, axis=1))
    l2[l2==0] = 1
    return C / np.expand_dims(l2, 1)

'''
Reduce rank of a vector U, by replacing N
elements which are close to zero with zeros
'''
def reduce_rank(U, N):
    U[-N:] = 0
    return U

def take(n, iterable):
    return list(islice(iterable, n))

def plot_hist(TagFreq):
    fig = plt.figure(figsize=(18,4))
    ax = plt.subplot(111)
    width=1.0

    ax.bar(range(0, len(TagFreq)), TagFreq.values(), width=0.5)
    ax.set_xticks(np.arange(0,len(TagFreq)) + width/2)
    ax.set_xticklabels(TagFreq.keys())
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    #plt.hist(Tags)
    #plt.setp(Tags, rotation=90)
    plt.show()

'''
Given Tags and a unicode character, returns the 
frequency of tags followed and precedded by the tag which
starts with give unicode character.
'''
def freqs_post_pre(Tags, C):
    prec = {}
    follow = {}
    prevTag = Tags[0]

    for tag in Tags[1:]:
        if tag[0] == C:
            if prevTag in prec:
                prec[prevTag] += 1
            else:
                prec[prevTag] = 1
        if prevTag[0] == C:
            if tag in follow:
                follow[tag] += 1
            else:
                follow[tag] = 1
        prevTag = tag
    return prec, follow

'''
Given humongous json and Tags,
Return N sentences which contain that tag.
'''
def sentences_from_tag(data, tag, N):
    rsentences=[]
    r = 0
    for section in data:
        for sentence in data[section]:
            for word in sentence:
                if word['tag'] == tag:
                    rsentences.append(sentence)
                    r = r+1
                    break
                if r > N:
                    return rsentences
'''
Given humongous json,
Return all sentences.
'''
def all_sentences(data, N):
    rsentences=[]
    r = 0
    for section in data:
        for sentence in data[section]:
            rsentences.append(sentence)
            r = r+1
            if (N != 0) and (r > N):
                return rsentences
    return rsentences

fname="../Data/a3-data/train.json"
data = json.load(open(fname))

nsec = 0 # number of sections
nsent= 0 # number of sentences
ntokens = 0 #number of tokens
ntypes = 0 #number of types

Tags=[]
Types={}

for section in data:
    nsec = nsec+1
    for sentence in data[section]:
        nsent = nsent+1
        for word in sentence:
            Tags.append(word['tag'])
            ntokens = ntokens+1
            stype = word['text'].lower()
            if stype not in Types:
                ntypes += 1
                Types[stype] = 1
            else:
                Types[stype] += 1

print "Summary is as follows:"
print "Number of sections:", nsec
print "Number of sentences:", nsent
print "Number of tokens", ntokens
print "Number of types", ntypes
TagFreq = Counter(Tags)
print "Number of Tags:", len(TagFreq)
'''
plot_hist(TagFreq)

precNoun, followNoun = freqs_post_pre(Tags, u'N')
print "Most frequent tags which preced Noun are:", sorted(precNoun.items(), key=operator.itemgetter(1), reverse=True)[0:3]
print "Most frequent tags which follow Noun are:", sorted(followNoun.items(), key=operator.itemgetter(1), reverse=True)[0:3]
plot_hist(precNoun)
plot_hist(followNoun)

precVerb, followVerb = freqs_post_pre(Tags, u'V')
print "Most frequent tags which preced Verb are:", sorted(precVerb.items(), key=operator.itemgetter(1), reverse=True)[0:3]
print "Most frequent tags which follow Verb are:", sorted(followVerb.items(), key=operator.itemgetter(1), reverse=True)[0:3]
plot_hist(precVerb)
plot_hist(followVerb)
'''
sentences = all_sentences(data, 0)
TypetoIndex = {}
TypeFreq = Counter()

for sentence in sentences:
    for word in sentence:
        TypeFreq[word['text'].lower()] += 1

#TypeFreq['ovv'] = 0

index = 0
for wtype in TypeFreq.most_common():
    stype = wtype[0].lower()
    TypetoIndex[stype] = index
    index = index+1

'''
We are interested in only top 10 most frequent words
'''
w1 = 1000

Mostcommon = dict(TypeFreq.most_common(w1))
oovC = w1
#oovR = len(TypetoIndex)-1

#tau = 5
Lcontext =  np.zeros((len(TypetoIndex), w1+1))
Rcontext =  np.zeros((len(TypetoIndex), w1+1))

print "Started computing Lcontext and Rcontext"
PrevToken = '__seq__' 
for sentence in sentences:
    for curw, nextw in zip(sentence, sentence[1:]):
        ctype = curw['text'].lower()
        ntype = nextw['text'].lower()
        curindex = TypetoIndex[ctype]
        nextindex = TypetoIndex[ntype]
        '''
        if TypeFreq[ctype] < tau and TypeFreq[ntype] < tau:
            Lcontext[oovR][oovC] += 1
            Rcontext[oovR][oovC] += 1
            continue
        '''

        #if TypeFreq[ctype] < tau:
        #    Lcontext[nextindex][oovC] += 1
        if ctype in Mostcommon:
            Lcontext[nextindex][curindex] += 1
        else:
            Lcontext[nextindex][oovC] += 1

        #if TypeFreq[ntype] < tau:
        #    Rcontext[curindex][oovC] += 1
        if ntype in Mostcommon:
            Rcontext[curindex][nextindex] += 1
        else:
            Rcontext[curindex][oovC] += 1
        '''
        #stype = word['text'].lower()
        pretindex = TypetoIndex[curw.lower()]
        curindex = TypetoIndex[stype]

        if TypeFreq[stype] < tau and TypeFreq[PrevToken] < tau:
            Lcontext[ovvR][ovvC] += 1
            Rcontext[ovvR][ovvC] += 1
            continue
        #if TypeFreq[stype] < tau:
        #    Rcontext[pretindex][ovvC] += 1
        if curindex < w1:
            Rcontext[pretindex][curindex] += 1
        else:
            Rcontext[pretindex][ovvC] += 1

        #if TypeFreq[PrevToken] < tau:
        #    Lcontext[curindex][ovvC] += 1
        if pretindex < w1:
            Lcontext[curindex][pretindex] += 1
        else:
            Lcontext[curindex][ovvC] += 1
        PrevToken = stype
        '''
print "Done computing Lcontext and Rcontext"
#plt.imshow(Lcontext, cmap='hot', interpolation='nearest')

del sentences

#sns.heatmap(Lcontext, vmin=0, vmax=0.56)
#plt.show()

#sns.heatmap(Rcontext, vmin=0, vmax=0.56)
#plt.show()

#TypetoIndex
#TypeFreq

with open("Frequency", 'wb') as f:
    pickle.dump(TypeFreq, f, -1)
f.close()

del TypeFreq

print "Started computing SVD"
UL, SL, VL = np.linalg.svd(Lcontext, full_matrices=False)
UR, SR, VR = np.linalg.svd(Rcontext, full_matrices=False)
print "Done Computing SVD.."

SL = reduce_rank(SL, w1-100+1)
SR = reduce_rank(SR, w1-100+1)


SLstar = SL*np.identity(len(SL))
SRstar = SR*np.identity(len(SR))

Lstar = UL.dot(SLstar)
Rstar = UR.dot(SRstar)

#sns.heatmap(Rstar, vmin=0, vmax=0.56)
#plt.show()



#sns.heatmap(Rstar, vmin=0, vmax=0.56)
#plt.show()

Ldstar = normalize_row(Lstar)
del Lstar
Rdstar = normalize_row(Rstar)
del Rstar

D = np.concatenate((Ldstar, Rdstar), axis=1)
del Ldstar
del Rdstar
del Lcontext
del Rcontext
del Mostcommon
#k = 500
#nm = kmeans(k)
#centroids = nm.cluster(D, TypeFreq)


with open("Dmatrix", 'wb') as f:
    pickle.dump(D, f, -1)
f.close()
#plt.imshow(D, cmap='hot', interpolation='nearest')
#plt.show()
