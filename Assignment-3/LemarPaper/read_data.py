import json
import numpy as np
np.set_printoptions(threshold=np.nan)
import operator
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import matrix_rank

'''
Normalizes each row of a given matrix to unit length
'''
def normalize_row(C):
    l2 = np.atleast_1d(np.linalg.norm(C, 2, 1))
    l2[l2==0] = 1
    return C / np.expand_dims(l2, 1)
    #l2 = C.sum(axis=1)
    #l2[l2 == 0] = 1
    #return C/l2[:, None]

'''
Reduce rank of a vector U, by replacing N
elements which are close to zero with zeros
'''
def reduce_rank(U, N):
    Uprime = U.flatten()
    Uprime.sort()
    return np.where(U <= Uprime[-N], 0, U)

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
            for word in sentence:
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
            if(word['text'].isalpha()):
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
'''
TagFreq = Counter(Tags)
plot_hist(TagFreq)
print "Number of Tags:", len(TagFreq)
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
#sentences = sentences_from_tag(data, u'IN', 10)
sentences = all_sentences(data, 0)
print "Sentences size", len(sentences)
'''
for sentence in sentences:
    for word in sentence:
        print word['text'],
    print "\n"
    print "-------------------"
    for word in sentence:
        print word['tag'],
    print "\n"
    print "*********************"

    for word in sentence:
        if word['tag'] == u'IN':
            print word['text'],
            print word['tag']
    print "++++++++++++++++++++++++++++++++++"
'''

TypetoIndex = {}
TypeFreq = Counter()

for sentence in sentences:
    for word in sentence:
        if (word['text'] == '__SEQ__') or (word['text'].isalpha()):
            TypeFreq[word['text']] += 1

index = 0
it = 0
for wtype in TypeFreq.most_common():
    if wtype[0] not in TypetoIndex:
        TypetoIndex[wtype[0]] = index
        index = index+1

'''
We are interested in only top 10 most frequent words
'''
w1 = 1000

'''
print most common w1 words
'''
Lcontext =  np.zeros((len(TypetoIndex), w1))
Rcontext =  np.zeros((len(TypetoIndex), w1))

print "Started computing Lcontext and Rcontext"
PrevToken = '__SEQ__' 
for sentence in sentences:
    for word in sentence:
        if word['text'] != '__SEQ__' and not word['text'].isalpha():
            continue
        pretindex = TypetoIndex[PrevToken]
        curindex = TypetoIndex[word['text']]
        if curindex < w1:
            Rcontext[pretindex][curindex] += 1
        if pretindex < w1:
            Lcontext[curindex][pretindex] += 1
        PrevToken = word['text']

print "Done computing Lcontext and Rcontext"
#for i in range(len(Lcontext)):
#    print "Word=", (TypeFreq.most_common())[i]," ", Lcontext[i]
#print Lcontext
#print "********************************"
#print Rcontext
#plt.imshow(Lcontext, cmap='hot', interpolation='nearest')
#plt.show()
print "Lcontext shape=", Lcontext.shape
del sentences
#TypetoIndex
#TypeFreq

with open("Frequency", 'wb') as f:
    pickle.dump(TypeFreq, f, -1)
f.close()

print "Started computing SVD"
UL, SL, VL = np.linalg.svd(Lcontext, full_matrices=False)
UR, SR, VR = np.linalg.svd(Rcontext, full_matrices=False)
print UL.shape
print "Done Computing SVD.."
del Lcontext
del Rcontext

SL = reduce_rank(SL, 100)
SR = reduce_rank(SR, 100)

SLstar = SL*np.identity(len(SL))
SRstar = SR*np.identity(len(SR))

Lstar = UL.dot(SLstar)
Rstar = UR.dot(SRstar)

Ldstar = normalize_row(Lstar)
del Lstar
Rdstar = normalize_row(Rstar)
del Rstar

D = np.concatenate((Ldstar, Rdstar), axis=1)
del Ldstar
del Rdstar

print D.shape
#k = 500
#nm = kmeans(k)
#centroids = nm.cluster(D, TypeFreq)


with open("Dmatrix", 'wb') as f:
    pickle.dump(D, f, -1)
f.close()
#plt.imshow(D, cmap='hot', interpolation='nearest')
#plt.show()
