import json
import numpy as np
import operator
import pickle
import matplotlib.pyplot as plt
from collections import Counter

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
TagFreq = Counter(Tags)
#plot_hist(TagFreq)
print "Number of Tags:", len(TagFreq)
'''
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

sentences = sentences_from_tag(data, u'IN', 10)
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
with open("toy_data", 'wb') as f:
    pickle.dump(sentences, f)
