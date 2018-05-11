import numpy as np

states = {0:"START", 1:"X", 2:"Y", 3:"END"}
observable = {0:"a", 1:"b", 2:"c", 3:"d", 4:"#"}
observableR = {"a":0, "b":1, "c":2, "d":3, "#":4}

transitions = [[0, 0.7, 0.2, 0.1], [0, 0.15,0.8,0.05], [0, 0.6,0.35,0.05], [0, 0, 0, 0]]
emissions = [[0, 0, 0, 0, 0],[0.7, 0.2, 0.05, 0.05, 0], [0.2, 0.6, 0.1, 0.1, 0], [0, 0, 0, 0, 1]]

sequence = "dab#"

nstates = 4

#V = [[float(0) for y in range(nstates)] for x in range(len(sequence)+1)]
#B = [[0 for y in range(nstates)] for x in range(len(sequence)+1)]

V = np.zeros((len(sequence)+1, nstates), dtype=np.float64)
B = np.zeros((len(sequence)+1, nstates), dtype=int)
V[0][0] = float(1)

print "Transition Table:\n"
for row in transitions:
    print row
    print "\n"

print "Emmision Table:\n"
for row in emissions:
    print row
    print "\n"

idx = 1
for c in sequence:
    onum = observableR[c]
    for state in range(nstates):
        oemissionp = emissions[state][onum]
        for oldstate in range(nstates):
            stransp = transitions[oldstate][state]
            mprob = stransp * oemissionp * V[idx-1][oldstate]
            if mprob > V[idx][state]:
                V[idx][state] = mprob
                B[idx][state] = oldstate
    idx = idx + 1

print "V table:\n"
for row in V:
    print row
    print "\n"

print "V table summation:", sum(map(sum, V))
print "B table:\n"
for row in B[1:]:
    print row
    print "\n"

mlikely = []

rnum = 4
ptr = B[rnum][3]
mprob = V[rnum][3]
rnum -= 1
for row in B[3::-1]:
    mprob += V[rnum][ptr]
    ptr = B[rnum][ptr]
    rnum -= 1
    if rnum == 0:
        break

print "Marginal probability:", mprob

print "Most likley sequence...."
for row in V:
    mlikely.append(np.argmax(row))

for state in mlikely:
    print states[state]

jdist = float(1)
idx = 1
for state in mlikely[1:]:
    jdist = jdist * V[idx][state]
    idx = idx+1

print "Joint probability of most likley state:", jdist

post = 0
for row in V:
    post = post+row[mlikely[2]]

print "Posterior probability is:", post
