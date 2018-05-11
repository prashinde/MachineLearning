from math import exp, log
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b

class conditional_random_field:
    def __init__(self, features):
        self.features = features

    def forward_prop(self, w, t, x):
        n = len(x)
        start = time.time()
        mem = np.zeros((n, 26), dtype=np.float128)
        for i in range(0, 26):
            mem[0, i] = np.exp(np.dot(w[i], x[0]))

        for i in range(1, n):
            for j in  range(0, 26):
                #transition = sum(mem[i-1, k]*np.exp(t[k, j]) for k in range(0, 26))
                #mem[i, j] = transition*np.exp(np.dot(w[j], x[i]))
                mem[i, j] = sum(mem[i-1, k]*np.exp(t[k, j]) for k in range(0, 26))*np.exp(np.dot(w[j], x[i]))
        end = time.time()
        return mem
    
    def backward_prop(self, w, t, x):
        n = len(x)
        mem = np.zeros((n, 26), dtype=np.float128)

        for i in range(0, 26):
            mem[-1, i] = np.exp(np.dot(w[i], x[0]))

        for i in range(n-2, -1, -1):
            for j in  range(0, 26):
                #transition = sum(mem[i+1, k]*np.exp(t[j, k]) for k in range(0, 26))
                #mem[i, j] = transition*np.exp(np.dot(w[j], x[i]))
                mem[i, j] = sum(mem[i+1, k]*np.exp(t[j, k]) for k in range(0, 26))*np.exp(np.dot(w[j], x[i]))
        return mem

    def belief_prop(self, w, t, x):
        fbelief = self.forward_prop(w, t, x)
        bbelief = self.backward_prop(w, t, x)
        return fbelief, bbelief

    def gradient(self, w, t, x, y):
        n = len(x)

        start = time.time()
        fbelief, bbelief = self.belief_prop(w, t, x)

        z = sum(fbelief[-1])
        uw = np.empty((26, 128), dtype=np.float64)
        for i in range(0, 26):
            potfunc = x[0]*bbelief[0, i] + x[-1]*fbelief[-1, i]
            for j in range(1, n-1):
                potfunc += x[j]*(fbelief[j, i]*bbelief[j, i]/np.exp(np.dot(w[i], x[j])))
            uw[i] = sum(x[j] for j in range(0, n) if y[j]==i) - potfunc/z

        ut = np.zeros((26, 26), dtype=np.float64)
        for i in range(1, n):
            for j in range(0, 26):
                for k in range(0, 26):
                    ut[j, k] -= fbelief[i-1, j]*bbelief[i, k]*np.exp(t[j ,k])
        ut /= z
        for i in range(1, n):
            ut[y[i-1], y[i]] += 1
        end = time.time()
        return uw, ut

    def probs(self, w, t, x, y):
        pfunc = sum(np.dot(w[y[i]], x[i]) for i in range(0, len(x)))
        pfunc += sum(t[y[i], y[i+1]] for i in range(0, len(x)-1))
        fbelief = self.forward_prop(w, t, x)
        z = sum(fbelief[-1])

        ret = np.log(np.exp(pfunc)/z)
        return ret

    def objective(self, theta):
        c = 1000
        w = np.reshape(theta[:26*128], (26, 128))
        t = np.reshape(theta[26*128:], (26, 26))

        score = -c * sum(self.probs(w, t, x, y) for k,(x,y) in \
                self.features.iteritems())/len(self.features)
        score += sum(np.linalg.norm(x)**2 for x in w)/2
        score += sum(sum(x**2 for x in row) for row in t)/2
        return score

    def dobjective(self, theta):
        c = 1000
        w = np.reshape(theta[:26*128], (26, 128))
        t = np.reshape(theta[26*128:], (26, 26))

        UW, UT = [], []

        for k, (x,y) in self.features.iteritems():
            nw, nt = self.gradient(w, t, x, y)
            UW.append(nw)
            UT.append(nt)
        UW = -c * sum(UW)/len(self.features) + w
        UT = -c * sum(UT)/len(self.features) + t
        return np.concatenate((np.reshape(UW, 26*128), np.reshape(UT, 26**2)))

    def optimize(self):
        theta, minobj,_ = fmin_l_bfgs_b(func=self.objective, x0=np.zeros(26*128+26*26),\
                fprime=self.dobjective,  iprint=99)
        
        p = 128
        w = np.reshape(theta[:26 * p], (26, p))
        t = np.reshape(theta[26 * p:], (26, 26))
        np.savetxt("learnt_w.txt", w)
        np.savetxt("learnt_t.txt", t)
        print "Minimum objective", minobj

    def max_sum(self, x, w, t):
        m = len(x)
        dp_argmax = np.zeros((m, 26), dtype=np.int)  # backward pointers
        dp_vmax = np.zeros((m, 26), dtype=np.float64)  # max values corresponding to the pointers
        for i in range(0, 26):  # first row of the dp table
            dp_vmax[0, i] = np.dot(w[i], x[0])
        for i in range(1, m):  # for all rows of the dp table
            for j in range(0, 26):  # for each current letter
                prev = np.copy(dp_vmax[i - 1])  # the previous row of the dp table
                for k in range(0, 26):  # for each previous letter
                    prev[k] += t[k, j]  # the dot product shall be added later since it is a constant wrt argmax_k
                k_max = np.argmax(prev)
                dp_argmax[i, j] = k_max  # point to the previous link. note that @dp_argmax[0] is empty
                dp_vmax[i, j] = prev[k_max] + np.dot(w[j], x[i])  # the dot product is for the current letter
        word = np.zeros(m, dtype=np.int)  # backtrack the dp tables
        print(dp_vmax[m-1])
        word[m-1] = np.argmax(dp_vmax[m-1])  # the last choice depends on the min_obj
        for i in range(m-1, 0, -1):  # all previous choices have been calculated
            word[i-1] = dp_argmax[i][word[i]]
        return word

    def predict(self, data):
        w = np.loadtxt("learnt_w.txt", dtype=np.float64)
        t = np.loadtxt("learnt_t.txt", dtype=np.float64)
        
        predictions = []
        correct_letter, correct_word = 0, 0
       
        for k,(x, y) in data.iteritems():
            infer = self.max_sum(x, w, t).tolist()
            correct_letter += sum(1 if infer[i] == y[i] else 0 for i in range(0, len(y)))
            if infer == y.tolist():
                correct_word += 1
            predictions += infer
        print("letter-wise accuracy={}%".format(100 * correct_letter / len(predictions)))
        print("word-wise accuracy={}%".format(100 * correct_word / len(data)))
