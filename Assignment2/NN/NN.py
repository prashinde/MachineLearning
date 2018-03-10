'''
Reference:
https://github.com/rohan-varma/neuralnets/blob/master/NeuralNetwork.py
https://medium.com/@curiousily/tensorflow-for-hackers-part-iv-neural-network-from-scratch-1a4f504dfa8
'''
import random
import numpy as np
import pickle as pk
class NeuralNet:
    def __init__(self, nrclasses=10, nrfeatures=784, nrhunits=30, epoch=3, ll=0.001, nrbatches=1):
        self.nrclasses = nrclasses
        self.nrfeatures = nrfeatures
        self.nrhunits = nrhunits
        self.epoch = epoch
        self.ll = ll
        self.nrbatches = nrbatches

        '''
        weights from input to hidden layer
        +1 for bias
        self.Wih is 30x785
        '''
        self.Wih = np.random.uniform(-1, 1, self.nrhunits*(self.nrfeatures+1))
        self.Wih = self.Wih.reshape(self.nrhunits, self.nrfeatures+1)
        '''
        Weights from hidden to output layer
        self.Who is 10x31
        '''
        self.Who = np.random.uniform(-1, 1, (self.nrhunits+1)*nrclasses)
        self.Who = self.Who.reshape(self.nrclasses, self.nrhunits+1)

    def relu(self, W):
        W[W < 0] = 0
        return W

    def reluprime(self, W):
        return np.where(W > 0, 1, 0)

    def softmax(self, X):
        #print np.sum((np.exp(X-np.max(X))), axis=0)
        smax = np.exp(X-np.max(X))/np.sum((np.exp(X-np.max(X))), axis=0)
        return smax

    def conehot(self, y):
	onehot = np.zeros((10, y.shape[0]))
        for i in range(y.shape[0]):
            onehot[y[i], i] = 1.0
        return onehot

    '''
    Adds column to the begining
    '''
    def addcolumn(self, TD):
        TDN = np.zeros((TD.shape[0], TD.shape[1]+1))
        #TDN = np.ones((TD.shape[0], TD.shape[1]+1))
        TDN[:, 1:] = TD
        return TDN

    '''
    Add row in the begning
    '''
    def addrow(self, X):
        XN = np.zeros((X.shape[0]+1, X.shape[1]))
        #XN = np.ones((X.shape[0]+1, X.shape[1]))
        XN[1:, :] = X
        return XN

    def forward(self, td):
        '''
        Each training example is 1x784 size
        Add bias to training example along a column
        '''
        biasinput = self.addcolumn(td)

        print "Input shape is"
        print biasinput.shape
        '''
        Compute the output of the hidden layer
        training example matrix is lx785
        and Wih is 30x785. We will transpose input.
        T(TD) = 785xl.
        Result is 30xl
        '''
        hout = self.Wih.dot(biasinput.T)

        '''
        Apply Activation function Relu
        '''
        hiddenact = self.relu(hout)

        '''
        Add Bias to activation output
        activation output is 30xl.
        Each column is corresponds to training example.
        Now, we have to apply next weight which is 10x31
        We will add bias along the row, i.e. is each training example.
        '''
        hiddenact = self.addrow(hiddenact)

        '''
        Compute the actual output
        Who is 10x31
        hiddenact is 31xl 
        After applying this, we will get a new matrix of size: 10xl
        For each training example(col), we will have 10 classes(row)
        We will later apply softmax activation on this output
        '''
        fout = self.Who.dot(hiddenact)
        
        '''
        Apply Softmax
        '''
        outact = self.softmax(fout)
        print "*****************"
        print "outact dimentiosn"
        print outact.shape
        print "*******************"

        return biasinput, hout, hiddenact, fout, outact

    def backward(self, binput, hout, hact, fout, fact, y):
        s3 = fact - y
        print "ST*************************"
        print s3.shape
        print fact.shape
        print y.shape
        print "STOP*************************"

        '''
        print "fact[:, 0]"
        print fact[:, 0]

        print "y[:, 0] is:"
        print y[:, 0]

        print "s3: is"
        print s3
        '''
        hout = self.addrow(hout)
        l2loss = self.Who.T.dot(s3)*self.reluprime(hout)
        l2loss = l2loss[1:, :]
        grad1 = l2loss.dot(binput)
        grad2 = s3.dot(hact.T)
        return grad1, grad2

    def calcost(self, predictions, labels):
        cost = -np.sum(labels*np.log(predictions))
        return cost/predictions.shape[1]

    def TrainNet(self, x_train, y_train):
        for i in range(self.epoch):
            '''
            Split in minibatches
            '''
            xmini = np.array_split(x_train, self.nrbatches)
            ymini = np.array_split(y_train, self.nrbatches)
            for xi, yi in zip(xmini, ymini):
                binput, hout, hact, fout, fact = self.forward(xi)
                cost = self.calcost(fact, self.conehot(yi))
                grad1, grad2 = self.backward(binput, hout, hact, fout, fact, self.conehot(yi))
                self.Wih -= (self.ll * grad1)
                self.Who -= (self.ll * grad2)
            #ri = random.randint(0, 99)
            #print fact[:, 2] 
            #print self.conehot(yi)[:, 2]
        return self

training_d = "../Perceptron/processed_data.pkl"
with open(training_d, "rb") as f:
    data=pk.load(f)
td = data['in_train_data']
tl = data['in_train_label']

'''
dummy test data
nrex = 30
td=np.random.uniform(0,1,nrex*784)
tl=np.random.uniform(-1,1,nrex*1)

tl = np.where(tl > 0, 1, 0)
td = td.reshape(nrex, 784)
tl = tl.reshape(nrex, 1)
'''

NN = NeuralNet(nrclasses=10, nrfeatures=784, nrhunits=30, epoch=50, ll=0.001, nrbatches=100)
NN.TrainNet(td, tl)
