import numpy as np

class Neuron:

    def __init__(self, length, flag = False):
        self.bias = np.random.randn()
        #self.bias = 0.1
        if flag:
            self.bias = 0
        self.weightList = []
        self.value = None
        for x in range (length):
            num = np.random.randn()
            #num = 1
            self.weightList.append(num)

    def getBias(self):
        return self.bias

    def setBias(self, n):
        self.bias = n

    def getWeights(self):
        return self.weightList

    def __repr__(self):
        return "VAL: " + str(self.value) + " BIAS: " + str(self.bias) + " WEIGHTS:" + str(self.weightList)

    def setValue(self, n):
        self.value = n

    def getValue(self):
        return self.value