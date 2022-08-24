from neuron import Neuron


class Layer:

    def __init__(self, length, lenNext, flag = False):
        self.length = length
        self.neurons = []
        self.biases = []
        if flag:
            for x in range(length):
                neuron = Neuron(lenNext, flag)
                self.neurons.append(neuron)
        else:
            for x in range(length):
                neuron = Neuron(lenNext)
                self.neurons.append(neuron)
                self.biases.append(neuron.getBias())

    def getLen(self):
        return self.length

    def getNeurons(self):
        return self.neurons

    def getNthNeuron(self, n):
        return self.neurons[n]

    def setNthNeuron(self, i, n):
        self.neurons[i].setBias(n)

    def getBiases(self):
        return self.biases

    def __repr__(self):
        return "" + str(self.length)