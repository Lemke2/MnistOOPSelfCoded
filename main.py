import numpy as np
import pandas as pd
from layer import Layer


class MnistNetwork:

    def __init__(self, dimensions):
        # ucitavanje podataka - podela na trening i test podatke.
        self.yTest = []
        self.yTraining = []
        self.xTest = []
        self.xTraining = []
        self.data = pd.read_csv("./data/train.csv")
        self.dataTest = np.array(self.data.iloc[:2000])
        self.dataTrain = np.array(self.data.iloc[2000:])


        # inicijalizacija biasa i weightova
        self.dimensions = dimensions
        self.errors = []
        self.layers = []
        self.learningRate = 0.5
        self.preactivationValues = []

        for x in range(len(dimensions)):
            if x == 0:
                layer = Layer(dimensions[x], dimensions[x+1], True)
            elif x == len(dimensions) - 1:
                layer = Layer(dimensions[x], 0)
            else:
                layer = Layer(dimensions[x], dimensions[x+1])
            self.layers.append(layer)

        self.learn(20)


    def learn(self, epochs):
        # self.forward([0.1, 0.1, 0.1, 0.1, 0.1])
        # for layer in self.layers:
        #     for neuron in layer.getNeurons():
        #         print(neuron)
        # print(self.preactivationValues)
        # self.errorOutputLayer([0,1])
        # print(self.errors)
        # self.errorsPreviousLayers()
        # print(self.errors)
        # self.errors.reverse()
        # print(self.errors)
        # self.updateBiases()
        # for layer in self.layers:
        #     for neuron in layer.getNeurons():
        #         print(neuron)
        # self.updateWeights()
        # for layer in self.layers:
        #     for neuron in layer.getNeurons():
        #         print(neuron)
        for e in range(epochs):
            np.random.shuffle(self.dataTrain)
            for i in range(40000):
                self.yTraining.append(self.dataTrain[i][0])
                self.xTraining.append(self.dataTrain[i][1:])
            #learn funkcija
            for i in range(len(self.yTraining)):
                example = self.normalize(self.xTraining[i])
                self.forward(example)
                solution = self.decodeSolution(self.yTraining[i])
                self.errorOutputLayer(solution)
                self.errorsPreviousLayers()
                #posto smo unatraske punili self.errors sada ga obrcemo
                self.errors.reverse()
                self.updateBiases()
                self.updateWeights()
                self.errors = []
                if i % 200 == 0:
                    print("EPOCH: " + str(e + 1) + " | EXAMPLE: " + str(i))
            self.evaluate(e)
            self.yTraining = []
            self.xTraining = []

    def forward(self, example):
        #prvi sloj
        for i in range(len(example)):
            self.layers[0].getNeurons()[i].setValue(example[i])
        #forward aj = sigmoid(xiwij + b)
        weightedSum = 0
        preacVal = []
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].getNeurons())):
                neuron = self.layers[i].getNeurons()[j]
                for prevNeuron in self.layers[i-1].getNeurons():
                    weightedSum += prevNeuron.getValue() * prevNeuron.getWeights()[j]
                preacVal.append(weightedSum + neuron.getBias())
                neuron.setValue(self.sigmoid(weightedSum + neuron.getBias()))
                weightedSum = 0
            self.preactivationValues.append(preacVal)
            preacVal = []

    def errorOutputLayer(self, solution):
        errors = []
        # quadratic cost funkcija ima parcijalni izvod C/aj = (aj - yi) * sigmoidPrime(z)
        for i in range(len(self.layers[-1].getNeurons())):
            neuron = self.layers[-1].getNeurons()[i]
            error = (neuron.getValue() - solution[i]) * self.sigmoidPrime(self.preactivationValues[-1][i])
            errors.append(error)
        self.errors.append(errors)

    def errorsPreviousLayers(self):
        #pocinjemo od pretposlednjeg sloja jer je poslednji vec izracunat, poslednji sloj greski je prvi (nulti) u self.errors pa errore redjamo unatraske
        i = len(self.layers) - 2
        br = 0
        errorsCurrentLayer = []
        error = 0
        while i>0:
            errorsNextLayer = self.errors[br]
            for j in range(len(self.layers[i].getNeurons())):
                neuron = self.layers[i].getNeurons()[j]
                for z in range(len(neuron.getWeights())):
                    error += neuron.getWeights()[z] * errorsNextLayer[z]
                error = error * self.sigmoidPrime(self.preactivationValues[i - 1][j])
                errorsCurrentLayer.append(error)
                error = 0
            self.errors.append(errorsCurrentLayer)
            errorsCurrentLayer = []
            i-=1
            br+=1

    def updateBiases(self):
        #bias gradient je parcijalni izvod cost funkcije sa biasom, sto je jednako nasem "erroru" odgovarajuceg neurona
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].getNeurons())):
                neuron = self.layers[i].getNeurons()[j]
                neuron.setBias(neuron.getBias() - (self.errors[i-1][j] * self.learningRate))

    def updateWeights(self):
        #weight gradient je parcijalni izvod cost funkcije sa weightom, sto je nasa ulazna vrednost neurona * njegov izlazni "error"
        for i in range(len(self.layers) - 1):
            for j in range(len(self.layers[i].getNeurons())):
                neuron = self.layers[i].getNeurons()[j]
                for k in range(len(neuron.getWeights())):
                    gradient = neuron.getValue() * self.errors[i][k]
                    neuron.getWeights()[k] -= gradient * self.learningRate

    def evaluate(self, e):
        sum = 0
        np.random.shuffle(self.dataTest)
        for i in range(2000):
            self.yTest.append(self.dataTest[i][0])
            self.xTest.append(self.dataTest[i][1:])
        for i in range(len(self.yTest)):
            example = self.normalize(self.xTest[i])
            self.forward(example)
            result = self.decodeResult()
            if result == self.yTest[i]:
                sum+=1
        print("EVALUATION: " + str(e + 1) + " : " + str(sum * 1.0 / 2000))
        self.xTest = []
        self.yTest = []

    def decodeSolution(self, solution):
        niz = np.zeros(len(self.layers[-1].getNeurons()))
        niz[solution] = 1
        return niz

    def decodeResult(self):
        max = self.layers[-1].getNeurons()[0].getValue()
        maxi = 0
        for i in range(len(self.layers[-1].getNeurons())):
            neuron = self.layers[-1].getNeurons()[i].getValue()
            if neuron > max:
                max = neuron
                maxi = i
        return maxi

    def sigmoid(self, value):
        if -value > np.log(np.finfo(type(value)).max):
            return 0.0
        a = np.exp(-value)
        return 1.0 / (1.0 + a)

    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def normalize(self, example):
        normalized = example / 255.0
        return normalized

if __name__ == '__main__':
    mnist = MnistNetwork([784,10,10])
    #mnist = MnistNetwork([5,3,2])