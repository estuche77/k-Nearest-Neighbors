'''
Created on 16 Feb 2018

@author: estuche
'''

import numpy as np
import time

fileNames = ["cifar-10-batches-py/data_batch_1",
             "cifar-10-batches-py/data_batch_2",
             "cifar-10-batches-py/data_batch_3",
             "cifar-10-batches-py/data_batch_4",
             "cifar-10-batches-py/data_batch_5"]

class kNearestNeighbors:
    
    def train(self, X, Y):
        self.X = X
        self.Y = Y
    
    def classifierManhattan(self, X, k):
        itemsCount = X.shape[0]
        classificatedY = np.zeros(itemsCount, dtype = 'object')
        for i in range(itemsCount):
            print(i)
            distances = np.sum(np.abs(self.X - X[i,:]), axis = 1)
            classificatedY[i] = self.__classify(distances, k)
        
        return classificatedY
    
    def classifierChebychev(self, X, k):
        itemsCount = X.shape[0]
        classificatedY = np.zeros(itemsCount, dtype = 'object')
        for i in range(itemsCount):
            print(i)
            distances = np.max(np.abs(self.X - X[i,:]), axis = 1)
            classificatedY[i] = self.__classify(distances, k)
        
        return classificatedY
    
    def classifierLevenshtein(self, X, k):
        itemsCount = X.shape[0]
        classificatedY = np.zeros(itemsCount, dtype = 'object')
        for i in range(itemsCount):
            print(i)
            distances = np.count_nonzero(self.X - X[i,:], axis = 1)
            classificatedY[i] = self.__classify(distances, k)
        
        return classificatedY
    
    def __classify(self, distances, k):
        if k < 3:
            smaller_distance_index = np.argmin(distances)
            return self.Y[smaller_distance_index]
        elif k > 2:
            smaller_distance_index = np.argpartition(distances, k)[:k]
            print(smaller_distance_index)
            votes = self.Y[smaller_distance_index]
            return np.bincount(votes).argmax()

def unpickle(fileName):
    import pickle
    with open(fileName, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary

def loadTrainingData():
    xs = []
    ys = []
    for file in fileNames:
        d = unpickle(file)
        x = d[b'data']
        y = d[b'labels']
        xs.append(x)
        ys.append(y)
    
    return np.concatenate(xs), np.concatenate(ys)

def loadTestData():
    test = unpickle("cifar-10-batches-py/test_batch")
    return test[b'data'], test[b'labels']

def getAccuracy(clasificationSet, testSet):
    total = len(testSet)
    results = clasificationSet - testSet
    correct = total - np.count_nonzero(results)
    return (correct/float(total)) * 100.0

def log(string):
    print(string)
    file.write(string + "\n")

file = open("k Nearest Neighbors.log","w")

if __name__ == '__main__':
    
    log("Loading data...")
    startTime = time.time()
    
    trainData, trainLabels = loadTrainingData()
    testData, testLabels = loadTestData()
    
    endTime = time.time()
    log(str(endTime - startTime) + " seconds\n")
    
    log("Training...")
    kNN = kNearestNeighbors()
    kNN.train(trainData, trainLabels)
    log("No seconds\n")
    
    for k in range(3):
    
        log("Using Manhattan with k = " + str(k + 1))
        startTime = time.time()
        clasificationLabels = kNN.classifierManhattan(testData, k + 1)
        endTime = time.time()
        log("Accuracy: " + str(getAccuracy(testLabels, clasificationLabels)))
        log("Elapsed time: " + str(endTime - startTime))
        
        log("Using Chebychev with k = " + str(k + 1))
        startTime = time.time()
        clasificationLabels = kNN.classifierChebychev(testData, k + 1)
        endTime = time.time()
        log("Accuracy: " + str(getAccuracy(testLabels, clasificationLabels)))
        log("Elapsed time: " + str(endTime - startTime))
        
        log("Using Levenshtein with k = " + str(k + 1))
        startTime = time.time()
        clasificationLabels = kNN.classifierLevenshtein(testData, k + 1)
        endTime = time.time()
        log("Accuracy: " + str(getAccuracy(testLabels, clasificationLabels)))
        log("Elapsed time: " + str(endTime - startTime))
    
    file.close()
