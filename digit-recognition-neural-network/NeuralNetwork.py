import pandas as pds
import math
import numpy as np
import sys
import time

#Constants for network
#Learning rate
LEARNING_RATE = 0.01
#Number of epochs to run
EPOCHS = 80
#Number of images in the batches
BATCH_SIZE = 60
#Nodes for each layer
INPUT_NODES = 784
HIDDEN_NODES1 = 512
HIDDEN_NODES2 = 256
OUTPUT_NODES = 10

#Max digit values
MAX_DIGIT = 9


#Use a network object so I don't have to keep passing the network in and out of functions
class Network:
    #Initialize the network weights and bias
    def __init__(self):
        self.rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        #Weights and biases are random on a normal distribution divided by the square root of half the nodes for the
        # "from" layer
        self.weights = {"w1": np.random.randn(HIDDEN_NODES1, INPUT_NODES) / np.sqrt(INPUT_NODES / 2.0),
                         "w2": np.random.randn(HIDDEN_NODES2, HIDDEN_NODES1) / np.sqrt(HIDDEN_NODES1 / 2.0),
                         "w3": np.random.randn(OUTPUT_NODES, HIDDEN_NODES2) / np.sqrt(HIDDEN_NODES2 / 2.0)
        }
        #Bias only needs to be a vector
        self.bias = {"b1": np.random.randn(HIDDEN_NODES1, 1)/np.sqrt(INPUT_NODES/2.0),
                     "b2": np.random.randn(HIDDEN_NODES2,1)/np.sqrt(HIDDEN_NODES1/2.0),
                     "b3": np.random.randn(OUTPUT_NODES,1)/np.sqrt(HIDDEN_NODES2/2.0)
                     }

    #Define the softmax activation function
    def softmax(self, x):
        return np.exp(x-x.max())/np.sum(np.exp(x-x.max()), axis=0)

    #Define the sigmoid activation function (I kept writing 1- in the denom for some odd reason)
    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    #Define the deriviative of the sigmoid function with the netinputs for back propagation
    def sigmoidPrime(self, x, inp):
        #For some reason it was giving me weird values when I didn't do this
        s = self.sigmoid(x)
        return inp * s * (1 - s)

    #Define forward propagation using the input vector (that contains a batch of images)
    def forwardProp(self, inputVector):
        activations = dict()
        inputs = dict()
        #get all the inputs for the hidden layer 1
        inputs[1] = np.dot(self.weights["w1"], inputVector) + self.bias["b1"]
        #activate all the inputs for hidden layer 1
        activations[1] = self.sigmoid(inputs[1])
        #get all the inputs for hidden layer 2
        inputs[2] = np.dot(self.weights["w2"], activations[1]) + self.bias["b2"]
        #activate all the inputs for hidden layer 2
        activations[2] = self.sigmoid(inputs[2])
        #get all the inputs for output layer
        inputs[3] = np.dot(self.weights["w3"], activations[2]) + self.bias["b3"]
        #activate output layer using softmax
        activations[3] = self.softmax(inputs[3])

        return activations, inputs


    #Define back propagation. this was the hardest one
    def backProp(self, inputVector, answerVector, activations, inputs):
        #calculate number in batch
        numberInBatch = inputVector.shape[1]
        #get error for output layer
        dIn3 = activations[3] - answerVector
        #calculate the average delta weight change between hidden layer 2 and output layer 3
        dWeight3 = np.dot(dIn3, activations[2].T) / numberInBatch
        #calculate the average delta bias change for output layer
        dBias3 = np.sum(dIn3, axis=1, keepdims=True)/numberInBatch

        #calculate the delta activations for output layer
        dActivation2 = np.dot(self.weights["w3"].T, dIn3)
        #calculate the error for hidden layer 2
        dIn2 = self.sigmoidPrime(inputs[2], dActivation2)
        #calculate the average delta weight change between hidden layer 1 and hidden layer 2
        dWeight2 = np.dot(dIn2, activations[1].T)/numberInBatch
        #calculate the average delta bias change for hidden layer 2
        dBias2 = np.sum(dIn2, axis=1, keepdims=True)/numberInBatch

        #calculate the delta activations for hidden layer 2
        dActivation1 = np.dot(self.weights["w2"].T, dIn2)
        #calculate the error for hidden layer 1
        dIn1 = self.sigmoidPrime(inputs[1], dActivation1)
        #calculate the average delta weight change between input layer and hidden layer 1
        dWeight1 = np.dot(dIn1, inputVector.T)/numberInBatch
        #calculate the average delta bias change for hidden layer 1
        dBias1 = np.sum(dIn1, axis=1, keepdims=True)/numberInBatch

        #update weights and bias using average deltas and learning rate
        self.weights["w1"] = self.weights["w1"] - self.rate * dWeight1
        self.weights["w2"] = self.weights["w2"] - self.rate * dWeight2
        self.weights["w3"] = self.weights["w3"] - self.rate * dWeight3
        self.bias["b1"] = self.bias["b1"] - self.rate * dBias1
        self.bias["b2"] = self.bias["b2"] - self.rate * dBias2
        self.bias["b3"] = self.bias["b3"] - self.rate * dBias3


    #Define learn function (this will train the network)
    def learn(self, trainingSet, answerSet):
        #Loop for the number of epochs set
        for ind in range(self.epochs):
            #rearrange the training set at the beginning of each epoch
            order = np.arange(len(trainingSet[1]))
            np.random.shuffle(order)
            trainingSet = trainingSet[:,order]
            answerSet = answerSet[:,order]
            #this should help us learn better
            #batch the set so that it's in batches of BATCH_SIZE
            batch = getBatch(trainingSet, answerSet, self.batch_size)
            #Each batch contains a set of image vectors and a set of answer vectors (one-hot coded)
            for (inputVec, answerVec) in batch:
                #forward propogate a batch and get the activations and net inputs for the batch
                activations, inputs = self.forwardProp(inputVec)
                #back propagate and change weights and bias based on average changes
                self.backProp(inputVec, answerVec, activations, inputs)
            print(self.accuracy(trainingSet, answerSet))

    #Define function for calculating accuracy after an epoch
    def accuracy(self, trainingSet, answerSet):
        #Pass the whole training set into the network
        activations, inputs = self.forwardProp(trainingSet)
        #Get all the final activations
        output = activations[3]
        #Calculate the predictions using argmax
        prediction = np.argmax(output, axis=0)
        #Calculate correct percentage by taking the average of all "true" values
        return np.mean(prediction == np.argmax(answerSet, axis=0))

    def test(self, testSet, fileOut):
        activations, inputs, = self.forwardProp(testSet)
        predictions = np.argmax(activations[3], axis=0)
        for pred in predictions:
            print(pred, file=fileOut)


#Define splitting the training set into batches
def getBatch(trainingSet, answerSet, batchSize):
    #get total number of images
    numImages = trainingSet.shape[1]
    batches = list()
    #calculate how many batches we can create (there may be some leftover)
    numBatches = math.floor(numImages/batchSize)
    #create the batches
    for ind in range(numBatches):
        #slice the sets and add them as tuples to the batches list
        imageVec = trainingSet[:, ind * batchSize:(ind+1)*batchSize]
        answerVec = answerSet[:, ind * batchSize:(ind+1)*batchSize]
        batches.append((imageVec, answerVec))
    #if there's any leftover, put them into the final batch, regardless of size
    if numImages % batchSize != 0:
        imageVec = trainingSet[:, batchSize*math.floor(numImages/batchSize):numImages]
        answerVec = answerSet[:, batchSize*math.floor(numImages/batchSize):numImages]
        batches.append((imageVec, answerVec))

    return batches

if __name__ == "__main__":
    # LEFTOVER FROM LOCALMACHINE TRAINING
    # trainSetFile = "train_image.csv"
    # trainLabelFile = "train_label.csv"
    # testSetFile = "test_image.csv"

    #Parse the command line arguments
    trainSetFile = sys.argv[1]
    trainLabelFile = sys.argv[2]
    testSetFile = sys.argv[3]

    #load the files
    trainingSet = pds.read_csv(filepath_or_buffer=trainSetFile, header=None)
    answerSet = pds.read_csv(filepath_or_buffer=trainLabelFile, header=None)
    testSet = pds.read_csv(filepath_or_buffer=testSetFile, header=None)

    #transpose the sets so they match what we need for our network
    trainingSet = trainingSet.T
    answerSet = answerSet.T
    testSet = testSet.T

    #extract trainingSet and answerSet values
    trainingSet = trainingSet.values
    answerSet = answerSet.values

    #create the answer vectors: one-hot coded vectors for each answer
    answerVector = np.zeros((answerSet.size, MAX_DIGIT+1))
    answerVector[np.arange(answerSet.size), answerSet] = 1
    answerVector = answerVector.T

    #initialize the network
    network = Network()
    #train the network
    network.learn(trainingSet[:,:10000], answerVector[:,:10000])

    #run the network against the test set
    fileOut = open("test_predictions.csv","w")
    network.test(testSet, fileOut)
    fileOut.close()
