import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils as skl
from scipy.special import expit

class MLP():
	def __init__(self, inLength, hid1Length, outLength):

		#network configuration
		self.inLength = inLength
		self.hid1Length = hid1Length
		self.outLength = outLength

		#initializing weights and biases
		self.W1, self.B1 = self.initializingLayer(hid1Length, inLength)
		self.Wout, self.Bout = self.initializingLayer(outLength, hid1Length)

	def sigmoidFunc(self, x, deriv):
		#sigmoid derivative
		if deriv == True:
			z = x - np.power(x, 2)
			#z = s*(1 - s)
			return z
		#sigmoid function
		else:
			z = expit(x)
			return z

	def tanhFunc(self, x, deriv):
		#tanh function
		if deriv == False:
			return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
		#tanh derivative
		else:
			return (1.0 - (self.tanhFunc(x, False)**2))

	def reluFunc(self, x, deriv):
		#relu derivative
		if deriv == True:
			return np.heaviside(x, 1)
		#relu function
		else:
			return np.maximum(x, 0)

	def initializingLayer(self, layerLength, prevLayerLength):
		#fill parameters with random values between -0.5 and 0.5
		W = np.random.ranf((layerLength, prevLayerLength)) - 0.5
		B = np.random.ranf((layerLength, 1)) - 0.5

		return W, B

	def forwardProp(self, x):
		#first hidden layer
		self.x1 = np.dot(self.W1, x) + self.B1 	#each train sample is a column matrix
		self.a1 = self.sigmoidFunc(self.x1, deriv = False)
		#output layer
		self.x2 = np.dot(self.Wout, self.a1) + self.Bout
		y = self.sigmoidFunc(self.x2, deriv = False)

		return y

	def backProp(self, X, Y, parameters):
		learningRate, epochs, batchSize = parameters

		errorList = []
		qtBatch = 0
		for epoch in range(0, epochs):
			#randomize train samples to avoid overfitting
			X, Y = skl.shuffle(X.T, Y.T)	#shuffle lines
			X = X.T		#put each train sample into a column
			Y = Y.T

			#batches
			qtCases = int(len(X[0]))
			qtBatch = int(int(len(X[0])/batchSize))
			if (int(len(X[0]))%batchSize > 0):
				qtBatch+=1

			for batch in range(0, qtBatch):
				if batch == qtBatch:
					Xbatch = X[0:len(X), batch*batchSize:batch*batchSize+(qtCases - batch*batchSize)]
					Ybatch = Y[0:len(Y), batch*batchSize:batch*batchSize+(qtCases - batch*batchSize)]
					batchSize = qtCases - batch*batchSize
				else:
					Xbatch = X[0:len(X), batch*batchSize:batch*batchSize+batchSize]
					Ybatch = Y[0:len(Y), batch*batchSize:batch*batchSize+batchSize]

				res = self.forwardProp(Xbatch) # h2 x batchSize
				costsum = 1/batchSize*float(np.sum((np.sum((res - Ybatch)**2, axis = 0, keepdims = True)), axis = 1, keepdims = True))

				dZout = -2*(res - Ybatch)*(self.sigmoidFunc(res, deriv = True)) # h2 x batchSize

				# dZout(h2 x batchSize) * A1(h1 x batchSize).T
				dWout = (1/batchSize)*np.dot(dZout, self.a1.T)

				dbout = (1/batchSize)*np.sum(dZout, axis = 1, keepdims = True) # h2 x 1

				dZ1 = np.dot(self.Wout.T, dZout)*(self.sigmoidFunc(self.a1, deriv = True)) # h1 x batchSize

				#dZ1(h1 x batchSize) * X(x x batchSize).T
				dW1 = (1/batchSize)*np.dot(dZ1, Xbatch.T)

				db1 = (1/batchSize)*np.sum(dZ1, axis = 1, keepdims = True) # h1 x 1	

				#updating parameters after each processed mini-batch
				self.Wout += learningRate*dWout
				self.Bout += learningRate*dbout
				self.W1 += learningRate*dW1
				self.B1 += learningRate*db1
				
				#average cost of the mini-batch processed
				errorList.append(costsum)
				#print(error/qtCases)

		#plot average cost per mini batch
		plt.figure(1)
		plt.plot(range(0,epochs*qtBatch), errorList, 'm')
		plt.xlabel('Mini-batch')
		plt.ylabel('Average cost')
		plt.title('Average cost with Mini-Batch Gradient Descent')

#dataset reading
data = pd.read_csv('files/iris.csv', sep = ',', header = None)
#print(data.shape)
data = data.to_numpy()
Xtrain = data[0:, 0:4].T
Ytrain = data[0:, 4:7].T

#training neural network
mlp = MLP(4, 8, 3)	#inLength, hidLength, outLength
mlp.backProp(Xtrain, Ytrain, [0.01, 10000, 15])	#learningRate, amount of epochs, mini-batch size

#testing neural network
Ytest = mlp.forwardProp(Xtrain)

#writing results to csv file
pd.DataFrame(Ytest.T).to_csv("files/testResults.csv", header=None, index=None)

plt.show()