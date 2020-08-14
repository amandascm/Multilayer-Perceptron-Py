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
		for epoch in range(0, epochs):
			#randomize train samples to avoid overfitting
			X, Y = skl.shuffle(X.T, Y.T)	#shuffle lines
			X = X.T		#put each train sample into a column
			Y = Y.T

			#batches
			for batch in range(0, int(len(X[0])/batchSize)):
				Xbatch = X[0:len(X), batch*batchSize:batch*batchSize+batchSize]
				Ybatch = Y[0:len(Y), batch*batchSize:batch*batchSize+batchSize]

				#initialize matrix of gradient's sum for each weight and bias
				W1Grad = np.zeros([self.hid1Length, self.inLength])
				B1Grad = np.zeros([self.hid1Length, 1])
				WoutGrad = np.zeros([self.outLength, self.hid1Length])
				BoutGrad = np.zeros([self.outLength, 1])

				cost = 0
				qtCases = 0
				for i in range(0, batchSize):
					#forward propagation
					Results = self.forwardProp(Xbatch[0:,i:i+1])

					#calculating error
					cost += float(np.sum(((Results - Ybatch[0:,i:i+1])**2), axis = 0))

					#backpropagation of the error
					#output layer
					deltaOut = -2*(Results - Ybatch[0:,i:i+1])*self.sigmoidFunc(Results, deriv = True)
					#column matrix with delta value for each unit from output layer

					#first layer
					deltaH1 = np.reshape((np.sum(self.Wout*deltaOut, axis = 0).T), (self.hid1Length, 1))*(self.sigmoidFunc(self.a1, deriv = True))
					#column matrix with delta value for each hidden unit from the first hidden layer


					#getting gradients sum
					#output layer
					WoutGrad += np.dot(deltaOut, self.a1.T)
					BoutGrad += deltaOut

					#first hidden layer
					W1Grad += np.dot(deltaH1, Xbatch[0:,i:i+1].T)
					B1Grad += deltaH1

					qtCases += 1

				#updating parameters after each processed mini-batch
				self.Wout += 1*learningRate*WoutGrad/batchSize
				self.Bout += 1*learningRate*BoutGrad/batchSize
				self.W1 += 1*learningRate*W1Grad/batchSize
				self.B1 += 1*learningRate*B1Grad/batchSize
				
				#average cost of the epoch processed
				errorList.append((cost/qtCases))
				#print(error/qtCases)

		#plot average cost per mini batch
		plt.figure(1)
		plt.plot(range(0,epochs*int(len(X[0])/batchSize)), errorList, 'm')
		#plt.plot(range(0,epochs), errorList, 'm')
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
mlp.backProp(Xtrain, Ytrain, [0.01, 10000, 10])	#learningRate, amount of epochs, mini-batch size


#testing neural network
Ytest = mlp.forwardProp(Xtrain)


#writing results to csv file
pd.DataFrame(Ytest.T).to_csv("files/testResults.csv", header=None, index=None)


plt.show()
