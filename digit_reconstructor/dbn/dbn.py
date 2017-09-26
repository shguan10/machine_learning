import sys
sys.path.insert(0,'..\\rbm')
import numpy as np
import os
import rbm0_m as rbm0

rootdir = '.\\'
if(os.getcwd()[-4:-1]=="rbm"): rootdir = '..\\..\\'

#@params im is a binary Image object
#returns the 784x1 training vector, without a bias unit
def getV(im):
	return np.reshape(list(im.getdata()), (784, 1))/255

class dbn:
	#assume sizes doesn't include the bias unit
	#takes in an rbmList with the number of final hidden units, along with a sparsity target
	#assumes all the rbm.data is the same
	#assumes sizes of the rbmList are consecutively the same (i.e. this could be a legit dbn)
	def __init__(self,rbmList,no_final_h,mrat=.5,eta=.001, p=.01):
		sizes = [rbmList[0].size[0]]
		sizes.extend([
			rbmList[k].size[1] for k in xrange(len(rbmList))
					  ])
		sizes.extend([no_final_h])
		self.rbmList = [rbmList[k] for k in xrange(len(rbmList))]

		self.rbmList.append(rbm0.rbm0((sizes[-2],sizes[-1])))
		self.rbmList[-1].data=rbmList[0].data

		self.sizes = sizes
		#initialize weights
		#initalize network

		self.data = rbmList[0].data
		self.num_layers=len(rbmList)

		self.default_weight_initializer()
		self.reconError=0 #sum of squared error terms, averaged over a mini_batch
			#np.empty((785,0))#this is the distance between the trainV and the reconV, summed over a mini_batch
		self.p=p # the desired sparsity of the hidden activations
		self.q=0
		self.mrat=mrat
		self.eta=eta
		#self.recon_vs = np.empty((785,0))#debuginng, delete when unneeded

	@staticmethod
	def fromDBN(dbn0,no_final_h,p=.01):
		return dbn(dbn0.rbmList,no_final_h,p)

	#self.data is a matrix where the columns are training vectors, and the last row are the bias units
	def train(self,epochs,monitor=True,qpersist=.9,sc=.02):
		for j in xrange(epochs):
			np.random.shuffle(self.data.T)#shuffle the order of the training vectors, not their content!
			mini_batches = [
				self.data[:,k:k+10]
				for k in xrange(0, len(self.data[0]), 10)]
			vel=self.eta
			for mini_batch in mini_batches:
				#convert mini_batch of images to array
				vel = vel*self.mrat + self.CD1(mini_batch,qpersist,sc) / len(mini_batch[0]) * self.eta
				self.rbmList[-1].weights += vel

			if monitor:
				print "Epoch {0} Reconstruction Error: {1}".format(
					j,self.reconError/10)
			else:
				print "Epoch {0} complete".format(j)
			#mD.display(self.recon_vs[:-1]*255) #debugging

	# starting from image input, returns the second-to-last hidden layer, as a column vector, to be used as input for the
	# last rbm
	def forwardpassforCD(self, train_vs):
		for rbm in self.rbmList[:-1]:
			hiddenprobs = sigmoid(np.dot(rbm.weights, train_vs))
			hiddenprobs[-1] = 1  # fix the 1s for the hidden biases
			train_vs = hiddenprobs > np.random.rand(np.shape(hiddenprobs)[0],np.shape(hiddenprobs)[1])
		return train_vs

	# starting from matrix of columns hidden_vs, ends with matrix of columns image_vs
	def backwardpass(self, hidden_vs):#TODO not tested, should work
		for rbm in self.rbmList[::-1]:
			hidden_vs = sigmoid(np.dot(rbm.weights.T, hidden_vs))
			hidden_vs[-1]=1
		return hidden_vs

	#@param mini_batch is a matrix where each column is a training vector, the last row is for the bias
	#assumes the weights is (h+1)*(v+1) big, where h is len(self.hidden) and v is len(self.visible)
		#the last column is weights are the bias weights
	#returns list of weight changes with respect to one training vector
	def CD1(self,train_vs,qpersist,sc):
		#set the visible units to be the training vector
		#visible = np.reshape(np.append(train_vs,1),(785,1))
		#print "mini_batch avg: ",np.mean(train_vs)
		input = self.forwardpassforCD(train_vs)
		#update hidden
		hiddenprobs = sigmoid(np.dot(self.rbmList[-1].weights,input))
		hiddenprobs[-1]=1 #fix the 1s for the hidden biases
		hidden =  hiddenprobs > np.random.rand(self.rbmList[-1].size[1]+ 1,len(input[0]))

		self.q=self.q*qpersist+(1-qpersist)*np.mean(hidden)
		sparsitychg= -sc*np.dot((self.q-self.p)+np.zeros((self.rbmList[-1].size[1]+1,len(input[0]))),input.T)#TODO finish statement. do you need both pos and neg sparsity?

		#collect v_i*h_j
		pos = np.dot(hiddenprobs,input.T)

		#reconstruct visible, but only use probabilities
		reconVprob=sigmoid(np.dot(self.rbmList[-1].weights.T,hidden))
		reconVprob[-1]=1 #fix for the bias

		#print "mini_batch recon avg: ",np.mean(reconVprob)," \n\n"

		#update hidden units again, only probabilities
		hiddenprobs=sigmoid(np.dot(self.rbmList[-1].weights,reconVprob))
		hiddenprobs[-1] = 1  # fix for the bias

		#collect p(v_i)*p(h_i)
		neg=np.dot(hiddenprobs,reconVprob.T)

		a= (reconVprob-input)*(reconVprob-input)
		self.reconError = np.sum(a)

		#self.recon_vs=np.append(self.recon_vs,reconVprob,axis=1)#debugging delete when done

		#return weight changes
		return pos-neg+sparsitychg

	#self.sizes is assumed to be only 2 elements
	#self.sizes is assumed to not count the bias units
	def default_weight_initializer(self):#TODO remove because redundant
		self.weights = [np.random.randn(y+1, x+1)
						for x, y in zip(self.sizes[:-1], self.sizes[1:])]
		#self.weights = np.random.randn(self.sizes[1]+1, self.sizes[0]+1) / np.sqrt(self.sizes[1]+self.sizes[0])

	def displayWeights(self):
		import matrix784xDisplayer as mD
		a = self.backwardpass(np.identity(len(self.rbmList[-1].weights)))
		mD.display(a[:-1]*254)
		
	

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))
