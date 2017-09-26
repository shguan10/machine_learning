import random

import numpy as np
from PIL import Image
import os
import cPickle
#import matrix784xDisplayer as mD #debuggins
import gzip

rootdir = '.\\'
if(os.getcwd()[-3:]!="ing"): rootdir = '..\\..\\'#TODO this checks if the cwd is not the root dir (a dir that ends in 'ing'), so I should make this adaptable to more use cases in the future

#@params im is a binary Image object
#returns the 784x1 training vector, without a bias unit
def getV(im):
	return np.reshape(list(im.getdata()), (784, 1))/255

class rbm0:
	#assume sizes doesn't include the bias unit
	def __init__(self, size=(784, 300),mrat=.5,eta=.001, p=.01,dfile=rootdir+"data\\binaryImageMatrix785x.pklb.gz"):
		#initialize weights
		#initalize network
		self.dfile = dfile
		self.data = np.empty((785,0))
		self.num_layers=len(size)
		self.size=size
		self.default_weight_initializer()
		self.parentf = 'C:\\Users\\Xinyu\\Documents\\GitHub\\PNG-JPG_MNIST-format\\binary\\training'
		self.reconError=0 #sum of squared error terms, averaged over a mini_batch
			#np.empty((785,0))#this is the distance between the trainV and the reconV, summed over a mini_batch
		self.p=p # the desired sparsity of the hidden activations
		self.q=0
		self.datasorted=(dfile=="785xImagesSorted.pklb.gz")

		self.mrat=mrat
		self.eta=eta
		#self.recon_vs = np.empty((785,0))#debuginng, delete when unneeded

	#self.data is a binary matrix where each column is a training vector, and the botoom row is the bias units
	#TODO don't assume image files will be binary
	def load_data(self):
		if(self.dfile==None):
			for rootdir, childdirs, files in os.walk(self.parentf):
				for file in files:
					if file.endswith('.png'):
						im = Image.open(os.path.join(rootdir, file))
						self.data = np.append(self.data, np.append(getV(im), [[1]], axis=0), axis=1)
		elif(self.dfile[-2:]=="gz"):
			with gzip.open(self.dfile,"rb") as f:
				self.data = cPickle.load(f)
		else:
			with open(self.dfile,"rb") as f:
				self.data = cPickle.load(f)

	#self.data is a matrix where the columns are training vectors, and the last row are the bias units
	def train(self,epochs,mini_batch_size=10,monitor=True,qpersist=.9,sc=.02):
		for j in xrange(epochs):
			if self.datasorted:
				for ilist in self.data:
					np.random.shuffle(ilist.T)
			else: np.random.shuffle(self.data.T)#shuffle the order of the training vectors, not their content!
			if self.datasorted:
				mini_batches=[]
				for k in xrange(0,50000,mini_batch_size):
					mini_batch = np.zeros((785,0))
					for ilist in self.data:
						mini_batch=np.append(mini_batch,np.reshape(ilist[:,0],(785,1)),axis=1)
					while len(mini_batch.T)<mini_batch_size:
						mini_batch = np.append(mini_batch, np.reshape(self.data[random.randint(0,len(self.data.T))][:,0],(785,1)), axis=1)
					mini_batches.append(mini_batch)
			else:
				mini_batches = [
				self.data[:,k:k+mini_batch_size]
				for k in xrange(0, len(self.data[0]), mini_batch_size)]

			vel = self.eta
			for mini_batch in mini_batches:
				#convert mini_batch of images to array
				vel =vel*self.mrat+ self.CD1(mini_batch,qpersist,sc) / len(mini_batch[0]) * self.eta
				self.weights += vel

			if monitor:
				print "Epoch {0} Reconstruction Error: {1}".format(
					j,self.reconError/mini_batch_size)
			else:
				print "Epoch {0} complete".format(j)
			#mD.display(self.recon_vs[:-1]*255) #debugging

	#@param mini_batch is a matrix where each column is a training vector, the last row is for the bias
	#assumes the weights is (h+1)*(v+1) big, where h is len(self.hidden) and v is len(self.visible)
		#the last column is weights are the bias weights
	#returns list of weight changes with respect to one training vector
	def CD1(self,train_vs,qpersist,sc):
		#set the visible units to be the training vector
		#visible = np.reshape(np.append(train_vs,1),(785,1))
		#print "mini_batch avg: ",np.mean(train_vs)
		#update hidden
		hiddenprobs = sigmoid(np.dot(self.weights,train_vs))
		hiddenprobs[-1]=1 #fix the 1s for the hidden biases
		hidden =  hiddenprobs > np.random.rand(self.size[1] + 1, len(train_vs[0]))

		self.q=self.q*qpersist+(1-qpersist)*np.mean(hidden)
		sparsitychg= -sc*np.dot((self.q-self.p) + np.zeros((self.size[1] + 1, len(train_vs[0]))), train_vs.T)#TODO finish statement. do you need both pos and neg sparsity?

		#collect v_i*h_j
		pos = np.dot(hiddenprobs,train_vs.T)

		#reconstruct visible, but only use probabilities
		reconVprob=sigmoid(np.dot(self.weights.T,hidden))
		reconVprob[-1]=1 #fix for the bias

		#print "mini_batch recon avg: ",np.mean(reconVprob)," \n\n"

		#update hidden units again, only probabilities
		hiddenprobs=sigmoid(np.dot(self.weights,reconVprob))
		hiddenprobs[-1] = 1  # fix for the bias

		#collect p(v_i)*p(h_i)
		neg=np.dot(hiddenprobs,reconVprob.T)

		a= (reconVprob-train_vs)*(reconVprob-train_vs)
		self.reconError = np.sum(a)

		#self.recon_vs=np.append(self.recon_vs,reconVprob,axis=1)#debugging delete when done

		#return weight changes
		return pos-neg+sparsitychg

	#self.sizes is assumed to be only 2 elements
	#self.sizes is assumed to not count the bias units
	def default_weight_initializer(self):
		self.weights = np.random.randn(self.size[1] + 1, self.size[0] + 1) / np.sqrt(self.size[1] + self.size[0])

#### Miscellaneous functions
def sigmoid(z):
	"""The sigmoid function."""
	return 1.0/(1.0+np.exp(-z))
