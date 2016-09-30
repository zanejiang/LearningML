import numpy as np
from matplotlib import pyplot  as plt
import abc

class PLA(object):
	"""
	The freqme of Perceptron Learning Algorithm
	"""
	error_counter = 0
	def __init__(self,X,Y):
		self.Y = Y
		(m,n) = X.shape
		self.X=np.hstack((np.ones(m).reshape(m,1),X))
		self.W = np.zeros(n+1)
		return
		
	def hyst(self):
		return np.sign(np.dot(self.X,self.W))	
		
	def choise(self,x,y,size=1):
		np.random.shuffle(x)
		(m,n)=x.shape
		#print m
		if (m!=0):
			B = np.random.randint(m,size=size)
			return x[B,:],y[B]
		else:
			return np.zeros((1,n)),np.zeros(1)	
		
	def __iter__(self):
		y=self.hyst()
		error_pt = y!=self.Y
		self.error_counter = np.sum(error_pt)
		corrX,corrY = self.choise(self.X[error_pt,:],self.Y[error_pt])
		self.W = self.W+corrX.flatten()*corrY.flatten()
		return  (self.error_counter,self.W)	
		
	def fit(self,num_of_iteration=100):
		"""Calculate the W by iteration"""
		for i in range(num_of_iteration):
			(latest_err_counter, w) =self.__iter__()
			#print latest_err_counter,
			#print w
			if(abs(latest_err_counter)<1):
				#print "!Convengence Gotten"
				break
		return i		
			
			
			
				