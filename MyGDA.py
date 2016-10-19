import numpy as np
from matplotlib import pyplot  as plt
import abc
import math

class GDAlgorithm(object):
	"""
	The frame of gradient descent algorithm for linear model
	"""
	def __init__(self,alpha,X,Y,theta):
		self.alpha = alpha
		self.X = X
		self.Y = Y
		self.theta = theta
		self.J = [0.0]
		
	def __iter__(self):
		return self
		
	def __next__(self):
		return (self.Widrow_Hoff(),self.self.J_theta())
		
				
	@abc.abstractmethod
	def J_theta(self):
		"""The cost function """
		return
		
	@abc.abstractmethod
	def h_theta(self,X):
		"""The hypothesis function"""
		return
		
	
	def plot_loss_trend(self):
		fig =plt.figure(1)
		ax = plt.subplot(111)
		ax.plot(self.J)
		fig.show()
	
	def shuffle(self,X,Y):
		p=np.random.permutation(len(Y))
		return X[p,:],Y[p]


	def fit(self,num_of_iteration=100, eps=0.001, alpha=1, algorithm='batch',batch_size=-1):
		"""Calculate theta by gradient descent"""
		self.J=[0]
		if algorithm=='batch':
			batch_size=len(self.Y)

		for i in range(num_of_iteration):
			Y=self.Y
			X=self.X
			if algorithm=='Stochatic':
				X,Y = self.shuffle(self.X,self.Y)

			for j in np.arange(0,len(Y),batch_size):
				#print j,
				Xt= X[j:j+batch_size,:]
				Yt= Y[j:j+batch_size]
				#print Xt
				loss = Yt-self.h_theta(Xt)
				gradient = np.dot(Xt.T,loss)/(len(Yt))
				self.theta = self.theta+alpha*gradient
				#print self.theta
			self.J.append(self.J_theta(X,Y))

			if abs(self.J[-1]-self.J[-2])<eps:
				print "!Convengence gotten"
				break
		return
			
class LMS(GDAlgorithm):
	def __init__(self,alpha,X,Y):
		(m,n)=X.shape
		X_=np.hstack((np.ones(m).reshape((m,1)),X))
		#print X_.shape
		theta = np.random.rand(n+1)
		#theta = np.zeros(n+1)
		GDAlgorithm.__init__(self,alpha,X_,Y,theta)
		
	def J_theta(self,X,Y):
		m = Y.size
		err = self.h_theta(X)-Y
		err = np.dot(err.T,err)
		return 0.5*err/m
		
	def h_theta(self,X):
		return np.dot(X,self.theta)
		
		
class LogisticRegression_Binary(LMS):
	def score(self,X,y,threshold=0.5):
		ypredicated = self.predicate(X,threshold)
		return np.sum(ypredicated == y)*1.0/len(y)
	
	def predicate(self,X, threshold=0.5):
		(m,n)=X.shape
		X_=np.hstack((np.ones(m).reshape((m,1)),X))
		z=np.dot(X_,self.theta)
		P= 1.0/(1.0+np.exp(-1.0*z))
		return (P>threshold)
	
	def h_theta(self,X):
		z=np.dot(X,self.theta)
		return 1.0/(1.0+np.exp(-1.0*z))
		
	def J_theta(self,X,Y):
		#print self.theta
		h=self.h_theta(X)
		Y_=np.ones_like(Y)-Y
		h_=np.ones_like(Y)-self.h_theta(X)
		logL=np.dot(Y.T,np.log(h))+np.dot(Y_.T,np.log(h_))
		#print logL
		return logL
