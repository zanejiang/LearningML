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
	def h_theta(self):
		"""The hypothesis function"""
		return
		
	
	def plot_loss_trend(self):
		fig =plt.figure(1)
		ax = plt.subplot(111)
		ax.plot(self.J)
		fig.show()
	
	def fit(self,num_of_iteration=100, eps=0.001):
		"""Calculate theta by gradient descent"""
		self.J=[0]
		m = self.Y.size
		for i in range(num_of_iteration):
			loss = self.Y-self.h_theta()
			gradient = np.dot(loss.T,self.X)/m
			self.theta = self.theta+self.alpha*gradient
			#print self.J_theta()
			self.J.append(self.J_theta())
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
		
	def J_theta(self):
		m = self.Y.size
		err = self.h_theta()-self.Y
		err = np.dot(err.T,err)
		return 0.5*err/m
		
	def h_theta(self):
		#print self.X
		return np.dot(self.X,self.theta)
		
		
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
	
	def h_theta(self):
		z=LMS.h_theta(self)
		#print z.shape
		return 1.0/(1.0+np.exp(-1.0*z))
		
	def J_theta(self):
		h=self.h_theta()
		Y_=np.ones_like(self.Y)-self.Y
		h_=np.ones_like(self.Y)-self.h_theta()
		logL=np.dot(self.Y.T,np.log(h))+np.dot(Y_.T,np.log(h_))
		return logL
