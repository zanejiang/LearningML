import numpy as np
from matplotlib import pyplot  as plt
import abc
import math

class GDAlgorithm(object):
	"""
	The frame of gradient descent algorithm for linear model
	"""
	def __init__(self,X,Y,theta):
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


	def SGD(self, num_of_iteration=100, eps=0.001,alpha=1):
		"""Calculate theta by Stochastic Gradient Descent"""
		self.J=[0]

		actual_iteration=0
		for i in range(num_of_iteration):
			X,Y=self.shuffle(self.X,self.Y)
			theta = self.theta
			actual_iteration=actual_iteration+1

			for j in range(len(Y)):
				Xt=X[j,:].flatten()
				Yt=Y[j]
				loss = Yt-self.h_theta(Xt)
				gradient = loss*Xt
				self.theta = self.theta+alpha*gradient

			self.J.append(self.J_theta(X,Y))
			if(np.linalg.norm(self.theta-theta)<eps):
				break

		return actual_iteration

	def fit(self,num_of_iteration=100, eps=0.001, alpha=1, algorithm='batch'):
		"""Calculate theta by gradient descent"""
		self.J=[0]
		if algorithm=='batch':
			batch_size=len(self.Y)
		else:
			return self.SGD(num_of_iteration,eps,alpha)

		actual_iteration=0
		for i in range(num_of_iteration):
			theta = self.theta
			loss = self.Y-self.h_theta(self.X)
			gradient = np.dot(self.X.T,loss)/(len(self.Y))
			self.theta = self.theta+alpha*gradient
			self.J.append(self.J_theta(self.X,self.Y))

			if np.linalg.norm(self.theta-theta)<eps:
				break

		return actual_iteration
			
class LMS(GDAlgorithm):
	def __init__(self,X,Y):
		(m,n)=X.shape
		X_=np.hstack((np.ones(m).reshape((m,1)),X))
		theta = np.zeros(n+1)
		GDAlgorithm.__init__(self,X=X_,Y=Y,theta=theta)
		
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
		return (P>threshold).astype(np.int8)
	
	def h_theta(self,X):
		z=np.dot(X,self.theta)
		return (1.0/(1.0+np.exp(-1.0*z)))
		
	def J_theta(self,X,Y):
		#print self.theta
		h=self.h_theta(X)
		Y_=np.ones_like(Y)-Y
		h_=np.ones_like(Y)-self.h_theta(X)
		logL=1.0/Y.shape[0]*(np.dot(Y.T,np.log(h))+np.dot(Y_.T,np.log(h_)))
		return logL

class SoftmaxRegression(LMS):
	def h_theta(self,X):
		Z=np.dot(X,self.theta)
		a=1.0/(1.0+np.exp(-1.0*Z))
		return

	def __init__(self,X,Y):
		(m, n) = X.shape
		X_ = np.hstack((np.ones(m).reshape((m, 1)), X))
		#number of classes
		Num_of_Classes = len(set(Y.tolist()))
		theta=np.zeros((n+1,Num_of_Classes),dtype=np.float)
		# print X_.shape
		GDAlgorithm.__init__(self,X_, Y, theta)
