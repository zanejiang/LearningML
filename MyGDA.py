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
	
	def hot1(self,X,number_of_class):
		m=1
		if type(X) is np.ndarray:
			m = len(X)

		_hot1=np.zeros(shape=(m,number_of_class),dtype=np.float)
		_hot1[np.arange(m),X]=1.0
		return _hot1

	def shuffle(self,X,Y):
		p=np.random.permutation(len(Y))
		return X[p,:],Y[p]

	def indicator(self,X,k):
		return (X==k).astype(np.float)

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
				gradient = self.gradient(Xt,Yt,algorithm='Stochatic')
				self.theta = self.theta+alpha*gradient

			#self.J.append(self.J_theta(X,Y))
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
			gradient = self.gradient(self.X,self.Y)
			#print theta,gradient
			self.theta = self.theta+alpha*gradient
			#self.J.append(self.J_theta(self.X,self.Y))

			if np.linalg.norm(self.theta-theta)<eps:
				break

		return actual_iteration

	def gradient(self,X,Y,algorithm='batch'):
		if(algorithm=='batch'):
			loss = Y - self.h_theta(X)
			gradient = np.dot(X.T, loss) / (len(Y))
		else:
			loss = Y - self.h_theta(X)
			gradient = loss*X
		return gradient


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
	Num_of_Classes=0
	def J_theta(selfself,X,Y):
		pass

	def score(self,X,y):
		ypredicated = self.predicate(X)
		return np.sum(ypredicated == y) * 1.0 / len(y)

	def predicate(self,X):
		(m, n) = X.shape
		X_ = np.hstack((np.ones(m).reshape((m, 1)), X))
		pik=self.h_theta(X_)
		#pik is (m,k)
		return np.argmax(pik,axis=1)

	def gradient(self,X,Y,algorithm='batch'):
		if(algorithm=='Stochatic'):
			X=X[:,np.newaxis].T
		Y_hot1= self.hot1(Y,number_of_class=self.Num_of_Classes)
		#Y_hot1 is matrix of (m,k)
		pik=Y_hot1-self.h_theta(X)
		#pik is matrix of (m,k) representing the progability of each x sample classified to class 1,....K
		#x is matrix of (m,n+1)
		gradient=[]
		for pk in pik.T:
			#pk is a vector of k components
			gradient.append((X*pk[:,np.newaxis]).mean(axis=0).tolist())
		#print gradient
		#the gradient will be (n+1,k)
		return np.array(gradient).T


	def h_theta(self,X):
		#X is the matrix of (m,n+1)
		s = np.dot(X,self.theta)
		#now s is (m,k)
		exps= np.exp(s)
		return exps/(np.sum(exps,axis=1)[:,np.newaxis])

	def __init__(self,X,Y):
		(m, n) = X.shape
		X_ = np.hstack((np.ones(m).reshape((m, 1)), X))
		#number of classes
		self.Num_of_Classes = len(np.unique(Y))
		#here our theta is a matrix of (n+1,K)
		theta=np.zeros((n+1,self.Num_of_Classes),dtype=np.float)
		print theta.shape,X_.shape
		GDAlgorithm.__init__(self,X_, Y, theta)
