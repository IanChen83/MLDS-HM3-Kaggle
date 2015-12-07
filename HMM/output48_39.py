import theano
import theano.tensor as T
import numpy as np
__author__= 'jason'

class MAP:
	"""docstring for ClassName"""
	def __init__(self):
		self.map_data = open('48_39.map','r')
		self.in_48 = []
		self.in_39 = []	
		for line in self.map_data:
			in_x = line.split('\t')
			self.in_48.append(in_x[0])
			self.in_39.append(in_x[1])	

	def map(self,index):
		#y = z.tolist()
		#big = max(y)
		#big_index = y.index(big)
		#print big_index
		#if index_in==0:
		#	return self.in_48[big_index]
		#else :
		return self.in_39[index].split('\n')[0]
#m = MAP()
#arr = np.array([0,1,2,3])
#print m.map(arr)
