import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
	"""
	A three-layer convolutional network with the following architecture:
	
	conv - relu - 2x2 max pool - affine - relu - affine - softmax
	
	The network operates on minibatches of data that have shape (N, C, H, W)
	consisting of N images, each with height H and width W and with C input
	channels.
	"""
	
	def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
							 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
							 dtype=np.float32):
		"""
		Initialize a new network.
		
		Inputs:
		- input_dim: Tuple (C, H, W) giving size of input data
		- num_filters: Number of filters to use in the convolutional layer
		- filter_size: Size of filters to use in the convolutional layer
		- hidden_dim: Number of units to use in the fully-connected hidden layer
		- num_classes: Number of scores to produce from the final affine layer.
		- weight_scale: Scalar giving standard deviation for random initialization
			of weights.
		- reg: Scalar giving L2 regularization strength
		- dtype: numpy datatype to use for computation.
		"""
		self.params = {}
		self.reg = reg
		self.dtype = dtype
		
		############################################################################
		# TODO: Initialize weights and biases for the three-layer convolutional		 #
		# network. Weights should be initialized from a Gaussian with standard		 #
		# deviation equal to weight_scale; biases should be initialized to zero.	 #
		# All weights and biases should be stored in the dictionary self.params.	 #
		# Store weights and biases for the convolutional layer using the keys 'W1' #
		# and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the			 #
		# hidden affine layer, and keys 'W3' and 'b3' for the weights and biases	 #
		# of the output affine layer.																							 #
		############################################################################
		C, H, W = input_dim
		self.params['W1'] = np.random.normal(0, scale=weight_scale, size=(num_filters, C, filter_size, filter_size))
		self.params['W2'] = np.random.normal(0, scale=weight_scale, size=(int(num_filters*H*W*0.25), hidden_dim))
		self.params['W3'] = np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes))
		
		self.params['b1'] = np.zeros(num_filters)
		self.params['b2'] = np.zeros(hidden_dim)
		self.params['b3'] = np.zeros(num_classes) 
		############################################################################
		#															END OF YOUR CODE														 #
		############################################################################
		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)
		 
 
	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for the three-layer convolutional network.
		
		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']
		
		# pass conv_param to the forward pass for the convolutional layer
		filter_size = W1.shape[2]
		conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

		scores = None
		############################################################################
		# TODO: Implement the forward pass for the three-layer convolutional net,  #
		# computing the class scores for X and storing them in the scores					 #
		# variable.																																 #
		############################################################################
		z1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
		z2, cache2 = affine_relu_forward(z1, W2, b2)
		y3, cache3 = affine_forward(z2, W3, b3)
		scores = y3
		############################################################################
		#															END OF YOUR CODE														 #
		############################################################################
		
		if y is None:
			return scores
		
		loss, grads = 0, {}
		############################################################################
		# TODO: Implement the backward pass for the three-layer convolutional net, #
		# storing the loss and gradients in the loss and grads variables. Compute  #
		# data loss using softmax, and make sure that grads[k] holds the gradients #
		# for self.params[k]. Don't forget to add L2 regularization!							 #
		############################################################################
		loss, dout = softmax_loss(scores, y)
		loss += self.reg * 0.5 * (np.power(self.params['W3'], 2).sum() + 
																np.power(self.params['W2'], 2).sum() + 
																np.power(self.params['W1'], 2).sum())

		dx3, grads['W3'], grads['b3'] = affine_backward(dout, cache3)
		dx2, grads['W2'], grads['b2'] = affine_relu_backward(dx3, cache2)
		dx1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx2, cache1)
		############################################################################
		#															END OF YOUR CODE														 #
		############################################################################
		
		return loss, grads

def affine_relu_bn_forward(x, w, b, gamma, beta, bn_param):
	a, fc_cache = affine_forward(x, w, b)
	a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
	out, relu_cache = relu_forward(a)
 
	cache = (fc_cache, bn_cache, relu_cache)
	return out, cache

def affine_relu_bn_backward(dout, cache):
	fc_cache, bn_cache, relu_cache = cache

	da = relu_backward(dout, relu_cache)
	da, dgamma, dbeta = batchnorm_backward_alt(da, bn_cache)
	dx, dw, db = affine_backward(da, fc_cache)
 
	return dx, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(x, w, b, dropout_param):
	a, fc_cache = affine_forward(x, w, b)
	a, relu_cache = relu_forward(a)
	out, dropout_cache = dropout_forward(a, dropout_param)

	cache = (fc_cache, relu_cache, dropout_cache)
	return out, cache

def affine_relu_dropout_backward(dout, cache):
	fc_cache, relu_cache, dropout_cache = cache

	da = dropout_backward(dout, dropout_cache)
	da = relu_backward(da, relu_cache)
	dx, dw, db = affine_backward(da, fc_cache)

	return dx, dw, db

class AlexandrosNet(object):
	""" {conv-relu -> conv-relu-pool -> spatial_batchnorm -> 
				affine-relu-batchnorm -> dropout -> affine -> softmax} """
	def __init__(self, input_dim=(3, 32, 32), num_filters=[32, 96], filter_size=[5, 3],
								hidden_dim=[500, 100], num_classes=10, weight_scale=1e-2, reg=0.0,
								dtype=np.float32, use_batchnorm=True, dropout=0):

		self.use_batchnorm = use_batchnorm
		self.use_dropout = dropout > 0
		self.num_layers = 5
	
		self.params = {}
		self.reg = reg
		self.dtype = dtype

		C, H, W = input_dim

		# Convolutional layers
		self.params['W1'] = np.random.normal(0, scale=weight_scale, size=(num_filters[0], C, 
																																			filter_size[0], filter_size[0]))
		self.params['W2'] = np.random.normal(0, scale=weight_scale, size=(num_filters[1], num_filters[0],
																																				filter_size[1], filter_size[1]))
		# Fully connected layers	
		self.params['W3'] = np.random.normal(0, scale=weight_scale, size=(int(num_filters[1]*H*W*0.25), hidden_dim[0]))
		
		# Output layer
		self.params['W5'] = np.random.normal(0, scale=weight_scale, size=(hidden_dim[0], num_classes))
	
		# Biases	
		self.params['b1'] = np.zeros(num_filters[0])
		self.params['b2'] = np.zeros(num_filters[1])
		self.params['b3'] = np.zeros(hidden_dim[0])
		self.params['b5'] = np.zeros(num_classes)

		# Parameters
		self.params['gamma1'] = np.ones(num_filters[1])
		self.params['beta1'] = np.zeros(num_filters[1])
		self.params["gamma2"] = np.ones(hidden_dim[0])
		self.params["beta2"] = np.zeros(hidden_dim[0])
		self.params["gamma3"] = np.ones(num_classes)
		self.params["beta3"] = np.zeros(num_classes)

		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_params = {'mode': 'train', 'p': dropout}

		self.bn_params = []
		if self.use_batchnorm:
			self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]

		for k, v in self.params.iteritems():
			self.params[k] = v.astype(dtype)


	def loss(self, X, y=None):
		"""
		Evaluate loss and gradient for convolutional network.
		
		Input / output: Same API as TwoLayerNet in fc_net.py.
		"""
		mode = 'test' if y is None else 'train'
		if self.dropout_param is not None:
			self.dropout_param['mode'] = mode
		if self.use_batchnorm:
			for bn_param in self.bn_params:
				bn_param[mode] = mode

		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		W3, b3 = self.params['W3'], self.params['b3']
		W5, b5 = self.params['W5'], self.params['b5']
		
		gamma1, beta1 = self.params['gamma1'], self.params['beta1']
		gamma2, beta2 = self.params['gamma2'], self.params['beta2']
		gamma3, beta3 = self.params['gamma3'], self.params['beta3']	

		# pass conv_param to the forward pass for the convolutional layer
		filter_size1 = W1.shape[2]
		conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}
		filter_size2 = W2.shape[2]
		conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
		
		# pass pool_param to the forward pass for the max-pooling layer
		pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
		
		scores = None
	
		# Convolutional layers	
		z1, cache1 = conv_relu_forward(X, W1, b1, conv_param1)
		z2, cache2 = conv_relu_pool_forward(z1, W2, b2, conv_param2, pool_param)
		z3, cache3 = spatial_batchnorm_forward(z2, gamma1, beta1, self.bn_params[1])

		# Fully Connected layers
		z4, cache4 = affine_relu_bn_forward(z3, W3, b3, gamma2, beta2, self.bn_params[2])
		z4, cache9 = dropout_forward(z4, self.dropout_params)

		# Output layer
		z6, cache6 = affine_forward(z4, W5, b5)
		z7, cache7 = batchnorm_forward(z6, gamma3, beta3, self.bn_params[3])
		#z8, cache8 = dropout_forward(z7, self.dropout_params)
		scores = z7
		
		if y is None:
			return scores
		
		loss, grads = 0, {}
		loss, dout = softmax_loss(scores, y)
		loss += self.reg * 0.5 * (np.power(self.params['W1'], 2).sum() +
																np.power(self.params['W2'], 2).sum() +
																np.power(self.params['W5'], 2).sum() +
																np.power(self.params['W3'], 2).sum())
		
		#dx8 = dropout_backward(dout, cache8)
		dx7, grads['gamma3'], grads['beta3'] = batchnorm_backward(dout, cache7)
		dx6, grads['W5'], grads['b5'] = affine_backward(dx7, cache6)
		dx6 = dropout_backward(dx6, cache9)
		dx4, grads['W3'], grads['b3'], grads['gamma2'], grads['beta2'] = affine_relu_bn_backward(dx6, cache4)
		
		dx3, grads['gamma1'], grads['beta1'] = spatial_batchnorm_backward(dx4, cache3)
		dx2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dx3, cache2)
		dx1, grads['W1'], grads['b1'] = conv_relu_backward(dx2, cache1)
		
		return loss, grads
