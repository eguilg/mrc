import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FeedForwardNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
		super(FeedForwardNetwork, self).__init__()
		self.dropout_rate = dropout_rate
		self.linear1 = nn.Linear(input_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
		x_proj = self.linear2(x_proj)
		return x_proj


# ------------------------------------------------------------------------------
# Functional Units
# ------------------------------------------------------------------------------

class Gate(nn.Module):
	"""Gate Unit
	g = sigmoid(Wx)
	x = g * x
	"""

	def __init__(self, input_size):
		super(Gate, self).__init__()
		self.linear = nn.Linear(input_size, input_size, bias=False)

	def forward(self, x):
		"""
		Args:
			x: batch * len * dim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			res: batch * len * dim
		"""
		x_proj = self.linear(x)
		gate = F.sigmoid(x)
		return x_proj * gate


class SFU(nn.Module):
	"""Semantic Fusion Unit
	The ouput vector is expected to not only retrieve correlative information from fusion vectors,
	but also retain partly unchange as the input vector
	"""

	def __init__(self, input_size, fusion_size):
		super(SFU, self).__init__()
		self.linear_r = nn.Linear(input_size + fusion_size, input_size)
		self.linear_g = nn.Linear(input_size + fusion_size, input_size)

	def forward(self, x, fusions):
		r_f = torch.cat([x, fusions], 2)
		r = F.tanh(self.linear_r(r_f))
		g = F.sigmoid(self.linear_g(r_f))
		o = g * r + (1 - g) * x
		return o


class VectorBasedFusion(nn.Module):
	"""VectorBasedFusion from SLQA """

	def __init__(self, input_size):
		super(VectorBasedFusion, self).__init__()
		self.input_size = input_size
		self.linear_m = nn.Linear(4 * input_size, input_size)
		self.linear_g = nn.Linear(4 * input_size, 1)

	def forward(self, x, fusion):
		r_f = torch.cat([x, fusion, x * fusion, x - fusion], 2)  # b, T, 4s
		m = F.tanh(self.linear_m(r_f))  # b, T, s
		g = F.sigmoid(self.linear_g(r_f)).expand(-1, -1, self.input_size)  # b, T, s
		o = g * m + (1 - g) * x  # b, T, s
		return o


# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
	"""Return uniform weights over non-masked x (a sequence of vectors).
	Args:
		x: batch * len * hdim
		x_mask: batch * len (1 for padding, 0 for true)
	Output:
		x_avg: batch * hdim
	"""
	alpha = Variable(torch.ones(x.size(0), x.size(1)))
	if x.data.is_cuda:
		alpha = alpha.cuda()
	alpha = alpha * x_mask.eq(0).float()
	alpha = alpha / alpha.sum(1).expand(alpha.size())
	return alpha


def weighted_avg(x, weights):
	"""Return a weighted average of x (a sequence of vectors).
	Args:
		x: batch * len * hdim
		weights: batch * len, sum(dim = 1) = 1
	Output:
		x_avg: batch * hdim
	"""

	return weights.unsqueeze(1).bmm(x).squeeze(1)
