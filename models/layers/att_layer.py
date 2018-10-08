import torch.nn as nn
import torch.nn.functional as F

from .common_layer import FeedForwardNetwork


class SeqAttnMatch(nn.Module):
	"""Given sequences X and Y, match sequence Y to each element in X.
	* o_i = sum(alpha_j * y_j) for i in X
	* alpha_j = softmax(y_j * x_i)
	"""

	def __init__(self, input_size, identity=False):
		super(SeqAttnMatch, self).__init__()
		if not identity:
			self.linear = nn.Linear(input_size, input_size)
		else:
			self.linear = None

	def forward(self, x, y, y_mask):
		"""
		Args:
			x: batch * len1 * hdim
			y: batch * len2 * hdim
			y_mask: batch * len2 (1 for padding, 0 for true)
		Output:
			matched_seq: batch * len1 * hdim
		"""
		# Project vectors
		if self.linear:
			x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
			x_proj = F.relu(x_proj)
			y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
			y_proj = F.relu(y_proj)
		else:
			x_proj = x
			y_proj = y

		# Compute scores
		scores = x_proj.bmm(y_proj.transpose(2, 1))

		# Mask padding
		y_mask = y_mask.unsqueeze(1).expand(scores.size())
		scores.data.masked_fill_(y_mask.data, -float('inf'))

		# Normalize with softmax
		alpha = F.softmax(scores, dim=2)

		# Take weighted average
		matched_seq = alpha.bmm(y)
		return matched_seq


class SelfAttnMatch(nn.Module):
	"""Given sequences X and Y, match sequence Y to each element in X.
	* o_i = sum(alpha_j * x_j) for i in X
	* alpha_j = softmax(x_j * x_i)
	"""

	def __init__(self, input_size, identity=False, diag=True):
		super(SelfAttnMatch, self).__init__()
		if not identity:
			self.linear = nn.Linear(input_size, input_size)
		else:
			self.linear = None
		self.diag = diag

	def forward(self, x, x_mask):
		"""
		Args:
			x: batch * len1 * dim1
			x_mask: batch * len1 (1 for padding, 0 for true)
		Output:
			matched_seq: batch * len1 * dim1
		"""
		# Project vectors
		if self.linear:
			x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
			x_proj = F.relu(x_proj)
		else:
			x_proj = x

		# Compute scores
		scores = x_proj.bmm(x_proj.transpose(2, 1))
		if not self.diag:
			x_len = x.size(1)
			for i in range(x_len):
				scores[:, i, i] = 0

		# Mask padding
		x_mask = x_mask.unsqueeze(1).expand(scores.size())
		scores.data.masked_fill_(x_mask.data, -float('inf'))

		# Normalize with softmax
		alpha = F.softmax(scores, dim=2)

		# Take weighted average
		matched_seq = alpha.bmm(x)
		return matched_seq


class BilinearSeqAttn(nn.Module):
	"""A bilinear attention layer over a sequence X w.r.t y:
	* o_i = softmax(x_i'Wy) for x_i in X.
	Optionally don't normalize output weights.
	"""

	def __init__(self, x_size, y_size, identity=False, normalize=True):
		super(BilinearSeqAttn, self).__init__()
		self.normalize = normalize

		# If identity is true, we just use a dot product without transformation.
		if not identity:
			self.linear = nn.Linear(y_size, x_size)
		else:
			self.linear = None

	def forward(self, x, y, x_mask):
		"""
		Args:
			x: batch * len * hdim1
			y: batch * hdim2
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			alpha = batch * len
		"""
		Wy = self.linear(y) if self.linear is not None else y
		xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
		xWy.data.masked_fill_(x_mask.data, -float('inf'))
		if self.normalize:
			if self.training:
				# In training we output log-softmax for NLL
				alpha = F.log_softmax(xWy, -1)
			else:
				# ...Otherwise 0-1 probabilities
				alpha = F.softmax(xWy, -1)
		else:
			alpha = xWy.exp()
		return alpha


class LinearSeqAttn(nn.Module):
	"""Self attention over a sequence:
	* o_i = softmax(Wx_i) for x_i in X.
	"""

	def __init__(self, input_size):
		super(LinearSeqAttn, self).__init__()
		self.linear = nn.Linear(input_size, 1)

	def forward(self, x, x_mask):
		"""
		Args:
			x: batch * len * hdim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			alpha: batch * len
		"""
		x_flat = x.view(-1, x.size(-1))
		scores = self.linear(x_flat).view(x.size(0), x.size(1))
		scores.data.masked_fill_(x_mask.data, -float('inf'))
		alpha = F.softmax(scores, -1)
		return alpha


class NonLinearSeqAttn(nn.Module):
	"""Self attention over a sequence:
	* o_i = softmax(function(Wx_i)) for x_i in X.
	"""

	def __init__(self, input_size, hidden_size):
		super(NonLinearSeqAttn, self).__init__()
		self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

	def forward(self, x, x_mask):
		"""
		Args:
			x: batch * len * dim
			x_mask: batch * len (1 for padding, 0 for true)
		Output:
			alpha: batch * len
		"""
		scores = self.FFN(x).squeeze(2)
		scores.data.masked_fill_(x_mask.data, -float('inf'))
		alpha = F.softmax(scores, -1)

		return alpha
