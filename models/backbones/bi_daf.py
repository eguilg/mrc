"""Implementation of the BiDAF."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.att_layer import SelfAttnMatch
from ..layers.common_layer import FeedForwardNetwork, VFU
from ..layers.rnn_layer import BiRNN


class BiDAF(nn.Module):

	def __init__(self, input_size, hidden_size, dropout, rnn_type=nn.LSTM):
		super(BiDAF, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.rnn_type = rnn_type

		# Encoder
		self.encoding_rnn = BiRNN(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		self.FFN = FeedForwardNetwork(
			input_size=2 * self.hidden_size,
			hidden_size=self.hidden_size,
			output_size=1,
			dropout_rate=self.dropout
		)

		self.self_att_a = SelfAttnMatch(2 * self.hidden_size)
		self.self_VFU_a = VFU(2 * self.hidden_size)

		self.self_att_b = SelfAttnMatch(2 * self.hidden_size)
		self.self_VFU_b = VFU(2 * self.hidden_size)

		self.co_VFU = VFU(2 * self.hidden_size)

		#  modeling layer
		self.modeling_rnn = BiRNN(
			input_size=2 * self.hidden_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		self.out1_dim = 2 * hidden_size
		self.out2_dim = 2 * hidden_size

	def forward(self, c, c_mask, q, q_mask):
		# Encode document with RNN
		c = self.encoding_rnn(c, c_mask)  # (batch, T, 2d)

		# Encode question with RNN
		q = self.encoding_rnn(q, q_mask)  # (batch, J, 2d)

		c_len = c.size(1)
		q_len = q.size(1)

		# self att
		c = self.self_VFU_a(c, self.self_att_a(c, c_mask))
		q = self.self_VFU_a(q, self.self_att_a(q, q_mask))

		# Attention Flow
		s = torch.bmm(F.relu(self.FFN(c)), F.relu(self.FFN(q).transpose(1, 2)))  # (batch, T, J)
		# s = torch.bmm(c, q.transpose(1, 2))

		cc_mask = c_mask.unsqueeze(2).expand(-1, -1, q_len)
		s.masked_fill_(cc_mask, -1e30)

		qq_mask = q_mask.unsqueeze(1).expand(-1, c_len, -1)
		s.masked_fill_(qq_mask, -1e30)

		#  C2Q
		q_hat = torch.bmm(F.softmax(s, -1), q)  # (batch, T, 2d)

		#  Q2C
		c_hat = torch.bmm(F.softmax(s.transpose(1, 2), -1), c)  # (batch, J, 2d)
		del s

		c = self.co_VFU(c, q_hat)  # (batch, T, 2d)
		q = self.co_VFU(q, c_hat)  # (batch, J, 2d)

		# self att
		c = self.self_VFU_b(c, self.self_att_b(c, c_mask))
		q = self.self_VFU_b(q, self.self_att_b(q, q_mask))

		c = self.modeling_rnn(c, c_mask)  # (batch, T, 2d)
		q = self.modeling_rnn(q, q_mask)  # (batch, J, 2d)

		return c, q
