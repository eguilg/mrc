"""Implementation of the SLQA."""
import torch.nn as nn

from ..layers.att_layer import NonLinearCoSeqAttn, SelfAttnMatch
from ..layers.common_layer import VFU
from ..layers.rnn_layer import BiRNN


class SLQAP(nn.Module):

	def __init__(self, input_size, hidden_size, dropout, rnn_type=nn.LSTM, hop=2):

		super(SLQAP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.rnn_type = rnn_type
		self.hop = hop

		# Encoder
		self.encoding_rnn = BiRNN(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		doc_hidden_size = 2 * self.hidden_size

		self.res_VFUs = nn.ModuleList()
		self.self_attn_matchs_a = nn.ModuleList()

		self.self_VFUs_a = nn.ModuleList()
		self.co_attn_layers = nn.ModuleList()
		self.co_VFUs = nn.ModuleList()
		self.self_attn_matchs_b = nn.ModuleList()
		self.self_VFUs_b = nn.ModuleList()
		self.aggregate_rnns = nn.ModuleList()
		self.layer_dropout = nn.Dropout(p=self.dropout / self.hop)

		for i in range(self.hop):
			# residual fusion
			if i > 0:
				self.res_VFUs.append(VFU(doc_hidden_size))

			#  self-attention before co-attention
			self.self_attn_matchs_a.append(SelfAttnMatch(doc_hidden_size))
			self.self_VFUs_a.append(VFU(doc_hidden_size))

			#  co-attention
			self.co_attn_layers.append(NonLinearCoSeqAttn(doc_hidden_size, self.hidden_size))
			self.co_VFUs.append(VFU(doc_hidden_size))

			#  self-attention
			self.self_attn_matchs_b.append(SelfAttnMatch(doc_hidden_size))
			self.self_VFUs_b.append(VFU(doc_hidden_size))

			#  aggregation
			self.aggregate_rnns.append(BiRNN(
				input_size=2 * self.hidden_size,
				hidden_size=self.hidden_size,
				num_layers=1,
				dropout=self.dropout,
				rnn_type=self.rnn_type,
			))

		self.out1_dim = 2 * hidden_size
		self.out2_dim = 2 * hidden_size

	def forward(self, c, c_mask, q, q_mask):

		# Encode document with RNN
		c = self.encoding_rnn(c, c_mask)

		# Encode question with RNN
		q = self.encoding_rnn(q, q_mask)

		# Align and aggregate
		c_i, q_i = c, q
		for i in range(self.hop):
			#  residuals from encoders
			if i > 0:
				c_i = self.res_VFUs[i - 1](c, c_i)
				q_i = self.res_VFUs[i - 1](q, q_i)

			#  self-attention before co-attention
			c_h = self.self_attn_matchs_a[i](c_i, c_mask)
			c_i = self.self_VFUs_a[i](c_i, c_h)

			q_h = self.self_attn_matchs_a[i](q_i, q_mask)
			q_i = self.self_VFUs_a[i](q_i, q_h)

			#  co-attention
			q_h, c_h = self.co_attn_layers[i](c_i, q_i, c_mask, q_mask)
			c_i = self.co_VFUs[i](c_i, q_h)
			q_i = self.co_VFUs[i](q_i, c_h)

			#  dropout
			c_i = self.layer_dropout(c_i)
			q_i = self.layer_dropout(q_i)

			#  self-attention after co attention
			c_h = self.self_attn_matchs_b[i](c_i, c_mask)
			c_i = self.self_VFUs_b[i](c_i, c_h)

			q_h = self.self_attn_matchs_b[i](q_i, q_mask)
			q_i = self.self_VFUs_b[i](q_i, q_h)

			#  aggregation
			c_i = self.aggregate_rnns[i](c_i, c_mask)
			q_i = self.aggregate_rnns[i](q_i, q_mask)

		return c_i, q_i
