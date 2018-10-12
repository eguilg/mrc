"""Implementation of the SLQA."""
import torch.nn as nn

from ..layers.att_layer import NonLinearCoSeqAttn, BilinearSelfAttn
from ..layers.common_layer import VFU
from ..layers.rnn_layer import BiRNN


class SLQA(nn.Module):

	def __init__(self, input_size, hidden_size, dropout, rnn_type=nn.LSTM, hop=2):

		super(SLQA, self).__init__()
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

		self.co_attn_layers = nn.ModuleList()
		self.co_VFUs = nn.ModuleList()
		self.self_bi_attns = nn.ModuleList()
		self.self_VFUs = nn.ModuleList()
		self.aggregate_rnns = nn.ModuleList()
		for i in range(self.hop):
			#  co-attention
			self.co_attn_layers.append(NonLinearCoSeqAttn(doc_hidden_size, self.hidden_size))
			self.co_VFUs.append(VFU(doc_hidden_size))

			#  self-attention
			self.self_bi_attns.append(BilinearSelfAttn(doc_hidden_size, self.hidden_size, self.dropout))
			self.self_VFUs.append(VFU(2 * self.hidden_size))

			#  aggregation
			self.aggregate_rnns.append(
				BiRNN(
					input_size=2 * self.hidden_size,
					hidden_size=self.hidden_size,
					num_layers=1,
					dropout=self.dropout,
					rnn_type=self.rnn_type,
				)
			)

		self.out1_dim = 2 * hidden_size
		self.out2_dim = 2 * hidden_size

	def forward(self, c, c_mask, q, q_mask):
		"""Inputs:
		x1_list = document word indices of different vocabs		list([batch * len_d])
		x1_f_list = document word features indices				list([batch * len_d])
		x1_mask = document padding mask      					[batch * len_d]
		x2_list = question word indices of different vocabs		list([batch * len_q])
		x2_f_list = document word features indices  			list([batch * len_q])
		x2_mask = question padding mask        					[batch * len_q]
		"""

		# Encode document with RNN
		c = self.encoding_rnn(c, c_mask)

		# Encode question with RNN
		q = self.encoding_rnn(q, q_mask)

		# Align and aggregate
		d_ii, q_ii = c, q
		for i in range(self.hop):
			#  residuals from encoders
			d_ii = c + d_ii + c * d_ii
			q_ii = q + q_ii + q * q_ii

			#  co-attention
			q_h, c_h = self.co_attn_layers[i](d_ii, q_ii, c_mask, q_mask)
			c_i = self.co_VFUs[i](d_ii, q_h)
			q_i = self.co_VFUs[i](q_ii, c_h)

			#  self-attention
			d, d_h = self.self_bi_attns[i](c_i, c_mask)
			d_i = self.self_VFUs[i](d, d_h)

			#  aggregation
			d_ii = self.aggregate_rnns[i](d_i, c_mask)
			q_ii = self.aggregate_rnns[i](q_i, q_mask)

		return d_ii, q_ii
