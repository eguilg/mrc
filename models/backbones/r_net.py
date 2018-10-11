"""Implementation of the R-Net based reader."""

import torch
import torch.nn as nn

from ..layers.att_layer import SeqAttnMatch, SelfAttnMatch
from ..layers.common_layer import Gate
from ..layers.rnn_layer import BiRNN


class R_Net(nn.Module):

	def __init__(self, input_size, hidden_size, dropout, rnn_type=nn.LSTM):
		super(R_Net, self).__init__()
		# Store config
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.dropout = dropout
		self.rnn_type = rnn_type

		# Encoder
		self.encode_rnn = BiRNN(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		# Output sizes of rnn encoder
		doc_hidden_size = 2 * self.hidden_size
		question_hidden_size = 2 * self.hidden_size

		# Gated-attention-based RNN of the whole question
		self.question_attn = SeqAttnMatch(question_hidden_size, identity=False)
		self.question_attn_gate = Gate(doc_hidden_size + question_hidden_size)
		self.question_attn_rnn = BiRNN(
			input_size=doc_hidden_size + question_hidden_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		question_attn_hidden_size = 2 * self.hidden_size

		# Self-matching-attention-baed RNN of the whole doc
		self.doc_self_attn = SelfAttnMatch(question_attn_hidden_size, identity=False)
		self.doc_self_attn_gate = Gate(question_attn_hidden_size + question_attn_hidden_size)
		self.doc_self_attn_rnn = BiRNN(
			input_size=question_attn_hidden_size + question_attn_hidden_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
		)

		doc_self_attn_hidden_size = 2 * self.hidden_size

		self.doc_self_attn_rnn2 = BiRNN(
			input_size=doc_self_attn_hidden_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
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
		c = self.encode_rnn(c, c_mask)

		# Encode question with RNN
		q = self.encode_rnn(q, q_mask)

		# Match questions to docs
		question_attn_hiddens = self.question_attn(c, q, q_mask)
		rnn_input = self.question_attn_gate(torch.cat([c, question_attn_hiddens], -1))
		c = self.question_attn_rnn(rnn_input, c_mask)

		# Match documents to themselves
		doc_self_attn_hiddens = self.doc_self_attn(c, c_mask)
		rnn_input = self.doc_self_attn_gate(torch.cat([c, doc_self_attn_hiddens], -1))
		c = self.doc_self_attn_rnn(rnn_input, c_mask)
		c = self.doc_self_attn_rnn2(c, c_mask)

		return c, q
