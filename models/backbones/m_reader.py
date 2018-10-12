"""Implementation of the Mnemonic Reader."""
import torch
import torch.nn as nn

from ..layers.att_layer import SeqAttnMatch, SelfAttnMatch
from ..layers.common_layer import SFU
from ..layers.rnn_layer import BiRNN


class MnemonicReader(nn.Module):

	def __init__(self, input_size, hidden_size, dropout, rnn_type=nn.LSTM, hop=3):

		super(MnemonicReader, self).__init__()
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

		# Interactive aligning, self aligning and aggregating
		self.interactive_aligners = nn.ModuleList()
		self.interactive_SFUs = nn.ModuleList()
		self.self_aligners = nn.ModuleList()
		self.self_SFUs = nn.ModuleList()
		self.aggregate_rnns = nn.ModuleList()
		for i in range(self.hop):
			# interactive aligner
			self.interactive_aligners.append(SeqAttnMatch(doc_hidden_size, identity=True))
			self.interactive_SFUs.append(SFU(doc_hidden_size, 3 * doc_hidden_size))

			# self aligner
			self.self_aligners.append(SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
			self.self_SFUs.append(SFU(doc_hidden_size, 3 * doc_hidden_size))

			# aggregating
			self.aggregate_rnns.append(
				BiRNN(
					input_size=doc_hidden_size,
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
		c_check = c
		for i in range(self.hop):
			#  residuals from encoders
			c_check = c + c_check + c * c_check

			#  interactive align
			q_tilde = self.interactive_aligners[i].forward(c_check, q, q_mask)
			c_bar = self.interactive_SFUs[i].forward(c_check,
													 torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))

			#  self align
			c_tilde = self.self_aligners[i].forward(c_bar, c_mask)
			c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
			c_check = self.aggregate_rnns[i].forward(c_hat, c_mask)

		return c_check, q
