"""Implementation of the Mnemonic Reader."""
import torch
import torch.nn as nn

from .layers.att_layer import SeqAttnMatch, SelfAttnMatch
from .layers.common_layer import SFU
from .layers.embedding_layer import MergedEmbedding
from .layers.pointer_layer import MemoryAnsPointer
from .layers.rnn_layer import BiRNN


class MnemonicReader(nn.Module):
	RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
	CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

	def __init__(self, param_dict, normalize=True):
		super(MnemonicReader, self).__init__()
		# Store config
		self.param_dict = param_dict
		self.embed_lists = param_dict['embed_lists']
		self.num_features = param_dict['num_features']
		self.hidden_size = param_dict['hidden_size']
		self.dropout = param_dict['dropout']
		self.rnn_type = param_dict['rnn_type']
		self.hop = param_dict['hop']
		self.merged_embeddings_jieba = MergedEmbedding(self.embed_lists['jieba'])
		self.merged_embeddings_pyltp = MergedEmbedding(self.embed_lists['pyltp'])
		self.merged_embeddings = {
			'jieba': self.merged_embeddings_jieba,
			'pyltp': self.merged_embeddings_pyltp
		}

		doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

		# Encoder
		self.encoding_rnn = BiRNN(
			input_size=doc_input_size,
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

		# Memmory-based Answer Pointer
		self.mem_ans_ptr = MemoryAnsPointer(
			x_size=2 * self.hidden_size,
			y_size=2 * self.hidden_size,
			hidden_size=self.hidden_size,
			hop=self.hop,
			dropout_rate=self.dropout,
			normalize=normalize
		)

	def forward(self, x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method):
		"""Inputs:
		x1_list = document word indices of different vocabs		list([batch * len_d])
		x1_f_list = document word features indices				list([batch * len_d])
		x1_mask = document padding mask      					[batch * len_d]
		x2_list = question word indices of different vocabs		list([batch * len_q])
		x2_f_list = document word features indices  			list([batch * len_q])
		x2_mask = question padding mask        					[batch * len_q]
		"""
		# Embed both document and question
		x1_emb = self.merged_embeddings[method](x1_list)
		x2_emb = self.merged_embeddings[method](x2_list)

		# Combine input
		crnn_input = [x1_emb]
		qrnn_input = [x2_emb]

		# Add manual features
		if self.num_features > 0:
			x1_f = torch.cat([f.unsqueeze(2) for f in x1_f_list], -1)
			x2_f = torch.cat([f.unsqueeze(2) for f in x2_f_list], -1)
			crnn_input.append(x1_f)
			qrnn_input.append(x2_f)

		# Encode document with RNN
		c = self.encoding_rnn(torch.cat(crnn_input, -1), x1_mask)

		# Encode question with RNN
		q = self.encoding_rnn(torch.cat(qrnn_input, -1), x2_mask)

		# Align and aggregate
		c_check = c
		for i in range(self.hop):
			q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
			c_bar = self.interactive_SFUs[i].forward(c_check,
													 torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
			c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
			c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
			c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)

		# Predict
		start_scores, end_scores = self.mem_ans_ptr(c_check, q, x1_mask, x2_mask)

		return start_scores, end_scores
