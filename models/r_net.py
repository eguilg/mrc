"""Implementation of the R-Net based reader."""

import torch
import torch.nn as nn

from .layers.att_layer import SeqAttnMatch, SelfAttnMatch
from .layers.common_layer import Gate
from .layers.embedding_layer import MergedEmbedding
from .layers.pointer_layer import PointerNetwork
from .layers.rnn_layer import BiRNN


class R_Net(nn.Module):

	def __init__(self, param_dict, normalize=True):
		super(R_Net, self).__init__()
		# Store config
		self.param_dict = param_dict
		self.embed_lists = param_dict['embed_lists']
		self.num_features = param_dict['num_features']
		self.hidden_size = param_dict['hidden_size']
		self.dropout = param_dict['dropout']
		self.rnn_type = param_dict['rnn_type']

		self.merged_embeddings_jieba = MergedEmbedding(self.embed_lists['jieba'])
		self.merged_embeddings_pyltp = MergedEmbedding(self.embed_lists['pyltp'])
		self.merged_embeddings = {
			'jieba': self.merged_embeddings_jieba,
			'pyltp': self.merged_embeddings_pyltp
		}

		doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

		# Encoder
		self.encode_rnn = BiRNN(
			input_size=doc_input_size,
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

		self.ptr_net = PointerNetwork(
			x_size=doc_self_attn_hidden_size,
			y_size=question_hidden_size,
			hidden_size=self.hidden_size,
			dropout_rate=self.dropout,
			cell_type=nn.GRUCell,
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
		c = self.encode_rnn(torch.cat(crnn_input, -1), x1_mask)

		# Encode question with RNN
		q = self.encode_rnn(torch.cat(qrnn_input, -1), x2_mask)
		# print(c.mean())
		# Match questions to docs
		question_attn_hiddens = self.question_attn(c, q, x2_mask)
		rnn_input = self.question_attn_gate(torch.cat([c, question_attn_hiddens], -1))
		c = self.question_attn_rnn(rnn_input, x1_mask)

		# Match documents to themselves
		doc_self_attn_hiddens = self.doc_self_attn(c, x1_mask)
		rnn_input = self.doc_self_attn_gate(torch.cat([c, doc_self_attn_hiddens], -1))
		c = self.doc_self_attn_rnn(rnn_input, x1_mask)
		c = self.doc_self_attn_rnn2(c, x1_mask)

		# Predict
		start_scores, end_scores = self.ptr_net(c, q, x1_mask, x2_mask)

		return start_scores, end_scores
