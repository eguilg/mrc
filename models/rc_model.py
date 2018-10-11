""" Reader wrapper """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import BACKBONE_TYPES, RNN_TYPES, POINTER_TYPES
from .layers.embedding_layer import MergedEmbedding


class RCModel(nn.Module):

	def __init__(self, param_dict, embed_lists, normalize=True):
		super(RCModel, self).__init__()
		# Store config
		self.param_dict = param_dict
		self.num_features = param_dict['num_features']
		self.num_qtype = param_dict['num_qtype']
		self.hidden_size = param_dict['hidden_size']
		self.dropout = param_dict['dropout']
		self.backbone_kwarg = param_dict['backbone_kwarg']
		self.ptr_kwarg = param_dict['ptr_kwarg']

		try:
			self.backbone_type = BACKBONE_TYPES[param_dict['backbone_type']]
		except KeyError:
			raise KeyError('Wrong backbone type')
		try:
			self.rnn_type = RNN_TYPES[param_dict['rnn_type']]
		except KeyError:
			raise KeyError('Wrong rnn type')
		try:
			self.ptr_type = POINTER_TYPES[param_dict['ptr_type']]
		except KeyError:
			raise KeyError('Wrong pointer type')

		self.merged_embeddings_jieba = MergedEmbedding(embed_lists['jieba'])
		self.merged_embeddings_pyltp = MergedEmbedding(embed_lists['pyltp'])
		self.merged_embeddings = {
			'jieba': self.merged_embeddings_jieba,
			'pyltp': self.merged_embeddings_pyltp
		}

		self.doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

		self.backbone = self.backbone_type(
			input_size=self.doc_input_size,
			hidden_size=self.hidden_size,
			dropout=self.dropout,
			rnn_type=self.rnn_type,
			**self.backbone_kwarg
		)

		self.ptr_net = self.ptr_type(
			x_size=self.backbone.out1_dim,
			y_size=self.backbone.out2_dim,
			hidden_size=self.hidden_size,
			dropout_rate=self.dropout,
			normalize=normalize,
			**self.ptr_kwarg
		)

		self.qtype_net = nn.Linear(
			in_features=2 * self.hidden_size,
			out_features=self.num_qtype,
			bias=True
		)

		self.isin_net = nn.Linear(
			in_features=2 * self.hidden_size,
			out_features=1,
			bias=True
		)

	def reset_embeddings(self, embed_lists):
		self.merged_embeddings_jieba = MergedEmbedding(embed_lists['jieba'])
		self.merged_embeddings_pyltp = MergedEmbedding(embed_lists['pyltp'])
		self.merged_embeddings = {
			'jieba': self.merged_embeddings_jieba,
			'pyltp': self.merged_embeddings_pyltp
		}
		self.doc_input_size = self.merged_embeddings['jieba'].output_dim + self.num_features

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

		# cat document
		c = torch.cat(crnn_input, -1)

		# cat question
		q = torch.cat(qrnn_input, -1)

		c, q = self.backbone(c, x1_mask, q, x2_mask)

		score_s, score_e = self.ptr_net(c, q, x1_mask, x2_mask)

		if self.training:
			qtype_vec = F.sigmoid(self.qtype_net(q[:, -1, :]))
			c_in_a = F.sigmoid(self.isin_net(c).squeeze(-1))
			q_in_a = F.sigmoid(self.isin_net(q).squeeze(-1))
			return score_s, score_e, (qtype_vec, c_in_a, q_in_a)
		else:
			return score_s, score_e, None

	@staticmethod
	def decode(score_s, score_e, top_n=1, max_len=None):
		"""Take argmax of constrained score_s * score_e.

		Args:
			score_s: independent start predictions
			score_e: independent end predictions
			top_n: number of top scored pairs to take
			max_len: max span length to consider
		"""
		pred_s = []
		pred_e = []
		pred_score = []
		max_len = max_len or score_s.size(1)
		for i in range(score_s.size(0)):
			# Outer product of scores to get full p_s * p_e matrix
			scores = torch.ger(score_s[i], score_e[i])

			# Zero out negative length and over-length span scores
			scores.triu_().tril_(max_len - 1)

			# Take argmax or top n
			scores = scores.numpy()
			scores_flat = scores.flatten()
			if top_n == 1:
				idx_sort = [np.argmax(scores_flat)]
			elif len(scores_flat) < top_n:
				idx_sort = np.argsort(-scores_flat)
			else:
				idx = np.argpartition(-scores_flat, top_n)[0:top_n]
				idx_sort = idx[np.argsort(-scores_flat[idx])]
			s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
			pred_s.append(s_idx[0])
			pred_e.append(e_idx[0])
			pred_score.append(scores_flat[idx_sort][0])
		del score_s, score_e
		return pred_s, pred_e, pred_score

	@staticmethod
	def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
		"""Take argmax of constrained score_s * score_e. Except only consider
		spans that are in the candidates list.
		"""
		pred_s = []
		pred_e = []
		pred_score = []
		for i in range(score_s.size(0)):
			# Extract original tokens stored with candidates
			tokens = candidates[i]['input']
			cands = candidates[i]['cands']

			# if not cands:
			# 	# try getting from globals? (multiprocessing in pipeline mode)
			# 	from ..pipeline.wrmcqa import PROCESS_CANDS
			# 	cands = PROCESS_CANDS
			if not cands:
				raise RuntimeError('No candidates given.')

			# Score all valid candidates found in text.
			# Brute force get all ngrams and compare against the candidate list.
			max_len = max_len or len(tokens)
			scores, s_idx, e_idx = [], [], []
			for s, e in tokens.ngrams(n=max_len, as_strings=False):
				span = tokens.slice(s, e).untokenize()
				if span in cands or span.lower() in cands:
					# Match! Record its score.
					scores.append(score_s[i][s] * score_e[i][e - 1])
					s_idx.append(s)
					e_idx.append(e - 1)

			if len(scores) == 0:
				# No candidates present
				pred_s.append([])
				pred_e.append([])
				pred_score.append([])
			else:
				# Rank found candidates
				scores = np.array(scores)
				s_idx = np.array(s_idx)
				e_idx = np.array(e_idx)

				idx_sort = np.argsort(-scores)[0:top_n]
				pred_s.append(s_idx[idx_sort])
				pred_e.append(e_idx[idx_sort])
				pred_score.append(scores[idx_sort])
		del score_s, score_e, candidates
		return pred_s, pred_e, pred_score
