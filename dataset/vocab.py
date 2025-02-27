""" simple vocab class load from prepared .pkl files or vocab&embeddings"""

import numpy as np

from utils.serialization import read_pkl


class Vocab(object):

	def __init__(self, vocab_path=None, embed_path=None, tokens=None, embeds=None):
		self.pad = '<pad>'
		self.unk = '<unk>'
		self.index2word = [self.pad, self.unk]
		self.word2index = {self.pad: 0, self.unk: 1}
		self.pad_idx = 0
		self.unk_idx = 1
		self.size = 2
		if vocab_path and embed_path is not None:
			self.load(vocab_path, embed_path)
		else:
			self.__init_words__(tokens, embeds)

	def __init_words__(self, tokens, embeds):
		self.dim = embeds.shape[1]
		self.embeddings = np.zeros([len(self.index2word), self.dim])

		if tokens and embeds is not None:
			self.index2word.append(tokens)
			self.word2index.update(zip(tokens, range(2, len(tokens) + 2)))
			self.embeddings = np.concatenate([self.embeddings, np.array(embeds)])
			# set embedding of '<unk>' to mean value of given embeds
			self.embeddings[self.unk_idx] = np.mean(embeds, axis=0)
			self.size = self.embeddings.shape[0]

	def load(self, vocab_pkl_path, embed_pkl_path):
		tokens = read_pkl(vocab_pkl_path)
		embeds = read_pkl(embed_pkl_path)
		if tokens is not None and embeds is not None:
			self.__init_words__(tokens, embeds)
		else:
			raise ValueError('wrong token/embeds input')

	def words_to_idxs(self, seq):
		return [self.word2index[w] if w in self.word2index.keys() else self.unk_idx for w in seq]

	def idxs_to_words(self, idxs):
		return [self.index2word[idx] if
				(isinstance(idx, int) and 0 < idx < len(self.index2word))
				else '' for idx in idxs]
