""" samplers """

from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class MethodBasedBatchSampler(Sampler):
	def __init__(self, data_source, batch_size, shuffle=True, seed=502):
		self.data_source = data_source
		self.batch_size = batch_size
		self.method_dict = defaultdict(int)
		self.shuffle = shuffle
		self.seed = seed
		for _, _, method, _ in self.data_source:
			self.method_dict[method] += 1

	def __iter__(self):
		method_batch_dict = defaultdict(list)
		shuffled_indices = list(range(len(self.data_source)))
		if self.shuffle:
			np.random.seed(self.seed)
			np.random.shuffle(shuffled_indices)
		for idx in shuffled_indices:
			_, _, method, _ = self.data_source[idx]
			method_batch_dict[method].append(int(idx))
			if len(method_batch_dict[method]) == self.batch_size:
				yield method_batch_dict[method]
				method_batch_dict[method] = []
		for b in method_batch_dict.values():
			if len(b) > 0:
				yield b

	def __len__(self):
		lens = [(ll + self.batch_size - 1) // self.batch_size for ll in self.method_dict.values()]
		return sum(lens)
