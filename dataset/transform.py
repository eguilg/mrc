import torch


def pad(seqs, max_len, pad_val):
	""" padding """
	tmp = []
	for d in seqs:
		if len(d) > max_len:
			tmp.append(d[: max_len])
		elif len(d) < max_len:
			tmp.append(d + [pad_val] * (max_len - len(d)))
		else:
			tmp.append(d)
	data_array = tmp
	return data_array


class MaiIndexTransform(object):
	base_vocab = {}
	sgns_vocab = {}
	flag_vocab = {}

	def __init__(self, jieba_base_v=None, jieba_sgns_v=None, jieba_flag_v=None,
				 pyltp_base_v=None, pyltp_sgns_v=None, pyltp_flag_v=None):
		self.base_vocab['jieba'] = jieba_base_v
		self.base_vocab['pyltp'] = pyltp_base_v

		self.sgns_vocab['jieba'] = jieba_sgns_v
		self.sgns_vocab['pyltp'] = pyltp_sgns_v

		self.flag_vocab['jieba'] = jieba_flag_v
		self.flag_vocab['pyltp'] = pyltp_flag_v

	def __call__(self, item, method):
		res = {
			'raw': item,
			'method': method,
			'c_base_idx': self.base_vocab[method].words_to_idxs(item['article_tokens']),
			'c_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['article_tokens']),
			'c_flag_idx': self.flag_vocab[method].words_to_idxs(item['article_flags']),
			'c_in_q': [1.0 if w in item['question_tokens'] else 0.0 for w in item['article_tokens']],

			'c_len': len(item['article_tokens']),

			'q_base_idx': self.base_vocab[method].words_to_idxs(item['question_tokens']),
			'q_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['question_tokens']),
			'q_flag_idx': self.flag_vocab[method].words_to_idxs(item['question_flags']),
			'q_in_c': [1.0 if w in item['article_tokens'] else 0.0 for w in item['question_tokens']],

			'q_len': len(item['question_tokens']),

		}
		if 'answer_token_start' in item:
			res.update({
				'start': item['answer_token_start'],
				'end': item['answer_token_end'],
				'r_starts': item['delta_token_ends'],
				'r_ends': item['delta_token_ends'],
				'r_scores': item['delta_rouges']
			})
		return res

	def batchify(self, res_batch):
		c_lens = [sample['c_len'] for sample in res_batch]
		q_lens = [sample['q_len'] for sample in res_batch]
		c_max_len = max(c_lens)
		q_max_len = max(q_lens)
		m = res_batch[0]['method']
		batch = {
			'raw': [sample['raw'] for sample in res_batch],
			'method': m,
			'c_base_idx': torch.LongTensor(pad([sample['c_base_idx'] for sample in res_batch], c_max_len, self.base_vocab[m].pad_idx)),
			'c_sgns_idx': torch.LongTensor(pad([sample['c_sgns_idx'] for sample in res_batch], c_max_len, self.sgns_vocab[m].pad_idx)),
			'c_flag_idx': torch.LongTensor(pad([sample['c_flag_idx'] for sample in res_batch], c_max_len, self.flag_vocab[m].pad_idx)),
			'c_in_q': torch.DoubleTensor(pad([sample['c_in_q'] for sample in res_batch], c_max_len, 0.0)),

			'c_lens': torch.LongTensor(c_lens),

			'q_base_idx': torch.LongTensor(pad([sample['q_base_idx'] for sample in res_batch], q_max_len, self.base_vocab[m].pad_idx)),
			'q_sgns_idx': torch.LongTensor(pad([sample['q_sgns_idx'] for sample in res_batch], q_max_len, self.sgns_vocab[m].pad_idx)),
			'q_flag_idx': torch.LongTensor(pad([sample['q_flag_idx'] for sample in res_batch], q_max_len, self.flag_vocab[m].pad_idx)),
			'q_in_c': torch.DoubleTensor(pad([sample['q_in_c'] for sample in res_batch], q_max_len, 0.0)),

			'q_lens': torch.LongTensor(q_lens)
		}
		if 'start' in res_batch[0]:
			batch.update({
				'start': torch.LongTensor([sample['start'] for sample in res_batch]),
				'end': torch.LongTensor([sample['end'] for sample in res_batch]),
				# 'r_starts': [sample['r_starts'] for sample in res_batch],
				# 'r_ends': [sample['r_ends'] for sample in res_batch],
				# 'r_scores': [sample['r_scores'] for sample in res_batch]
			})

		return batch
