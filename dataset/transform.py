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

			'q_base_idx': self.base_vocab[method].words_to_idxs(item['question_tokens']),
			'q_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['question_tokens']),
			'q_flag_idx': self.flag_vocab[method].words_to_idxs(item['question_flags']),
			'q_in_c': [1.0 if w in item['article_tokens'] else 0.0 for w in item['question_tokens']]

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
