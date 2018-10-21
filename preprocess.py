# coding = utf-8
import argparse
import logging
import multiprocessing as mp
import os
import re
from functools import partial
from pyltp import Segmentor, Postagger

import jieba
import jieba.posseg as pseg
import pandas as pd
from gensim.models import Word2Vec

from metrics.rouge import RougeL
from utils.serialization import *

ltp_seg = Segmentor()
ltp_pos = Postagger()


def trans_to_df(raw_data_path):
	"""
	transfer json file to dataframe
	:param raw_data_path: path of raw data of json file
	:return: article_df, qa_df
	"""

	data = read_json(raw_data_path)
	questions = []
	articles = []

	for dc in data:
		temp = [dc['article_id'], dc['article_type'], dc['article_title'], dc['article_content']]
		articles.append(temp)
		for items in dc['questions']:
			r = [dc['article_id']]
			r = r + list(items.values())
			questions.append(r)

	article_columns = ['article_id', 'article_type', 'article_title', 'article_content']
	if 'answer' in data[0]['questions'][0]:
		question_columns = ['article_id', 'question_id', 'question', 'answer', 'question_type']
	else:
		question_columns = ['article_id', 'question_id', 'question']
	article_df = pd.DataFrame(data=articles, columns=article_columns)
	qa_df = pd.DataFrame(data=questions, columns=question_columns)
	qa_df.fillna('', inplace=True)
	return article_df, qa_df


def clean_text(article_df: pd.DataFrame, qa_df: pd.DataFrame):
	"""
	remove \u3000, swap \r\n with \001, swap \t with ' ',
	change numbers to half written
	:param df:
	:return: cleaned df
	"""

	def clean(row):
		row = re.sub('[\u3000\t]', ' ', row)
		row = re.sub('\s{2,}', '', row)
		# row = re.sub('[“”]', '', row)
		row = re.sub('[\r\n]', ' ', row)

		# p_l = re.compile(r'\s+([\u4e00-\u9fa5, ]{1})')
		# p_r = re.compile(r'([\u4e00-\u9fa5, ]{1})\s+')
		# row = p_l.sub('\1', row)
		# row = p_r.sub('\1', row)

		row = re.sub(r'０', '0', row)
		row = re.sub(r'１', '1', row)
		row = re.sub(r'２', '2', row)
		row = re.sub(r'３', '3', row)
		row = re.sub(r'４', '4', row)
		row = re.sub(r'５', '5', row)
		row = re.sub(r'６', '6', row)
		row = re.sub(r'７', '7', row)
		row = re.sub(r'８', '8', row)
		row = re.sub(r'９', '9', row)
		row = re.sub(r'．', '.', row)

		row = re.sub(r'ａ', 'a', row)
		row = re.sub(r'ｂ', 'b', row)
		row = re.sub(r'ｃ', 'c', row)
		row = re.sub(r'ｄ', 'd', row)
		row = re.sub(r'ｅ', 'e', row)
		row = re.sub(r'ｆ', 'f', row)
		row = re.sub(r'ｇ', 'g', row)
		row = re.sub(r'ｈ', 'h', row)
		row = re.sub(r'ｉ', 'i', row)
		row = re.sub(r'ｊ', 'j', row)
		row = re.sub(r'ｋ', 'k', row)
		row = re.sub(r'ｌ', 'l', row)
		row = re.sub(r'ｍ', 'm', row)
		row = re.sub(r'ｎ', 'n', row)
		row = re.sub(r'ｏ', 'o', row)
		row = re.sub(r'ｐ', 'p', row)
		row = re.sub(r'ｑ', 'q', row)
		row = re.sub(r'ｒ', 'r', row)
		row = re.sub(r'ｓ', 's', row)
		row = re.sub(r'ｔ', 't', row)
		row = re.sub(r'ｕ', 'u', row)
		row = re.sub(r'ｖ', 'v', row)
		row = re.sub(r'ｗ', 'w', row)
		row = re.sub(r'ｘ', 'x', row)
		row = re.sub(r'ｙ', 'y', row)
		row = re.sub(r'ｚ', 'z', row)

		row = re.sub(r'Ａ', 'A', row)
		row = re.sub(r'Ｂ', 'B', row)
		row = re.sub(r'Ｃ', 'C', row)
		row = re.sub(r'Ｄ', 'D', row)
		row = re.sub(r'Ｅ', 'E', row)
		row = re.sub(r'Ｆ', 'F', row)
		row = re.sub(r'Ｇ', 'G', row)
		row = re.sub(r'Ｈ', 'H', row)
		row = re.sub(r'Ｉ', 'I', row)
		row = re.sub(r'Ｊ', 'J', row)
		row = re.sub(r'Ｋ', 'K', row)
		row = re.sub(r'Ｌ', 'L', row)
		row = re.sub(r'Ｍ', 'M', row)
		row = re.sub(r'Ｎ', 'N', row)
		row = re.sub(r'Ｏ', 'O', row)
		row = re.sub(r'Ｐ', 'P', row)
		row = re.sub(r'Ｑ', 'Q', row)
		row = re.sub(r'Ｒ', 'R', row)
		row = re.sub(r'Ｓ', 'S', row)
		row = re.sub(r'Ｔ', 'T', row)
		row = re.sub(r'Ｕ', 'U', row)
		row = re.sub(r'Ｖ', 'V', row)
		row = re.sub(r'Ｗ', 'W', row)
		row = re.sub(r'Ｘ', 'X', row)
		row = re.sub(r'Ｙ', 'Y', row)
		row = re.sub(r'Ｚ', 'Z', row)

		if len(row) > 0 and row[-1] == '。':
			row = row[:-1].strip()
		return row

	def merge(row):
		"""
		merge the article title and content
		:param row:
		:return:
		"""
		row['article'] = row['article_title'] + '。' + row['article_content']
		return row

	article_df['article_title'] = article_df['article_title'].apply(clean)
	article_df['article_content'] = article_df['article_content'].apply(clean)

	article_df = article_df.apply(merge, axis=1)
	article_df.drop(['article_title', 'article_content'], axis=1, inplace=True)

	qa_df['question'] = qa_df['question'].apply(clean)
	if 'answer' in qa_df.columns:
		qa_df['answer'] = qa_df['answer'].apply(clean)
		qa_df['answer'] = qa_df['answer'].apply(str.strip)
		answers = qa_df[qa_df['answer'] != '']['answer'].values
		drop_list = ['。', '，', '、', '；', '：', '？', '！', ' ', '.', '?', '!', ';', ':', ',', '-', '...', '..', '....']
		answers = [answer[:-1].strip() if answer[-1] in drop_list else answer for answer in answers]
		answers = [answer[1:].strip() if answer[0] in drop_list else answer for answer in answers]
		qa_df.loc[qa_df['answer'] != '', 'answer'] = answers
	return article_df, qa_df


def _apply_cut_jieba(df, col):
	def _cut_jieba(row):
		"""
		cut the sentences into tokens
		:param row:
		:return:
		"""
		cut_words = []
		cut_flags = []
		if '。' in row:
			row = row.split('。')
			for idx, s in enumerate(row):
				if idx != len(row) - 1:
					s = s + '。'
				s_cut = list(pseg.lcut(s, HMM=False))
				cut_words.extend([c.word for c in s_cut])
				cut_flags.extend([c.flag for c in s_cut])
		else:
			s_cut = list(pseg.lcut(row, HMM=False))
			cut_words = [c.word for c in s_cut]
			cut_flags = [c.flag for c in s_cut]

		new_row = pd.Series()
		new_row['tokens'] = cut_words
		new_row['flags'] = cut_flags
		return new_row

	sentence_cut = df[col].apply(_cut_jieba)
	return sentence_cut


def _apply_cut_pyltp(df, col):
	def _cut_pyltp(row):
		"""
		cut the sentences into tokens
		:param row:
		:return:
		"""
		cut_words = []
		cut_flags = []
		if '。' in row:
			row = row.split('。')
			for idx, s in enumerate(row):
				if idx != len(row) - 1:
					s = s + '。'
				tokens = list(ltp_seg.segment(s))
				cut_words.extend(tokens)
				cut_flags.extend(list(ltp_pos.postag(tokens)))
		else:
			tokens = list(ltp_seg.segment(row))
			cut_words = tokens
			cut_flags = list(ltp_pos.postag(tokens))

		new_row = pd.Series()
		new_row['tokens'] = cut_words
		new_row['flags'] = cut_flags
		return new_row

	sentence_cut = df[col].apply(_cut_pyltp)
	return sentence_cut


def parallel_cut(df, col, method):
	n_cpu = mp.cpu_count()
	with mp.Pool(processes=n_cpu) as p:
		split_dfs = np.array_split(df, n_cpu)

		if method == 'jieba':
			pool_results = p.map(partial(_apply_cut_jieba, col=col), split_dfs)
		elif method == 'pyltp':
			pool_results = p.map(partial(_apply_cut_pyltp, col=col), split_dfs)
		else:
			pool_results = p.map(partial(_apply_cut_jieba, col=col), split_dfs)

	# merging parts processed by different processes
	res = pd.concat(pool_results, axis=0)
	return res


def clean_token(article_df: pd.DataFrame, qa_df: pd.DataFrame):
	"""
	clean data on token level
	:param article_df:
	:param qa_df:
	:return:
	"""

	def clean(row, token_col, flag_col):
		tokens_cleaned = []
		flags_cleaned = []
		for token, flag in zip(row[token_col], row[flag_col]):
			token = token.strip()
			if token != '':
				tokens_cleaned.append(token)
				flags_cleaned.append(flag)

		row[token_col] = tokens_cleaned
		row[flag_col] = flags_cleaned

		return row

	article_df = article_df.apply(lambda row: clean(row, 'article_tokens', 'article_flags'), axis=1)
	qa_df = qa_df.apply(lambda row: clean(row, 'question_tokens', 'question_flags'), axis=1)
	if 'answer_tokens' in qa_df.columns:
		qa_df = qa_df.apply(lambda row: clean(row, 'answer_tokens', 'answer_flags'), axis=1)

	return article_df, qa_df


def _apply_sample_article(df: pd.DataFrame, article_tokens_col, article_flags_col,
						  question_tokens_col,
						  max_token_len=500):
	def _sample_article(row, article_tokens_col, article_flags_col, question_tokens_col, max_token_len=500):

		"""

		:param row:
		:param article_tokens_col:
		:param article_flags_col:
		:param question_tokens_col:
		:param max_token_len:
		:return:
		"""
		article_tokens = row[article_tokens_col]
		article_flags = row[article_flags_col]
		question_tokens = row[question_tokens_col]

		if len(article_tokens) <= max_token_len:
			return row
		sentences, sentences_f = [], []
		cur_s, cur_s_f = [], []
		question = ''.join(question_tokens)

		cand, cand_f = [], []
		rl = RougeL()
		for idx, (token, flag) in enumerate(zip(article_tokens, article_flags)):
			cur_s.append(token)
			cur_s_f.append(flag)

			if token in '。' or idx == len(article_tokens) - 1:
				if len(cur_s) >= 2:
					sentences.append(cur_s)
					sentences_f.append(cur_s_f)
					cur_s_str = ''.join(cur_s)
					rl.add_inst(cur_s_str, question)
					if rl.p_scores[-1] == 1.0:
						rl.r_scores[-1] = 1.0
				cur_s, cur_s_f = [], []
				continue

		if '。' not in ''.join(article_tokens):
			row[article_tokens_col] = sentences[0]
			row[article_flags_col] = sentences_f[0]
			return row

		scores = rl.r_scores
		s_rank = np.zeros(len(sentences))
		arg_sorted = list(reversed(np.argsort(scores)))

		for i in range(10):
			if i >= len(sentences):
				break
			pos = arg_sorted[i]
			score = scores[pos]
			if pos in [0, 1, len(sentences) - 1, len(sentences) - 2] or score == 0:
				continue
			block_scores = np.array([0.5 * score, 0.9 * score, score, score, 0.9 * score, 0.5 * score, 0.4 * score])
			# block_scores = np.array([0.25*score, 0.5*score, score, 0.8*score, 0.64*score, 0.512*score, 0.4096*score])
			block = s_rank[pos - 2: pos + 5]
			block_scores = block_scores[:len(block)]
			block_scores = np.max(np.stack([block_scores, block]), axis=0)
			s_rank[pos - 2: pos + 5] = block_scores

		rank = list(reversed(np.argsort(s_rank)))
		flag = [0 for i in range(len(sentences))]
		flag[0], flag[1], flag[-1], flag[-2] = 1, 1, 1, 1
		cur_len = len(sentences[0]) + len(sentences[1]) + len(sentences[-1]) + len(sentences[-2])

		for pos in rank:
			if cur_len < max_token_len:
				if s_rank[pos] > 0:
					flag[pos] = 1
					cur_len += len(sentences[pos])
			else:
				break

		for i in range(len(flag)):
			if flag[i] != 0:
				cand.extend(sentences[i])
				cand_f.extend(sentences_f[i])

		row[article_tokens_col] = cand[:max_token_len]
		row[article_flags_col] = cand_f[:max_token_len]

		return row

	df = df.apply(
		lambda row: _sample_article(row, article_tokens_col, article_flags_col, question_tokens_col, max_token_len),
		axis=1)

	return df


def parallel_sample_article(article_df: pd.DataFrame, qa_df: pd.DataFrame, max_token_len=500):
	sample_df = pd.merge(article_df, qa_df, how='inner', on=['article_id'])
	n_cpu = mp.cpu_count()
	with mp.Pool(processes=n_cpu) as p:
		split_dfs = np.array_split(sample_df, n_cpu)
		pool_results = p.map(partial(_apply_sample_article,
									 article_tokens_col='article_tokens',
									 article_flags_col='article_flags',
									 question_tokens_col='question_tokens',
									 max_token_len=max_token_len), split_dfs)

	# merging parts processed by different processes
	res = pd.concat(pool_results, axis=0)

	return res


def _apply_find_gold_span(sample_df: pd.DataFrame, article_tokens_col, question_tokens_col, answer_tokens_col):
	def _find_golden_span(row, article_tokens_col, question_tokens_col, answer_tokens_col):

		article_tokens = row[article_tokens_col]
		question_tokens = row[question_tokens_col]
		answer_tokens = row[answer_tokens_col]
		row['answer_token_start'] = -1
		row['answer_token_end'] = -1
		row['delta_token_starts'] = []
		row['delta_token_ends'] = []
		row['delta_rouges'] = []
		row['max_rouge'] = 0
		rl = RougeL()
		rl_q = RougeL()
		ground_ans = ''.join(answer_tokens).strip()
		questrin_str = ''.join(question_tokens).strip()
		len_p = len(article_tokens)
		len_a = len(answer_tokens)
		s2 = set(ground_ans)
		star_spans = []
		end_spans = []
		rl_q_idx = []
		for i in range(len_p - len_a + 1):
			for t_len in range(len_a - 2, len_a + 3):
				if t_len <= 0 or i + t_len > len_p:
					continue
				cand_ans = ''.join(article_tokens[i:i + t_len]).strip()
				s1 = set(cand_ans)
				mlen = max(len(s1), len(s2))
				iou = len(s1.intersection(s2)) / mlen if mlen != 0 else 0.0
				if iou >= 0.2:
					rl.add_inst(cand_ans, ground_ans)
					if rl.inst_scores[-1] == 1.0:
						s = max(i - 7, 0)
						cand_ctx = ''.join(article_tokens[s:i + t_len + 3]).strip()
						rl_q.add_inst(cand_ctx, questrin_str)
						rl_q_idx.append(len(star_spans))

					star_spans.append(i)
					end_spans.append(i + t_len - 1)
		if len(star_spans) == 0:
			return row
		else:
			max_score = np.max(rl.inst_scores)
			row['max_rouge'] = max_score
			if max_score == 1:
				best_idx = rl_q_idx[int(np.argmax(rl_q.r_scores))]
			else:
				best_idx = np.argmax(rl.inst_scores)
			if best_idx is not None:
				row['answer_token_start'] = star_spans[best_idx]
				row['answer_token_end'] = end_spans[best_idx]

			row['delta_token_starts'] = star_spans
			row['delta_token_ends'] = end_spans
			row['delta_rouges'] = rl.inst_scores

		return row

	sample_df = sample_df.apply(
		lambda row: _find_golden_span(row,
									  article_tokens_col,
									  question_tokens_col,
									  answer_tokens_col),
		axis=1)

	return sample_df


def parallel_find_gold_span(sample_df: pd.DataFrame):
	n_cpu = mp.cpu_count()
	with mp.Pool(processes=n_cpu) as p:
		split_dfs = np.array_split(sample_df, n_cpu)
		pool_results = p.map(partial(_apply_find_gold_span,
									 article_tokens_col='article_tokens',
									 question_tokens_col='question_tokens',
									 answer_tokens_col='answer_tokens'),
							 split_dfs)

	# merging parts processed by different processes
	res = pd.concat(pool_results, axis=0)

	return res


def process_dataset(args, raw_file_path):
	if args.method == 'jieba':
		if osp.isfile(args.jieba_big_dict):
			jieba.set_dictionary(args.jieba_big_dict)
		jieba.del_word('日电')
		jieba.del_word('日刊')
		jieba.del_word('亿美元')
		jieba.del_word('英国伦敦')
		jieba.setLogLevel(logging.INFO)
	elif args.method == 'pyltp':
		ltp_seg.load(args.ltp_cws_path)
		ltp_pos.load(args.ltp_pos_path)

	# load and clean
	adf, qadf = trans_to_df(raw_file_path)
	adf, qadf = clean_text(adf, qadf)

	# cut words
	article_cut = parallel_cut(adf, 'article', args.method)
	adf.drop(['article'], axis=1, inplace=True)
	adf['article_tokens'] = article_cut['tokens']
	adf['article_flags'] = article_cut['flags']
	question_cut = parallel_cut(qadf, 'question', args.method)
	qadf.drop(['question'], axis=1, inplace=True)
	qadf['question_tokens'] = question_cut['tokens']
	qadf['question_flags'] = question_cut['flags']
	if 'answer' in qadf.columns and not args.test:
		ans_cut = parallel_cut(qadf, 'answer', args.method)
		qadf['answer_tokens'] = ans_cut['tokens']
		qadf['answer_flags'] = ans_cut['flags']
	if args.method == 'pyltp':
		ltp_pos.release()
		ltp_seg.release()

	adf, qadf = clean_token(adf, qadf)

	# sample article
	if 'answer' in qadf.columns and not args.test:
		sample_df = parallel_sample_article(adf, qadf, args.seq_len_train)
		sample_df = parallel_find_gold_span(sample_df)
	else:
		sample_df = parallel_sample_article(adf, qadf, args.seq_len_test)

	croups = list(adf['article_tokens']) + list(qadf['question_tokens'])
	flag_croups = list(adf['article_flags']) + list(qadf['question_flags'])
	if 'answer' in qadf.columns and not args.test:
		croups += list(qadf['answer_tokens'])
		flag_croups += list(qadf['answer_flags'])
	sample_df = sample_df.to_dict(orient='records')

	return sample_df, croups, flag_croups


def gen_w2v(args, total_croups, total_flag_croups, iter=60):
	print('training token vocab...')
	token_wv = Word2Vec(total_croups,
						size=args.token_emb_dim,
						window=5, compute_loss=True,
						min_count=2, iter=iter,
						workers=mp.cpu_count()).wv

	print('training flag vocab...')
	flag_wv = Word2Vec(total_flag_croups,
					   size=args.flag_emb_dim,
					   window=5, compute_loss=True, iter=iter,
					   workers=mp.cpu_count()).wv

	return token_wv, flag_wv


def gen_sgns_embed(args, tokens_in):
	print('generating sgns vocab and embeddings...')
	reader = bz2_vocab_reader(args.sgns_vocab_path)
	v_dict = {}
	total = len(tokens_in)
	tokens_in = set(tokens_in)
	for w, v in reader:
		if w in tokens_in:
			v_dict[w] = v
			tokens_in.remove(w)
	print('sgns covered {} of vocab'.format(len(v_dict) * 1. / total))
	return list(v_dict.keys()), np.array(list(v_dict.values()))


def main(args):
	if args.test:
		raw_dir = args.test_raw_path
		gen_dir = args.test_gen_path
		max_len = args.seq_len_test
	else:
		raw_dir = args.train_raw_path
		gen_dir = args.train_gen_path
		max_len = args.seq_len_train

	for raw_file in os.listdir(raw_dir):
		# path stuff
		raw_file_path = osp.join(raw_dir, raw_file)
		raw_file_name = osp.splitext(raw_file)[0]
		out_dir = osp.join(gen_dir, raw_file_name)
		samples_out_dir = osp.join(out_dir, 'samples_' + args.method + str(max_len))
		samples_out_path = osp.join(out_dir, 'total_samples_' + args.method + str(max_len) + '.json')

		if not osp.isfile(samples_out_path):
			samples, croups, flag_croups = process_dataset(args, raw_file_path)
			write_json(samples, samples_out_path)
			write_json(croups, osp.join(out_dir, 'croups_' + args.method + '.json'))
			write_json(flag_croups, osp.join(out_dir, 'flag_croups_' + args.method + '.json'))
			for sample in samples:
				if 'max_rouge' in sample.keys() and args.test:
					sample_fname = '_'.join([sample['article_id'],
															   sample['question_id'],
															   str(round(sample['max_rouge'], 4))]) + '.json'
				else:
					sample_fname = '_'.join([sample['article_id'],
											 sample['question_id']]) + '.json'

				write_json(sample, osp.join(samples_out_dir, sample_fname))

	total_croups = []
	total_flag_croups = []
	tokens = set()

	for raw_file in os.listdir(raw_dir):
		# path stuff

		raw_file_name = osp.splitext(raw_file)[0]
		out_dir = osp.join(gen_dir, raw_file_name)

		croups = read_json(osp.join(out_dir, 'croups_' + args.method + '.json'))
		flag_croups = read_json(osp.join(out_dir, 'flag_croups_' + args.method + '.json'))

		if not args.test:
			total_croups += croups
			total_flag_croups += flag_croups
		else:
			for ts in croups:
				tokens = tokens.union(set(ts))

	if not args.test:
		train_tv_path = osp.join(args.embed_path, 'base_token_vocab_' + args.method + '.pkl')
		train_fv_path = osp.join(args.embed_path, 'base_flag_vocab_' + args.method + '.pkl')
		train_tembed_path = osp.join(args.embed_path, 'base_token_embed_' + args.method + '.pkl')
		train_sgns_e_path = osp.join(args.embed_path, 'train_sgns_embed_' + args.method + '.pkl')
		train_sgns_v_path = osp.join(args.embed_path, 'train_sgns_vocab_' + args.method + '.pkl')
		train_fembed_path = osp.join(args.embed_path, 'base_flag_embed_' + args.method + '.pkl')

		if not osp.isfile(train_tv_path):
			token_wv, flag_wv = gen_w2v(args, total_croups, total_flag_croups)
			print('token vocab size: {}'.format(len(token_wv.vocab)))
			print('flag vocab size: {}'.format(len(flag_wv.vocab)))
			tokens = token_wv.index2word.copy()
			flags = flag_wv.index2word.copy()
			token_embeds = token_wv.vectors
			flag_embeds = flag_wv.vectors

			write_pkl(token_embeds, train_tembed_path)
			write_pkl(flag_embeds, train_fembed_path)

			write_pkl(tokens, train_tv_path)
			write_pkl(flags, train_fv_path)

			del token_wv, flag_wv, total_croups, total_flag_croups, token_embeds, flag_embeds
			sgns_vocab, sgns_embed = gen_sgns_embed(args, tokens)
			write_pkl(sgns_embed, train_sgns_e_path)
			write_pkl(sgns_vocab, train_sgns_v_path)

	else:
		tokens = list(tokens)
		test_sgns_e_path = osp.join(args.embed_path, 'test_sgns_embed_' + args.method + '.pkl')
		test_sgns_v_path = osp.join(args.embed_path, 'test_sgns_vocab_' + args.method + '.pkl')
		sgns_vocab, sgns_embed = gen_sgns_embed(args, tokens)
		write_pkl(sgns_embed, test_sgns_e_path)
		write_pkl(sgns_vocab, test_sgns_v_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Prepare Data")
	# data
	parser.add_argument('--method', type=str, default='jieba',
						choices=['jieba', 'pyltp'])
	parser.add_argument('--seq_len_train', type=int, default=500)
	parser.add_argument('--seq_len_test', type=int, default=500)

	parser.add_argument('--token_emb_dim', type=int, default=300)
	parser.add_argument('--flag_emb_dim', type=int, default=75)

	parser.add_argument('--test', type=bool, default=False)

	# misc
	working_dir = osp.dirname(osp.abspath(__file__))
	parser.add_argument('--train_raw_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/train/raw'))
	parser.add_argument('--test_raw_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/test/raw'))

	parser.add_argument('--embed_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/embed'))

	parser.add_argument('--sgns_vocab_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/embed/merge_sgns_bigram_char300.txt.bz2'))

	parser.add_argument('--train_gen_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/train/gen'))
	parser.add_argument('--test_gen_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/test/gen'))

	parser.add_argument('--jieba_big_dict', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/dict.txt.big'))
	parser.add_argument('--ltp_cws_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/ltp_data_v3.4.0/cws.model'))
	parser.add_argument('--ltp_pos_path', type=str, metavar='PATH',
						default=osp.join(working_dir, 'data/ltp_data_v3.4.0/pos.model'))

	main(parser.parse_args())
