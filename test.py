import multiprocessing as mp

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import postprocess
from config import config
from dataset import (MaiDirDataSource, MaiDirDataset, Vocab,
					 MaiIndexTransform, MethodBasedBatchSampler)
from models.rc_model import RCModel
from utils.osutils import *
from utils.serialization import *

bidaf1 = config.bi_daf_1()
bidaf2 = config.bi_daf_2()
bidaf3 = config.bi_daf_3()

rnet1 = config.r_net_1()
rnet2 = config.r_net_2()
rnet3 = config.r_net_3()

mreader1 = config.m_reader_1()
mreader2 = config.m_reader_2()
mreader3 = config.m_reader_3()

slqa1 = config.slqa_1()
slqa2 = config.slqa_2()
slqa3 = config.slqa_3()

slqa_plus1 = config.slqa_plus_1()
slqa_plus2 = config.slqa_plus_2()
slqa_plus3 = config.slqa_plus_3()


# cur_cfg = bidaf2
# cur_cfg = bidaf3

# cur_cfg = rnet2
# cur_cfg = rnet3

# cur_cfg = mreader2
# cur_cfg = mreader3

# cur_cfg = slqa1
# cur_cfg = slqa2
# cur_cfg = slqa3

cur_cfg = slqa_plus1
# cur_cfg = slqa_plus2
# cur_cfg = slqa_plus3

use_mrt = False
switch = False
use_data1 = False
cut_ans = True

testset_roots = [
		'./data/test/gen/test_question/samples_jieba500',
		# './data/train/gen/train_1/samples_jieba500'
	]
testset_raw_path = './data/test/raw/test_question.json'

if __name__ == '__main__':
	print(cur_cfg.model_params)
	model_dir = './data/models/'

	range_result_dir = osp.join('./results/range_prob', osp.splitext(osp.basename(testset_raw_path))[0])
	submission_dir = osp.join('./results/submissions', osp.splitext(osp.basename(testset_raw_path))[0])

	mkdir_if_missing(range_result_dir)
	mkdir_if_missing(submission_dir)

	model_name = cur_cfg.name
	if use_mrt:
		model_name += '_mrt'
	if switch:
		model_name += '_switch'
	if use_data1:
		model_name += '_full_data'
	model_path = os.path.join(model_dir, model_name + '.state')

	range_result_path = os.path.join(range_result_dir, model_name + '.pkl')
	if cut_ans:
		submission_path = os.path.join(submission_dir, model_name + '_cut.json')
	else:
		submission_path = os.path.join(submission_dir, model_name + '.json')

	jieba_base_v = Vocab('./data/embed/base_token_vocab_jieba.pkl',
						 './data/embed/base_token_embed_jieba.pkl')
	jieba_sgns_v = Vocab('./data/embed/test_sgns_vocab_jieba.pkl',
						 './data/embed/test_sgns_embed_jieba.pkl')
	jieba_sgns_train_v = Vocab('./data/embed/train_sgns_vocab_jieba.pkl',
							   './data/embed/train_sgns_embed_jieba.pkl')
	jieba_flag_v = Vocab('./data/embed/base_flag_vocab_jieba.pkl',
						 './data/embed/base_flag_embed_jieba.pkl')

	embed_lists_test = {
		'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
		'pyltp': []
	}

	embed_lists_train = {
		'jieba': [jieba_base_v.embeddings, jieba_sgns_train_v.embeddings, jieba_flag_v.embeddings],
		'pyltp': []
	}
	if switch:
		pyltp_base_v = Vocab('./data/embed/base_token_vocab_pyltp.pkl',
							 './data/embed/base_token_embed_pyltp.pkl')
		pyltp_sgns_train_v = Vocab('./data/embed/train_sgns_vocab_pyltp.pkl',
								   './data/embed/train_sgns_embed_pyltp.pkl')
		pyltp_flag_v = Vocab('./data/embed/base_flag_vocab_pyltp.pkl',
							 './data/embed/base_flag_embed_pyltp.pkl')

		embed_lists_train = {
			'jieba': [jieba_base_v.embeddings, jieba_sgns_train_v.embeddings, jieba_flag_v.embeddings],
			'pyltp': [pyltp_base_v.embeddings, pyltp_sgns_train_v.embeddings, pyltp_flag_v.embeddings]
		}

	transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v)

	test_data_source = MaiDirDataSource(testset_roots)

	test_loader = DataLoader(
		dataset=MaiDirDataset(test_data_source.data, transform),
		batch_sampler=MethodBasedBatchSampler(test_data_source.data, batch_size=32, shuffle=False),
		num_workers=mp.cpu_count(),
		collate_fn=transform.batchify
	)

	model_params = cur_cfg.model_params

	model = RCModel(model_params, embed_lists_train)#, normalize=(not use_mrt))
	print('loading model, ', model_path)

	state = torch.load(model_path)
	best_score = state['best_score']
	best_epoch = state['best_epoch']
	best_step = state['best_step']
	print('best_epoch:%2d, best_step:%5d, best_score:%.4f' %
		  (best_epoch, best_step, best_score))
	model.load_state_dict(state['best_model_state'])
	model.reset_embeddings(embed_lists_test)
	model = model.cuda()
	model.eval()

	answer_prob_dict = {}
	answer_dict = {}
	print('going through model...')
	for batch in tqdm(test_loader):
		inputs, _ = transform.prepare_inputs(batch)
		s_prob, e_prob, ans_len_logits = model(*inputs)
		s_prob = s_prob.detach().cpu()
		e_prob = e_prob.detach().cpu()
		ans_len_prob = F.softmax(ans_len_logits, dim=-1).detach().cpu()

		if cut_ans:
			batch_pos1, batch_pos2, confidence = model.decode_cut_ans(s_prob, e_prob, ans_len_prob)
		else:
			batch_pos1, batch_pos2, confidence = model.decode(s_prob, e_prob)

		batch_ans_len = ans_len_prob.numpy()
		batch_prob1 = s_prob.numpy()
		batch_prob2 = e_prob.numpy()
		for sample, prob1, prob2, pos1, pos2, prob_ans_len in zip(batch['raw'], batch_prob1, batch_prob2, batch_pos1,
																  batch_pos2, batch_ans_len):
			answer_prob_dict[sample['question_id']] = {'prob1': prob1,
													   'prob2': prob2,
													   'ans_len_prob': prob_ans_len,
													   'article_id': sample['article_id']}

			answer_dict[sample['question_id']] = postprocess.gen_ans(pos1, pos2, sample)

	write_pkl(answer_prob_dict, range_result_path)
	del answer_prob_dict

	sub = read_json(testset_raw_path)
	print('generating submission...')
	for article in tqdm(sub):
		for q in article['questions']:
			q['answer'] = answer_dict[q['questions_id']]

			if q['question'] == '':
				q['answer'] = ''
			if q['question'] == article['article_title']:
				q['answer'] = article['article_title']

			q.pop('question')
		article.pop('article_type')
		article.pop('article_title')
		article.pop('article_content')

	write_json(sub, submission_path)
