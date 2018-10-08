import multiprocessing as mp

from torch.utils.data import DataLoader
from tqdm import tqdm

import postprocess
from config import config
from dataset import (MaiDirDataSource, MaiDirDataset, Vocab,
					 MaiIndexTransform, MethodBasedBatchSampler)
from models.rc_model import RCModel
from utils.osutils import *
from utils.serialization import *

rnet1 = config.r_net_1()
rnet2 = config.r_net_2()
mreader1 = config.m_reader_1()
mreader2 = config.m_reader_2()


if __name__ == '__main__':
	cur_cfg = mreader2
	model_dir = './data/models/'
	range_result_dir = './results/range_prob'
	submission_dir = './results/submissions'
	mkdir_if_missing(range_result_dir)
	mkdir_if_missing(submission_dir)
	model_path = os.path.join(model_dir, cur_cfg.name+'.state')
	range_result_path = os.path.join(range_result_dir, cur_cfg.name+'.pkl')
	submission_path = os.path.join(submission_dir, cur_cfg.name+'.json')

	testset_roots = [
		'./data/test/gen/test_question/samples_jieba500',
	]
	testset_raw_path = './data/test/raw/test_question.json'

	jieba_base_v = Vocab('./data/embed/base_token_vocab_jieba.pkl',
						 './data/embed/base_token_embed_jieba.pkl')
	jieba_sgns_v = Vocab('./data/embed/test_sgns_vocab_jieba.pkl',
						 './data/embed/test_sgns_embed_jieba.pkl')
	jieba_sgns_train_v = Vocab('./data/embed/train_sgns_vocab_jieba.pkl',
							   './data/embed/train_sgns_embed_jieba.pkl')
	jieba_flag_v = Vocab('./data/embed/base_flag_vocab_jieba.pkl',
						 './data/embed/base_flag_embed_jieba.pkl')

	pyltp_base_v = Vocab('./data/embed/base_token_vocab_pyltp.pkl',
						 './data/embed/base_token_embed_pyltp.pkl')
	pyltp_sgns_train_v = Vocab('./data/embed/train_sgns_vocab_pyltp.pkl',
							   './data/embed/train_sgns_embed_pyltp.pkl')
	pyltp_flag_v = Vocab('./data/embed/base_flag_vocab_pyltp.pkl',
						 './data/embed/base_flag_embed_pyltp.pkl')

	transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v)
	test_data_source = MaiDirDataSource(testset_roots)

	test_loader = DataLoader(
		dataset=MaiDirDataset(test_data_source.data, transform),
		batch_sampler=MethodBasedBatchSampler(test_data_source.data, batch_size=32, shuffle=False),
		num_workers=mp.cpu_count(),
		collate_fn=transform.batchify
	)

	embed_lists_test = {
		'jieba': [jieba_base_v.embeddings, jieba_sgns_v.embeddings, jieba_flag_v.embeddings],
		'pyltp': []
	}

	embed_lists_train = {
		'jieba': [jieba_base_v.embeddings, jieba_sgns_train_v.embeddings, jieba_flag_v.embeddings],
		'pyltp': [pyltp_base_v.embeddings, pyltp_sgns_train_v.embeddings, pyltp_flag_v.embeddings]
	}

	model_params = cur_cfg.model_params

	model = RCModel(model_params, embed_lists_train)
	print('loading model, ', model_path)

	state = torch.load(model_path)
	best_loss = state['best_loss']
	best_epoch = state['best_epoch']
	best_step = state['best_step']
	print('best_epoch:%2d, best_step:%5d, best_loss:%.4f' %
		  (best_epoch, best_step, best_loss))
	model.load_state_dict(state['best_model_state'])
	model.reset_embeddings(embed_lists_test)
	model = model.cuda()
	model.eval()

	answer_prob_dict = {}
	answer_pos_dict = {}
	print('going through model...')
	for batch in tqdm(test_loader):
		inputs, _ = transform.prepare_inputs(batch)
		s_prob, e_prob = model(*inputs)
		s_prob = s_prob.detach().cpu()
		e_prob = e_prob.detach().cpu()
		batch_pos1, batch_pos2, confidence = model.decode(s_prob, e_prob)
		batch_prob1 = s_prob.numpy()
		batch_prob2 = e_prob.numpy()
		for sample, prob1, prob2, pos1, pos2 in zip(batch['raw'], batch_prob1, batch_prob2, batch_pos1, batch_pos2):
			answer_prob_dict[sample['question_id']] = {'prob1': prob1,
													   'prob2': prob2,

													   'article_id': sample['article_id']}

			answer_pos_dict[sample['question_id']] = {'pos1': pos1,
													  'pos2': pos2,
													  'raw': sample,
													  'article_id': sample['article_id']}
	write_pkl(answer_prob_dict, range_result_path)

	sub = read_json(testset_raw_path)
	print('generating submission...')
	for article in tqdm(sub):
		for q in article['questions']:
			res = answer_pos_dict[q['questions_id']]
			q['answer'] = postprocess.gen_ans(res['pos1'], res['pos2'], res['raw'])
			sample = res['raw']
			if len(sample['question_tokens']) == 0:
				q['answer'] = ''
			if article['article_content'] == '' or article['article_title'] == q['question']:
				q['answer'] = article['article_title']
			if len(sample['article_tokens']) == 0:
				q['answer'] = ''
			q.pop('question')
		article.pop('article_type')
		article.pop('article_title')
		article.pop('article_content')

	write_json(sub, submission_path)
