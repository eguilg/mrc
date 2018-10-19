from metrics import RougeL, Bleu
from utils.serialization import *

if __name__ == '__main__':
	pred_ans_file = './results/submissions/slqa_plus_2_mrt_cut.json'
	gt_ans_file = './data/测试集标准答案.json'

	pred_ans = read_json(pred_ans_file)
	gt_ans = read_json(gt_ans_file)

	gt_dict = {}
	for article in gt_ans:
		for q in article['questions']:
			gt_dict[q['questions_id']] = {'question': q['question'],
										  'answer': q['answer']}

	rl = RougeL()
	bl = Bleu()
	for article in pred_ans:
		for q in article['questions']:
			pred = q['answer']
			gt = gt_dict[q['questions_id']]['answer']
			q = gt_dict[q['questions_id']]['question']
			if gt is None:
				gt = ''

			rl.add_inst(pred, gt)
			bl.add_inst(pred, gt)
			if rl.inst_scores[-1] < 0.5:
				print('q: {}\n'
					  'gt: {}\n'
					  'pred: {}'.format(q, gt, pred))
				print('-' * 80)
	print('r: {}, b: {}'.format(rl.get_score(), bl.get_score()))
