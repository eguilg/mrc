""" conduct transformation from raw data to padded batches of inputs """

import torch
from .feature_handler.question_handler import QuestionTypeHandler
import numpy as np

def pad(seqs, max_len, pad_val):
  """ pad a batch to its max len """
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


def gen_sparse_indices(batch):
  """ generate sparse indices and values """
  idx = [[], [], []]
  val = []
  for i, sample in enumerate(batch):
    sample = sample['raw']
    select_index = list(
      filter(lambda i: sample['answer_token_start'] <= sample['delta_token_starts'][i] <= sample['answer_token_end'] or
                       sample['answer_token_start'] <= sample['delta_token_ends'][i] <= sample['answer_token_end'],
             list(range(len(sample['delta_token_starts'])))))
    idx[0].extend([i for m in range(len(select_index))])
    idx[1].extend([sample['delta_token_starts'][idx] for idx in select_index])
    idx[2].extend([sample['delta_token_ends'][idx] for idx in select_index])
    val.extend([sample['delta_rouges'][idx] for idx in select_index])
  return idx, val


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

    self.ques_type_handler = QuestionTypeHandler()

  def __call__(self, item, method):
    res = {
      'raw': item,
      'method': method,
      'c_base_idx': self.base_vocab[method].words_to_idxs(item['article_tokens']),
      'c_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['article_tokens']),
      'c_flag_idx': self.flag_vocab[method].words_to_idxs(item['article_flags']),
      'c_in_q': [1.0 if w in ''.join(item['question_tokens']) else 0.0 for w in item['article_tokens']],

      'c_mask': [0] * len(item['article_tokens']),
      'c_len': len(item['article_tokens']),

      'q_base_idx': self.base_vocab[method].words_to_idxs(item['question_tokens']),
      'q_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['question_tokens']),
      'q_flag_idx': self.flag_vocab[method].words_to_idxs(item['question_flags']),
      'q_in_c': [1.0 if w in ''.join(item['article_tokens']) else 0.0 for w in item['question_tokens']],

      'q_mask': [0] * len(item['question_tokens']),
      'q_len': len(item['question_tokens']),
    }
    if res['q_len'] == 0:
      res['q_mask'] = [0]
    if 'answer_token_start' in item:
      question_types, type_vec = self.ques_type_handler.ana_type(''.join(item['question_tokens']))
      res.update({
        'start': item['answer_token_start'],
        'end': item['answer_token_end'],
        # 'r_starts': item['delta_token_starts'],
        # 'r_ends': item['delta_token_ends'],
        # 'r_scores': item['delta_rouges'],
        'qtype_vec': type_vec,
        'c_in_a': [1.0 if w in ''.join(item['answer_tokens']) else 0.0 for w in item['article_tokens']],
        'q_in_a': [1.0 if w in ''.join(item['answer_tokens']) else 0.0 for w in item['question_tokens']],
        'ans_len': min(20, len(item['answer_tokens']))
      })
    return res

  def batchify(self, res_batch):
    c_lens = [sample['c_len'] for sample in res_batch]
    q_lens = [sample['q_len'] for sample in res_batch]

    c_max_len = 32 * (max(c_lens) // 32) + 32
    q_max_len = max(q_lens)
    m = res_batch[0]['method']
    batch = {
      'raw': [sample['raw'] for sample in res_batch],
      'method': m,
      'c_base_idx': torch.LongTensor(
        pad([sample['c_base_idx'] for sample in res_batch], c_max_len, self.base_vocab[m].pad_idx)),
      'c_sgns_idx': torch.LongTensor(
        pad([sample['c_sgns_idx'] for sample in res_batch], c_max_len, self.sgns_vocab[m].pad_idx)),
      'c_flag_idx': torch.LongTensor(
        pad([sample['c_flag_idx'] for sample in res_batch], c_max_len, self.flag_vocab[m].pad_idx)),
      'c_in_q': torch.FloatTensor(pad([sample['c_in_q'] for sample in res_batch], c_max_len, 0.0)),

      'c_mask': torch.ByteTensor(pad([sample['c_mask'] for sample in res_batch], c_max_len, 1)),
      'c_lens': torch.LongTensor(c_lens),

      'q_base_idx': torch.LongTensor(
        pad([sample['q_base_idx'] for sample in res_batch], q_max_len, self.base_vocab[m].pad_idx)),
      'q_sgns_idx': torch.LongTensor(
        pad([sample['q_sgns_idx'] for sample in res_batch], q_max_len, self.sgns_vocab[m].pad_idx)),
      'q_flag_idx': torch.LongTensor(
        pad([sample['q_flag_idx'] for sample in res_batch], q_max_len, self.flag_vocab[m].pad_idx)),
      'q_in_c': torch.FloatTensor(pad([sample['q_in_c'] for sample in res_batch], q_max_len, 0.0)),

      'q_mask': torch.ByteTensor(pad([sample['q_mask'] for sample in res_batch], q_max_len, 1)),
      'q_lens': torch.LongTensor(q_lens)
    }
    if 'start' in res_batch[0]:
      batch.update({
        'start': torch.LongTensor([sample['start'] for sample in res_batch]),
        'end': torch.LongTensor([sample['end'] for sample in res_batch]),
        'qtype_vec': torch.FloatTensor([sample['qtype_vec'] for sample in res_batch]),

        'c_in_a': torch.FloatTensor(pad([sample['c_in_a'] for sample in res_batch], c_max_len, 0.0)),
        'q_in_a': torch.FloatTensor(pad([sample['q_in_a'] for sample in res_batch], q_max_len, 0.0)),
        'ans_len': torch.LongTensor([sample['ans_len'] for sample in res_batch]),

        'delta_rouge': torch.sparse_coo_tensor(*gen_sparse_indices(res_batch),
                                               size=[len(res_batch), c_max_len, c_max_len]).to_dense(),

      })

    return batch

  @staticmethod
  def prepare_inputs(batch, rouge=False, cuda=True):
    x1_keys = [
      'c_base_idx',
      'c_sgns_idx',
      'c_flag_idx'
    ]
    x1_f_keys = [
      'c_in_q'
    ]

    x2_keys = [
      'q_base_idx',
      'q_sgns_idx',
      'q_flag_idx'
    ]
    x2_f_keys = [
      'q_in_c'
    ]

    if cuda:
      x1_list = [batch[key].cuda() for key in x1_keys]
      x1_f_list = [batch[key].cuda() for key in x1_f_keys]
      x1_mask = batch['c_mask'].cuda()
      x2_list = [batch[key].cuda() for key in x2_keys]
      x2_f_list = [batch[key].cuda() for key in x2_f_keys]
      x2_mask = batch['q_mask'].cuda()
    else:
      x1_list = [batch[key] for key in x1_keys]
      x1_f_list = [batch[key] for key in x1_f_keys]
      x1_mask = batch['c_mask']
      x2_list = [batch[key] for key in x2_keys]
      x2_f_list = [batch[key] for key in x2_f_keys]
      x2_mask = batch['q_mask']

    method = batch['method']

    inputs = [x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method]
    if 'start' in batch:
      if cuda:
        y_start = batch['start'].cuda()
        y_end = batch['end'].cuda()
        qtype_vec = batch['qtype_vec'].cuda()
        c_in_a = batch['c_in_a'].cuda()
        q_in_a = batch['q_in_a'].cuda()
        ans_len = batch['ans_len'].cuda()
        if rouge:
          delta_rouge = batch['delta_rouge'].cuda()
        else:
          delta_rouge = None
      else:
        y_start = batch['start']
        y_end = batch['end']
        qtype_vec = batch['qtype_vec']
        c_in_a = batch['c_in_a']
        q_in_a = batch['q_in_a']
        ans_len = batch['ans_len']
        if rouge:
          delta_rouge = batch['delta_rouge']
        else:
          delta_rouge = None
      targets = (y_start, y_end, (qtype_vec, c_in_a, q_in_a, ans_len, delta_rouge))
      return inputs, targets
    else:
      return inputs, None


class MaiDetectionTransform(MaiIndexTransform):
  def __init__(self, jieba_base_v=None, jieba_sgns_v=None, jieba_flag_v=None, pyltp_base_v=None, pyltp_sgns_v=None,
               pyltp_flag_v=None, B=1, S=16):
    super().__init__(jieba_base_v, jieba_sgns_v, jieba_flag_v, pyltp_base_v, pyltp_sgns_v, pyltp_flag_v)
    self.B = B
    self.S = S

  def batchify(self, res_batch):
    c_lens = [sample['c_len'] for sample in res_batch]
    q_lens = [sample['q_len'] for sample in res_batch]

    c_max_len = 512
    block_width = c_max_len // self.S

    q_max_len = max(q_lens)
    m = res_batch[0]['method']
    batch = {
      'raw': [sample['raw'] for sample in res_batch],
      'method': m,
      'c_base_idx': torch.LongTensor(
        pad([sample['c_base_idx'] for sample in res_batch], c_max_len, self.base_vocab[m].pad_idx)),
      'c_sgns_idx': torch.LongTensor(
        pad([sample['c_sgns_idx'] for sample in res_batch], c_max_len, self.sgns_vocab[m].pad_idx)),
      'c_flag_idx': torch.LongTensor(
        pad([sample['c_flag_idx'] for sample in res_batch], c_max_len, self.flag_vocab[m].pad_idx)),
      'c_in_q': torch.FloatTensor(pad([sample['c_in_q'] for sample in res_batch], c_max_len, 0.0)),

      'c_mask': torch.ByteTensor(pad([sample['c_mask'] for sample in res_batch], c_max_len, 1)),
      'c_lens': torch.LongTensor(c_lens),

      'q_base_idx': torch.LongTensor(
        pad([sample['q_base_idx'] for sample in res_batch], q_max_len, self.base_vocab[m].pad_idx)),
      'q_sgns_idx': torch.LongTensor(
        pad([sample['q_sgns_idx'] for sample in res_batch], q_max_len, self.sgns_vocab[m].pad_idx)),
      'q_flag_idx': torch.LongTensor(
        pad([sample['q_flag_idx'] for sample in res_batch], q_max_len, self.flag_vocab[m].pad_idx)),
      'q_in_c': torch.FloatTensor(pad([sample['q_in_c'] for sample in res_batch], q_max_len, 0.0)),

      'q_mask': torch.ByteTensor(pad([sample['q_mask'] for sample in res_batch], q_max_len, 1)),
      'q_lens': torch.LongTensor(q_lens)
    }
    if 'start' in res_batch[0]:
      if self.B <= 1:
        # FIXME: 更新成适配多个B的
        for sample in res_batch:
          sample['scores']=[0.0 for _ in range(self.S)]
          sample['widths']=[0.0 for _ in range(self.S)]
          sample['block_centers']=[0.0 for _ in range(self.S)]

          width = sample['end'] - sample['start']
          width /= c_max_len

          block_center= (sample['end'] + sample['start']) // 2
          idx=block_center//block_width

          block_center %= block_width
          block_center /= block_width

          score = 1.0

          sample['scores'][idx] = score
          sample['widths'][idx] = width
          sample['block_centers'][idx] = block_center

      batch.update({
        'start': torch.LongTensor([sample['start'] for sample in res_batch]),
        'end': torch.LongTensor([sample['end'] for sample in res_batch]),
        'qtype_vec': torch.FloatTensor([sample['qtype_vec'] for sample in res_batch]),

        'c_in_a': torch.FloatTensor(pad([sample['c_in_a'] for sample in res_batch], c_max_len, 0.0)),
        'q_in_a': torch.FloatTensor(pad([sample['q_in_a'] for sample in res_batch], q_max_len, 0.0)),
        'ans_len': torch.LongTensor([sample['ans_len'] for sample in res_batch]),

        'delta_rouge': torch.sparse_coo_tensor(*gen_sparse_indices(res_batch),
                                               size=[len(res_batch), c_max_len, c_max_len]).to_dense(),

        'widths': torch.FloatTensor([sample['widths'] for sample in res_batch]),
        'block_centers': torch.FloatTensor([sample['block_centers'] for sample in res_batch]),
        'scores': torch.FloatTensor([sample['scores'] for sample in res_batch]),
      })

    return batch

  @staticmethod
  def prepare_inputs(batch, rouge=False, cuda=True,obj_eval=False):
    x1_keys = [
      'c_base_idx',
      'c_sgns_idx',
      'c_flag_idx'
    ]
    x1_f_keys = [
      'c_in_q'
    ]

    x2_keys = [
      'q_base_idx',
      'q_sgns_idx',
      'q_flag_idx'
    ]
    x2_f_keys = [
      'q_in_c'
    ]

    if cuda:
      x1_list = [batch[key].cuda() for key in x1_keys]
      x1_f_list = [batch[key].cuda() for key in x1_f_keys]
      x1_mask = batch['c_mask'].cuda()
      x2_list = [batch[key].cuda() for key in x2_keys]
      x2_f_list = [batch[key].cuda() for key in x2_f_keys]
      x2_mask = batch['q_mask'].cuda()
    else:
      x1_list = [batch[key] for key in x1_keys]
      x1_f_list = [batch[key] for key in x1_f_keys]
      x1_mask = batch['c_mask']
      x2_list = [batch[key] for key in x2_keys]
      x2_f_list = [batch[key] for key in x2_f_keys]
      x2_mask = batch['q_mask']

    method = batch['method']

    inputs = [x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method]
    if 'start' in batch:
      if cuda:
        y_start = batch['start'].cuda()
        y_end = batch['end'].cuda()

        y_widths = batch['widths'].cuda()
        y_centers = batch['block_centers'].cuda()
        y_scores = batch['scores'].cuda()

        qtype_vec = batch['qtype_vec'].cuda()
        c_in_a = batch['c_in_a'].cuda()
        q_in_a = batch['q_in_a'].cuda()
        ans_len = batch['ans_len'].cuda()
        if rouge:
          delta_rouge = batch['delta_rouge'].cuda()
        else:
          delta_rouge = None
      else:
        y_start = batch['start']
        y_end = batch['end']

        y_widths = batch['width']
        y_centers = batch['block_center']
        y_scores = batch['score']
        qtype_vec = batch['qtype_vec']
        c_in_a = batch['c_in_a']
        q_in_a = batch['q_in_a']
        ans_len = batch['ans_len']
        if rouge:
          delta_rouge = batch['delta_rouge']
        else:
          delta_rouge = None
      if obj_eval:
        targets = (y_start,y_end,y_widths, y_centers, y_scores, (qtype_vec, c_in_a, q_in_a, ans_len, delta_rouge))
      else:
        targets = (y_widths, y_centers, y_scores, (qtype_vec, c_in_a, q_in_a, ans_len, delta_rouge))
      return inputs, targets
    else:
      return inputs, None

class TitleSummTransform(object):
  base_vocab = {}
  sgns_vocab = {}
  flag_vocab = {}

  def __init__(self, jieba_base_v=None, jieba_sgns_v=None, jieba_flag_v=None):
    self.base_vocab['jieba'] = jieba_base_v
    self.sgns_vocab['jieba'] = jieba_sgns_v
    self.flag_vocab['jieba'] = jieba_flag_v


  def __call__(self, item, method):
    res = {
      'raw': item,
      'method': method,
      'c_base_idx': self.base_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'c_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'c_flag_idx': self.flag_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'c_in_q': [1.0 if w in ''.join(item['ori_title_tokens']) else 0.0 for w in item['ori_title_tokens']],

      'c_mask': [0] * len(item['ori_title_tokens']),
      'c_len': len(item['ori_title_tokens']),

      'q_base_idx': self.base_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'q_sgns_idx': self.sgns_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'q_flag_idx': self.flag_vocab[method].words_to_idxs(item['ori_title_tokens']),
      'q_in_c': [1.0 if w in ''.join(item['ori_title_tokens']) else 0.0 for w in item['ori_title_tokens']],

      'q_mask': [0] * len(item['ori_title_tokens']),
      'q_len': len(item['ori_title_tokens']),
    }
    if res['q_len'] == 0:
      res['q_mask'] = [0]
    if 'answer_token_start' in item:
      fake_qtype_vec=np.zeros(10)
      fake_qtype_vec[0]=1.0
      res.update({
        'start': item['answer_token_start'],
        'end': item['answer_token_end'],
        # 'r_starts': item['delta_token_starts'],
        # 'r_ends': item['delta_token_ends'],
        # 'r_scores': item['delta_rouges'],
        'qtype_vec': fake_qtype_vec,
        'c_in_a': [1.0 if w in ''.join(item['short_title_tokens']) else 0.0 for w in item['ori_title_tokens']],
        'q_in_a': [1.0 if w in ''.join(item['short_title_tokens']) else 0.0 for w in item['ori_title_tokens']],
        'ans_len': min(20, len(item['short_title_tokens']))
      })
    return res

  def batchify(self, res_batch):
    c_lens = [sample['c_len'] for sample in res_batch]
    q_lens = [sample['q_len'] for sample in res_batch]

    c_max_len = 32 * (max(c_lens) // 32) + 32
    q_max_len = max(q_lens)
    m = res_batch[0]['method']
    batch = {
      'raw': [sample['raw'] for sample in res_batch],
      'method': m,
      'c_base_idx': torch.LongTensor(
        pad([sample['c_base_idx'] for sample in res_batch], c_max_len, self.base_vocab[m].pad_idx)),
      'c_sgns_idx': torch.LongTensor(
        pad([sample['c_sgns_idx'] for sample in res_batch], c_max_len, self.sgns_vocab[m].pad_idx)),
      'c_flag_idx': torch.LongTensor(
        pad([sample['c_flag_idx'] for sample in res_batch], c_max_len, self.flag_vocab[m].pad_idx)),
      'c_in_q': torch.FloatTensor(pad([sample['c_in_q'] for sample in res_batch], c_max_len, 0.0)),

      'c_mask': torch.ByteTensor(pad([sample['c_mask'] for sample in res_batch], c_max_len, 1)),
      'c_lens': torch.LongTensor(c_lens),

      'q_base_idx': torch.LongTensor(
        pad([sample['q_base_idx'] for sample in res_batch], q_max_len, self.base_vocab[m].pad_idx)),
      'q_sgns_idx': torch.LongTensor(
        pad([sample['q_sgns_idx'] for sample in res_batch], q_max_len, self.sgns_vocab[m].pad_idx)),
      'q_flag_idx': torch.LongTensor(
        pad([sample['q_flag_idx'] for sample in res_batch], q_max_len, self.flag_vocab[m].pad_idx)),
      'q_in_c': torch.FloatTensor(pad([sample['q_in_c'] for sample in res_batch], q_max_len, 0.0)),

      'q_mask': torch.ByteTensor(pad([sample['q_mask'] for sample in res_batch], q_max_len, 1)),
      'q_lens': torch.LongTensor(q_lens)
    }
    if 'start' in res_batch[0]:
      batch.update({
        'start': torch.LongTensor([sample['start'] for sample in res_batch]),
        'end': torch.LongTensor([sample['end'] for sample in res_batch]),
        'qtype_vec': torch.FloatTensor([sample['qtype_vec'] for sample in res_batch]),

        'c_in_a': torch.FloatTensor(pad([sample['c_in_a'] for sample in res_batch], c_max_len, 0.0)),
        'q_in_a': torch.FloatTensor(pad([sample['q_in_a'] for sample in res_batch], q_max_len, 0.0)),
        'ans_len': torch.LongTensor([sample['ans_len'] for sample in res_batch]),

        'delta_rouge': torch.sparse_coo_tensor(*gen_sparse_indices(res_batch),
                                               size=[len(res_batch), c_max_len, c_max_len]).to_dense(),

      })

    return batch

  @staticmethod
  def prepare_inputs(batch, rouge=False, cuda=True):
    x1_keys = [
      'c_base_idx',
      'c_sgns_idx',
      'c_flag_idx'
    ]
    x1_f_keys = [
      'c_in_q'
    ]

    x2_keys = [
      'q_base_idx',
      'q_sgns_idx',
      'q_flag_idx'
    ]
    x2_f_keys = [
      'q_in_c'
    ]

    if cuda:
      x1_list = [batch[key].cuda() for key in x1_keys]
      x1_f_list = [batch[key].cuda() for key in x1_f_keys]
      x1_mask = batch['c_mask'].cuda()
      x2_list = [batch[key].cuda() for key in x2_keys]
      x2_f_list = [batch[key].cuda() for key in x2_f_keys]
      x2_mask = batch['q_mask'].cuda()
    else:
      x1_list = [batch[key] for key in x1_keys]
      x1_f_list = [batch[key] for key in x1_f_keys]
      x1_mask = batch['c_mask']
      x2_list = [batch[key] for key in x2_keys]
      x2_f_list = [batch[key] for key in x2_f_keys]
      x2_mask = batch['q_mask']

    method = batch['method']

    inputs = [x1_list, x1_f_list, x1_mask, x2_list, x2_f_list, x2_mask, method]
    if 'start' in batch:
      if cuda:
        y_start = batch['start'].cuda()
        y_end = batch['end'].cuda()
        qtype_vec = batch['qtype_vec'].cuda()
        c_in_a = batch['c_in_a'].cuda()
        q_in_a = batch['q_in_a'].cuda()
        ans_len = batch['ans_len'].cuda()
        if rouge:
          delta_rouge = batch['delta_rouge'].cuda()
        else:
          delta_rouge = None
      else:
        y_start = batch['start']
        y_end = batch['end']
        qtype_vec = batch['qtype_vec']
        c_in_a = batch['c_in_a']
        q_in_a = batch['q_in_a']
        ans_len = batch['ans_len']
        if rouge:
          delta_rouge = batch['delta_rouge']
        else:
          delta_rouge = None
      targets = (y_start, y_end, (qtype_vec, c_in_a, q_in_a, ans_len, delta_rouge))
      return inputs, targets
    else:
      return inputs, None
