"""
delta_token_starts
delta_token_ends
delta_rouges
"""
import os
import random

import json
import numpy as np
from tqdm import tqdm
from metrics.rouge import RougeL
from title_summ_tokenizer import TitleSummBertTokenizer

rouge_eval = RougeL()


def read_data(file_path):
  data_set = []
  with open(file_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      data = line.strip().split('\t\t')
      data_set.append(data)

  return data_set


def remove_prefix(token, prefix='##'):
  if token.startswith(prefix):
    return token[len(prefix):]
  return token


def output_spans(ori_title_tokens, short_title_tokens):
  ori_title_tokens_ = [remove_prefix(token) for token in ori_title_tokens]
  short_title_tokens_ = [remove_prefix(token) for token in short_title_tokens]

  ori_length = len(ori_title_tokens_)
  short_length = len(short_title_tokens_)

  lengths = np.zeros(((ori_length + 1), (short_length + 1)), dtype=np.int)
  for i in range(1, ori_length + 1):
    for j in range(1, short_length + 1):
      if ori_title_tokens_[i - 1] == short_title_tokens_[j - 1]:
        lengths[i][j] = lengths[i - 1][j - 1] + 1
      else:
        lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

  ## 记录下lcs里的pos, 用于制作span
  i = ori_length - 1
  j = short_length - 1
  positions = []
  while i >= 0 and j >= 0:
    if ori_title_tokens_[i] == short_title_tokens_[j]:
      positions.append(i)
      i -= 1
      j -= 1
    elif lengths[i - 1][j] <= lengths[i][j - 1]:
      j -= 1
    else:
      i -= 1

  positions = positions[::-1]
  positions += [-1]  # -1 相当于归并排序里的底座
  spans = []
  start = positions[0]
  for i, pos in enumerate(positions[:-1]):
    if pos + 1 == positions[i + 1]:
      continue
    else:
      spans.append((start, pos))
      start = positions[i + 1]

  span_text = ''
  for span in spans:
    span_text += ''.join(ori_title_tokens_[span[0]:span[1] + 1])

  return spans, span_text


def append_extra_rouges(ori_title_tokens, short_title_tokens, starts, ends, rouge_scores, window_size=3):
  ori_title_tokens_ = [remove_prefix(token) for token in ori_title_tokens]
  short_title_tokens_ = [remove_prefix(token) for token in short_title_tokens]

  fixed_pos = set([(start, end) for start, end in zip(starts, ends)])
  extra_pos_dict = {}

  short_title_text = ''.join(short_title_tokens_)
  for start, end, score in zip(starts, ends, rouge_scores):
    for i in range(max(0, start - window_size), min(end, start + window_size)):
      for j in range(max(start, end - window_size), min(len(ori_title_tokens_), start + window_size)):
        if (i, j) in fixed_pos:
          continue
        if (i, j) in extra_pos_dict:
          continue

        cur_text = ''.join(ori_title_tokens_[i:j + 1])
        if len(cur_text) == 0:
          continue
        rouge, _, _ = rouge_eval.calc_score(cur_text, short_title_text)
        rouge /= len(cur_text)
        if rouge == 1.0:
          continue

        extra_pos_dict[(i, j)] = rouge

  for (i, j) in extra_pos_dict:
    rouge = extra_pos_dict[(i, j)]

    starts.append(i)
    ends.append(j)
    rouge_scores.append(rouge)

  return starts, ends, rouge_scores


def preproce(raw_path, save_path, extra_rouge, bert_tokenizer=None):
  rouges = []
  with open(save_path, 'w', encoding='utf-8') as wf:
    val_data = read_data(raw_path)
    for data in tqdm(val_data):
      ori_title = data[0]
      short_title = data[1]
      if bert_tokenizer is not None:
        ori_title_tokens = bert_tokenizer.tokenize(ori_title)
        ori_title_tokens = [token for token in ori_title_tokens if token != '[UNK]']

        short_title_tokens = bert_tokenizer.tokenize(short_title)
        short_title_tokens = [token for token in short_title_tokens if token != '[UNK]']
      else:
        ori_title_tokens = list(ori_title)
        short_title_tokens = list(short_title)

      spans, span_text = output_spans(ori_title_tokens, short_title_tokens)
      rouges.append(rouge_eval.calc_score(span_text, short_title))

      starts = [span[0] for span in spans]
      ends = [span[1] for span in spans]
      rouge_scores = [1.0]

      if extra_rouge:
        starts, ends, rouge_scores = append_extra_rouges(ori_title_tokens, short_title_tokens, starts, ends,
                                                         rouge_scores)

      item = {
        'ori_title_tokens': ori_title_tokens,
        'short_title_tokens': short_title_tokens,
        'answer': short_title,

        ## FIXME: start & end 应该没有用, 要删掉
        'answer_token_start': 1,
        'answer_token_end': 1,

        'delta_token_starts': starts,
        'delta_token_ends': ends,
        'delta_rouges': rouge_scores

      }
      wf.write('%s\n' % (json.dumps(item, ensure_ascii=False)))
  print(np.mean(rouges))


if __name__ == '__main__':
  extra_rouge = True
  # bert_mode=None
  bert_mode = 'bert-base-multilingual-cased'

  train_path = 'title_data/train.txt'
  val_path = 'title_data/val.txt'

  val_save_path = 'title_data/preprocessed/val.preprocessed.%s%sjson' % (
    'extra_rouge.' if extra_rouge else '', bert_mode + '.' if bert_mode else '')
  train_save_path = 'title_data/preprocessed/train.preprocessed.%s%sjson' % (
    'extra_rouge.' if extra_rouge else '', bert_mode + '.' if bert_mode else '')
  
  bert_tokenizer = None
  if bert_mode is not None:
    bert_tokenizer = TitleSummBertTokenizer(os.path.join('title_data', 'vocab', bert_mode + '.vocab'))

  preproce(val_path, val_save_path, extra_rouge=extra_rouge, bert_tokenizer=bert_tokenizer)
  preproce(train_path, train_save_path, extra_rouge=extra_rouge, bert_tokenizer=bert_tokenizer)
