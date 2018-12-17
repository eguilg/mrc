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

rouge_eval = RougeL()


def read_data(file_path):
  data_set = []
  with open(file_path, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
      data = line.strip().split('\t\t')
      data_set.append(data)

  return data_set


def output_spans(ori_title, short_title):
  ori_length = len(ori_title)
  short_length = len(short_title)

  lengths = np.zeros(((ori_length + 1), (short_length + 1)), dtype=np.int)
  for i in range(1, ori_length + 1):
    for j in range(1, short_length + 1):
      if ori_title[i - 1] == short_title[j - 1]:
        lengths[i][j] = lengths[i - 1][j - 1] + 1
      else:
        lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

  ## 记录下lcs里的pos, 用于制作span
  i = ori_length - 1
  j = short_length - 1
  positions = []
  while i >= 0 and j >= 0:
    if ori_title[i] == short_title[j]:
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
    span_text += ori_title[span[0]:span[1] + 1]

  return spans, span_text


def append_extra_rouges(ori_title, short_title, starts, ends, rouge_scores, window_size=3):
  fixed_pos = set([(start, end) for start, end in zip(starts, ends)])
  extra_pos_dict = {}

  for start, end, score in zip(starts, ends, rouge_scores):
    for i in range(max(0, start - window_size), min(end, start + window_size)):
      for j in range(max(start, end - window_size), min(len(ori_title), start + window_size)):
        if (i, j) in fixed_pos:
          continue

        cur_text = ori_title[i:j + 1]
        if len(cur_text) == 0:
          continue
        rouge, _, _ = rouge_eval.calc_score(cur_text, short_title)
        rouge /= len(cur_text)
        if rouge == 1.0:
          continue

        if (i, j) not in extra_pos_dict:
          extra_pos_dict[(i, j)] = []
        extra_pos_dict[(i, j)].append(rouge)

  for (i, j) in extra_pos_dict:
    rouges = extra_pos_dict[(i, j)]
    rouge = sum(rouges) / len(rouges)

    starts.append(i)
    ends.append(j)
    rouge_scores.append(rouge)

  return starts, ends, rouge_scores


def preproce(raw_path, save_path, extra_rouge):
  with open(save_path, 'w', encoding='utf-8') as wf:
    val_data = read_data(raw_path)
    for data in tqdm(val_data):
      ori_title = data[0]
      short_title = data[1]
      spans, span_text = output_spans(ori_title, short_title)

      starts = [span[0] for span in spans]
      ends = [span[1] for span in spans]
      rouge_scores = [1.0 - random.random() / 10 for _ in range(len(spans))]

      if extra_rouge:
        starts, ends, rouge_scores = append_extra_rouges(ori_title, short_title, starts, ends, rouge_scores)

      item = {
        'ori_title_tokens': list(ori_title),
        'short_title_tokens': list(short_title),
        'answer': short_title,

        ## FIXME: start & end 应该没有用, 要删掉
        'answer_token_start': 1,
        'answer_token_end': 1,

        'delta_token_starts': starts,
        'delta_token_ends': ends,
        'delta_rouges': rouge_scores

      }
      wf.write('%s\n' % (json.dumps(item, ensure_ascii=False)))


if __name__ == '__main__':
  train_path = 'title_data/train.txt'
  val_path = 'title_data/val.txt'
  extra_rouge = True
  preproce(val_path, 'title_data/preprocessed/val.preprocessed.%sjson' % ('extra_rouge.' if extra_rouge else ''),
           extra_rouge=extra_rouge)
  preproce(train_path, 'title_data/preprocessed/train.preprocessed.%sjson' % ('extra_rouge.' if extra_rouge else ''),
           extra_rouge=extra_rouge)
