import re

stop_words = [' ', ',', '.', '?', '!', ':', ';', '-', '..', '...', '....', '.....', '......',
              '，', '。', '；', '：', '？', '！', '、'
              ]
stop_words_set = set(stop_words)

unit_set = set()
unit_list_path = './data/unit_list.txt'
with open(unit_list_path, encoding='utf-8') as f:
  lines = f.readlines()
  for line in lines:
    word, _ = line.split()
    unit_set.add(word)


def gen_ans(pos1, pos2, raw_data, key='article_tokens', post_process=True):
  pred_list = raw_data[key][pos1:pos2 + 1]
  pred_text = ''.join(pred_list)

  # is_en = True
  # for char in pred_text:
  # 	if u'\u4e00' <= char <= u'\u9fa5':
  # 		is_en = False
  # 		break
  # if is_en:
  # 	pred_text = ' '.join(pred_list)

  if not post_process:
    return pred_text

  #  后处理
  if len(pred_text) > 0 and re.match('[0-9]+', pred_text[-1]):
    if pos2 + 1 < len(raw_data[key]) and raw_data[key][pos2 + 1] in unit_set:
      pos2 += 1
      pred_text += raw_data[key][pos2]

  while len(pred_text) >= 1 and pred_text[-1] in stop_words_set:
    pred_text = pred_text[: -1].strip()
  while len(pred_text) >= 1 and pred_text[0] in stop_words_set:
    pred_text = pred_text[1:].strip()

  if pred_text.endswith('日电'):
    pred_text = pred_text[:-1]
  if pred_text.endswith('日刊'):
    pred_text = pred_text[:-1]

  return pred_text
