import os
import json
import spacy

from tqdm import tqdm

nlp = spacy.blank("en")


def word_tokenize(sent):
  doc = nlp(sent)
  return [token.text for token in doc]


def convert_idx(text, tokens):
  current = 0
  spans = []
  for token in tokens:
    current = text.find(token, current)
    if current < 0:
      print("Token {} cannot be found".format(token))
      raise Exception()
    spans.append((current, current + len(token)))
    current += len(token)
  return spans


def process_file(data_path, output_path, data_type):
  print("Generating {} examples...".format(data_type))
  examples = []
  eval_examples = {}
  with open(output_path, 'w', encoding='utf-8') as wf:
    with open(data_path, "r") as fh:
      source = json.load(fh)
      for article in tqdm(source["data"]):
        raw_title = article['title']
        raw_title = raw_title.replace("''", '" ').replace("``", '" ')
        tmp_title_tokens = word_tokenize(raw_title)
        title_tokens = []
        for token in tmp_title_tokens:
          title_tokens += token.split('_')
        title_tokens += ['<eos>']

        for para in article["paragraphs"]:
          context = para["context"].replace("''", '" ').replace("``", '" ')
          context_tokens = word_tokenize(context)

          spans = convert_idx(context, context_tokens)
          context_tokens = title_tokens + context_tokens

          for qa in para["qas"]:
            if 'is_impossible' not in qa:
              is_impossible = False
            else:
              is_impossible = qa['is_impossible']
            ques = qa["question"].replace("''", '" ').replace("``", '" ')
            ques_tokens = word_tokenize(ques)

            if is_impossible:
              # TODO: 不可回答的话就全都是-1
              answer_text = ''
              token_start, token_end = -1, -1
              item = {
                'is_impossible': is_impossible,
                'article_tokens': context_tokens,
                'question_tokens': ques_tokens,
                'answer': answer_text,
                'answer_tokens': [],

                'answer_token_start': token_start,
                'answer_token_end': token_end,

                'delta_token_starts': [],
                'delta_token_ends': [],
                'delta_rouges': []
              }
            else:
              ## FIXME: 暂时只取第一个ans, 因为ans大多是类似的
              ## FIXME: 但是这么做会影响到最后的得分, 会导致得分偏低, 因为标准的算分程序是取pred 和多个answers进行匹配的
              answer = qa['answers'][0]
              answer_text = answer["text"]
              answer_start = answer['answer_start']
              answer_end = answer_start + len(answer_text)

              answer_span = []
              for idx, span in enumerate(spans):
                if (answer_end > span[0] and answer_start < span[1]):
                  answer_span.append(idx)

              token_start, token_end = answer_span[0] + len(title_tokens), answer_span[-1] + len(title_tokens)
              answer_tokens = context_tokens[token_start:token_end + 1]
              item = {
                'is_impossible': int(is_impossible),
                'article_tokens': context_tokens,
                'question_tokens': ques_tokens,
                'answer': answer_text,
                'answer_tokens': answer_tokens,

                'answer_token_start': token_start,
                'answer_token_end': token_end,

                'delta_token_starts': [token_start],
                'delta_token_ends': [token_end],
                'delta_rouges': [1.0]
              }
            wf.write('%s\n' % (json.dumps(item, ensure_ascii=False)))

  return examples, eval_examples


if __name__ == '__main__':
  data_folder = 'squad_data'
  preprocessed_folder = os.path.join(data_folder, 'preprocessed')
  if not os.path.exists(preprocessed_folder):
    os.makedirs(preprocessed_folder)

  # train_path = os.path.join(data_folder, 'train-v2.0.json')
  # train_output_path = os.path.join(preprocessed_folder, 'train-v2.0.preprocessed.json')
  #
  # val_path = os.path.join(data_folder, 'dev-v2.0.json')
  # val_output_path = os.path.join(preprocessed_folder, 'dev-v2.0.preprocessed.json')

  train_path = os.path.join(data_folder, 'train-v1.1.json')
  train_output_path = os.path.join(preprocessed_folder, 'train-v1.1.preprocessed.json')

  val_path = os.path.join(data_folder, 'dev-v1.1.json')
  val_output_path = os.path.join(preprocessed_folder, 'dev-v1.1.preprocessed.json')

  process_file(train_path, train_output_path, data_type='train')
  process_file(val_path, val_output_path, data_type='val')
