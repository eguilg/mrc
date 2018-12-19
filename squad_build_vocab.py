import json
import pickle
import numpy as np


def pickle_dump(data, file_path):
  f_write = open(file_path, 'wb')
  pickle.dump(data, f_write, False)


word_freq_dict = {}

with open('squad_data/preprocessed/train-v1.1.preprocessed.json', encoding='utf-8') as f:
  lines = f.readlines()
  for line in lines:
    sample = json.loads(line)
    for word in sample['question_tokens']:
      if word not in word_freq_dict:
        word_freq_dict[word] = 0
      word_freq_dict[word] += 1

    for word in sample['article_tokens']:
      if word not in word_freq_dict:
        word_freq_dict[word] = 0
      word_freq_dict[word] += 1

print(len(word_freq_dict))

glove_embedding_path = 'squad_data/vocab/glove.6B.100d.txt'
glove_dict = {}
with open(glove_embedding_path, encoding='utf-8') as f:
  for line in f.readlines():
    t = line.split()
    word, embedding = t[0], [float(v) for v in t[1:]]
    glove_dict[word] = embedding

tokens = []
embeddings = []
for word, freq in sorted(word_freq_dict.items(), key=lambda d: d[1], reverse=True):
  if freq <= 2:
    break
  if word not in glove_dict:
    continue
  tokens.append(word)
  embeddings.append(glove_dict[word])

print(len(tokens))

embeddings = np.array(embeddings)
pickle_dump(tokens, 'squad_data/vocab/squad-v1.1.vocab.pkl')
pickle_dump(embeddings, 'squad_data/vocab/squad-v1.1.emb.pkl')
