import json
import os.path as osp
import pickle
import shutil

import bz2file
import numpy as np
import torch
from torch.nn import Parameter

from .osutils import mkdir_if_missing


def read_elmo_vocab(vocab_path, embedding_path):
	word_to_id = {'<pad>': 0}
	with open(vocab_path, encoding='utf-8') as f:
		lines = f.readlines()
		for line in lines[2:]:  # 这里的[2:]是去掉开头的<S>和</S>
			word_to_id[line.strip()] = len(word_to_id)

	embeddings = read_pkl(embedding_path)
	embeddings = embeddings[2:]
	embeddings = np.concatenate([[np.zeros(len(embeddings[0]), dtype=np.float32)], embeddings], axis=0)
	assert len(word_to_id) == len(embeddings)

	return word_to_id, embeddings


def bz2_vocab_reader(fpath):
	with bz2file.open(fpath, 'r') as f:
		info = f.readline()
		while True:
			data = f.readline()
			data = data.decode('utf-8').strip()
			if data == "":
				break
			try:
				t_data = data.split()
				word = t_data[0].strip()
				embed = [float(v) for v in t_data[-300:]]
			except Exception:
				continue
			yield word, embed


def read_json(fpath):
	with open(fpath, 'r', encoding='utf-8') as f:
		obj = json.load(f, encoding='utf-8')
	return obj


def write_json(obj, fpath):
	mkdir_if_missing(osp.dirname(fpath))
	with open(fpath, 'w', encoding='utf-8') as f:
		json.dump(obj, f, ensure_ascii=False)


def write_pkl(data, fpath):
	mkdir_if_missing(osp.dirname(fpath))
	with open(fpath, 'wb') as f:
		pickle.dump(data, f, True)


def read_pkl(fpath):
	with open(fpath, 'rb') as f:
		data = pickle.load(f)
	return data


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
	mkdir_if_missing(osp.dirname(fpath))
	torch.save(state, fpath)
	if is_best:
		shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):
	if osp.isfile(fpath):
		checkpoint = torch.load(fpath)
		print("=> Loaded checkpoint '{}'".format(fpath))
		return checkpoint
	else:
		raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
	tgt_state = model.state_dict()
	copied_names = set()
	for name, param in state_dict.items():
		if strip is not None and name.startswith(strip):
			name = name[len(strip):]
		if name not in tgt_state:
			continue
		if isinstance(param, Parameter):
			param = param.data
		if param.size() != tgt_state[name].size():
			print('mismatch:', name, param.size(), tgt_state[name].size())
			continue
		tgt_state[name].copy_(param)
		copied_names.add(name)

	missing = set(tgt_state.keys()) - copied_names
	if len(missing) > 0:
		print("missing keys in state_dict:", missing)

	return model
