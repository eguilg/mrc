# encoding:utf-8
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from dataset import (TitleSummDataset, Vocab,
                     TitleSummTransform, MethodBasedBatchSampler)
from models.losses import PointerLoss, RougeLoss
from models.losses.obj_detection_loss import ObjDetectionLoss

from models.rc_model import RCModel
from utils.osutils import *
from metrics.rouge import rouge_eval
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

import postprocess
from title_summ_tokenizer import TitleSummBertTokenizer
from torch.utils.data import Dataset
import json
from pytorch_pretrained_bert import BertModel
from models.title_models import BertForTitleSumm
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import DataLoader

for i in range(2, 5 + 1):
  model_path = 'title_data/models/MODE_MRT.bert.state%d' % i
  model = torch.load(model_path)

  state = {}
  state['best_model_state'] = model.state_dict()
  state['best_epoch'] = 1
  state['best_step'] = 1
  state['best_loss'] = 1

  torch.save(state, 'title_data/models/MODE_MRT.bert.state%d.tmp' % i)
