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
from torch.autograd import Variable

row_embeds = nn.Embedding(64, 2)
col_embeds = nn.Embedding(64, 2)

hello_idx = torch.LongTensor(list(range(10)))
hello_idx = Variable(hello_idx)
row = row_embeds(hello_idx)
col = col_embeds(hello_idx).transpose(0, 1)

outer3 = torch.mm(row, col).unsqueeze(0)
outer3 = torch.cat([outer3 for _ in range(8)], 0)
print(outer3.shape)
outer3 = outer3.unsqueeze(1)
print(outer3.shape)
