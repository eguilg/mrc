import multiprocessing as mp

from torch.utils.data import DataLoader

from dataset import MaiDataSource, MaiDataset, Vocab, MaiIndexTransform, MethodBasedBatchSampler

trainset_roots = [
	'./data/train/gen/question/samples_jieba500',
	'./data/train/gen/question/samples_pyltp500',
]

jieba_base_v = Vocab('./data/embed/base_token_vocab_jieba.pkl',
					 './data/embed/base_token_embed_jieba.pkl')
jieba_sgns_v = Vocab('./data/embed/train_sgns_vocab_jieba.pkl',
					 './data/embed/train_sgns_embed_jieba.pkl')
jieba_flag_v = Vocab('./data/embed/base_flag_vocab_jieba.pkl',
					 './data/embed/base_flag_embed_jieba.pkl')

pyltp_base_v = Vocab('./data/embed/base_token_vocab_pyltp.pkl',
					 './data/embed/base_token_embed_pyltp.pkl')
pyltp_sgns_v = Vocab('./data/embed/train_sgns_vocab_pyltp.pkl',
					 './data/embed/train_sgns_embed_pyltp.pkl')
pyltp_flag_v = Vocab('./data/embed/base_flag_vocab_pyltp.pkl',
					 './data/embed/base_flag_embed_pyltp.pkl')

transform = MaiIndexTransform(jieba_base_v, jieba_sgns_v, jieba_flag_v, pyltp_base_v, pyltp_sgns_v, pyltp_flag_v)
train_data_source = MaiDataSource(trainset_roots)
train_data_source.split()
dev_data_set = MaiDataset(train_data_source.dev, transform)

train_loader = DataLoader(
	dataset=MaiDataset(train_data_source.train, transform),
	batch_sampler=MethodBasedBatchSampler(train_data_source.train, batch_size=32),
	num_workers=mp.cpu_count()
)

dev_loader = DataLoader(
	dataset=MaiDataset(train_data_source.dev, transform),
	batch_sampler=MethodBasedBatchSampler(train_data_source.dev, batch_size=32, shuffle=False),
	num_workers=mp.cpu_count()
)
