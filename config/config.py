class BaseConfig:
	model_params = {
		'backbone_type': 'r_net',
		'ptr_type': 'ptr',
		'rnn_type': 'lstm',
		'num_features': 1,
		'dropout': 0.2,
		'hidden_size': 150,

		'backbone_kwarg': {},
		'ptr_kwarg': {}
	}


class r_net_1(BaseConfig):
	def __init__(self):
		self.name = 'r_net_1'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'dropout': 0.2,
			'hidden_size': 150,
		})


class r_net_2(BaseConfig):
	def __init__(self):
		self.name = 'r_net_2'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'dropout': 0.3,
			'hidden_size': 200,
		})


class m_reader_1(BaseConfig):
	def __init__(self):
		self.name = 'm_reader_1'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'm_ptr',
			'dropout': 0.1,
			'hidden_size': 100,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {'hop': 2}
		})


class m_reader_2(BaseConfig):
	def __init__(self):
		self.name = 'm_reader_2'
		self.model_params.update({
			'backbone_type': 'm_reader',
			'ptr_type': 'm_ptr',
			'dropout': 0.2,
			'hidden_size': 150,

			'backbone_kwarg': {'hop': 3},
			'ptr_kwarg': {'hop': 2}
		})
