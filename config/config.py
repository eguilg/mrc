class BaseConfig:
	def __init__(self):
		self.model_params = {
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'rnn_type': 'lstm',
			'num_features': 1,
			'dropout': 0.2,
			'hidden_size': 75,

			'num_qtype': 10,

			'backbone_kwarg': {},
			'ptr_kwarg': {}
		}


#  R NET ####################################################
class r_net_1(BaseConfig):
	def __init__(self):
		super(r_net_1, self).__init__()
		self.name = 'r_net_1'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'dropout': 0.1,
			'hidden_size': 75,
		})


class r_net_2(BaseConfig):
	def __init__(self):
		super(r_net_2, self).__init__()
		self.name = 'r_net_2'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'dropout': 0.2,
			'hidden_size': 75,
		})


class r_net_3(BaseConfig):
	def __init__(self):
		super(r_net_3, self).__init__()
		self.name = 'r_net_3'
		self.model_params.update({
			'backbone_type': 'r_net',
			'ptr_type': 'ptr',
			'dropout': 0.3,
			'hidden_size': 100,
		})


#  M READER #################################################
class m_reader_1(BaseConfig):

	def __init__(self):
		super(m_reader_1, self).__init__()
		self.name = 'm_reader_1'
		self.model_params.update({
			'backbone_type': 'm_reader',
			'ptr_type': 'm_ptr',
			'dropout': 0.1,
			'hidden_size': 75,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {'hop': 1}
		})


class m_reader_2(BaseConfig):
	def __init__(self):
		super(m_reader_2, self).__init__()
		self.name = 'm_reader_2'
		self.model_params.update({
			'backbone_type': 'm_reader',
			'ptr_type': 'm_ptr',
			'dropout': 0.2,
			'hidden_size': 75,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {'hop': 2}
		})


class m_reader_3(BaseConfig):
	def __init__(self):
		super(m_reader_3, self).__init__()
		self.name = 'm_reader_3'
		self.model_params.update({
			'backbone_type': 'm_reader',
			'ptr_type': 'm_ptr',
			'dropout': 0.3,
			'hidden_size': 100,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {'hop': 2}
		})


#  BiDAF #################################################
class bi_daf_1(BaseConfig):

	def __init__(self):
		super(bi_daf_1, self).__init__()
		self.name = 'bi_daf_1'
		self.model_params.update({
			'backbone_type': 'bi_daf',
			'ptr_type': 'ptr',
			'dropout': 0.1,
			'hidden_size': 80,

		})


class bi_daf_2(BaseConfig):
	def __init__(self):
		super(bi_daf_2, self).__init__()
		self.name = 'bi_daf_2'
		self.model_params.update({
			'backbone_type': 'bi_daf',
			'ptr_type': 'ptr',
			'dropout': 0.2,
			'hidden_size': 100,

		})


class bi_daf_3(BaseConfig):
	def __init__(self):
		super(bi_daf_3, self).__init__()
		self.name = 'bi_daf_3'
		self.model_params.update({
			'backbone_type': 'bi_daf',
			'ptr_type': 'ptr',
			'dropout': 0.3,
			'hidden_size': 150,
		})


#  SLQA #################################################
class slqa_1(BaseConfig):

	def __init__(self):
		super(slqa_1, self).__init__()
		self.name = 'slqa_1'
		self.model_params.update({
			'backbone_type': 'slqa',
			'ptr_type': 'qv_ptr',
			'dropout': 0.2,
			'hidden_size': 64,

			'backbone_kwarg': {'hop': 3},
			'ptr_kwarg': {}
		})


class slqa_2(BaseConfig):
	def __init__(self):
		super(slqa_2, self).__init__()
		self.name = 'slqa_2'
		self.model_params.update({
			'backbone_type': 'slqa',
			'ptr_type': 'qv_ptr',
			'dropout': 0.3,
			'hidden_size': 100,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {}
		})


class slqa_3(BaseConfig):
	def __init__(self):
		super(slqa_3, self).__init__()
		self.name = 'slqa_3'
		self.model_params.update({
			'backbone_type': 'slqa',
			'ptr_type': 'qv_ptr',
			'dropout': 0.3,
			'hidden_size': 125,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {}
		})


#  SLQA #################################################
class slqa_plus_1(BaseConfig):

	def __init__(self):
		super(slqa_plus_1, self).__init__()
		self.name = 'slqa_plus_1'
		self.model_params.update({
			'backbone_type': 'slqa_plus',
			'ptr_type': 'qv_ptr',
			'dropout': 0.2,
			'hidden_size': 64,

			'backbone_kwarg': {'hop': 3},
			'ptr_kwarg': {}
		})


class slqa_plus_2(BaseConfig):

	def __init__(self):
		super(slqa_plus_2, self).__init__()
		self.name = 'slqa_plus_2'
		self.model_params.update({
			'backbone_type': 'slqa_plus',
			'ptr_type': 'qv_ptr',
			'dropout': 0.3,
			'hidden_size': 100,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {}
		})


class slqa_plus_3(BaseConfig):

	def __init__(self):
		super(slqa_plus_3, self).__init__()
		self.name = 'slqa_plus_3'
		self.model_params.update({
			'backbone_type': 'slqa_plus',
			'ptr_type': 'qv_ptr',
			'dropout': 0.3,
			'hidden_size': 125,

			'backbone_kwarg': {'hop': 2},
			'ptr_kwarg': {}
		})