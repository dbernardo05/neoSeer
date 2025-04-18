
# %% auto 0
__all__ = ['TransformerModelMod', 'TransformerPlus']

# %% ../../nbs/108_models.TransformerModel.ipynb 3
from tsai.imports import *
from tsai.models.layers import *
from tsai.models.utils import *
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer


class TransformerModelMod(Module):
	def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, activation="gelu", n_layers=1):
		"""
		Args:
			c_in: the number of features (aka variables, dimensions, channels) in the time series dataset
			c_out: the number of target classes
			d_model: total dimension of the model.
			nhead:  parallel attention heads.
			d_ffn: the dimension of the feedforward network model.
			dropout: a Dropout layer on attn_output_weights.
			activation: the activation function of intermediate layer, relu or gelu.
			num_layers: the number of sub-encoder-layers in the encoder.
			
		Input shape:
			bs (batch size) x nvars (aka variables, dimensions, channels) x seq_len (aka time steps)
			"""
		self.permute = Permute(2, 0, 1)
		self.inlinear = nn.Linear(c_in, d_model)
		self.relu = nn.ReLU()
		self.gelu = nn.GELU()
		encoder_layer = TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ffn, dropout=dropout, activation=activation)
		encoder_norm = nn.LayerNorm(d_model)        
		self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers, norm=encoder_norm)
		self.transpose = Transpose(1, 0)
		self.max = Max(1)
		# self.outlinear = nn.Linear(d_model, c_out)

	def forward(self,x):
		x = self.permute(x) # bs x nvars x seq_len -> seq_len x bs x nvars
		x = self.inlinear(x) # seq_len x bs x nvars -> seq_len x bs x d_model
		x = self.relu(x)
		x = self.transformer_encoder(x)
		x = self.transpose(x) # seq_len x bs x d_model -> bs x seq_len x d_model
		x = self.max(x)
		# x = self.gelu(x)
		# x = self.head(x)
		return x



 
		
def get_activation_fn(activation):
	if activation == "relu": return nn.ReLU()
	elif activation == "gelu": return nn.GELU()
	else: return activation()
	


class TransformerPlus(nn.Sequential):
	"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs"""
	def __init__(self, c_in, c_out, d_model=64, n_head=1, d_ffn=128, dropout=0.1, fc_dropout=0.1, activation="gelu", n_layers=1, **kwargs):
		# Backbone
		backbone = TransformerModelMod(c_in, c_out, d_model=d_model, n_head=n_head, 
			d_ffn=d_ffn, dropout=dropout, activation=activation, n_layers=n_layers, **kwargs)

		# Head
		self.head_nf = d_model
		self.c_out = c_out
		head = self.create_head(self.head_nf, c_out, act=activation, flatten=flatten, fc_dropout=fc_dropout)
		super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))


	def create_head(self, nf, c_out, act="gelu", fc_dropout=0.1, y_range=None, **kwargs):
		layers = [get_activation_fn(act), Flatten()]
		if fc_dropout: layers += [nn.Dropout(fc_dropout)]
		layers += [nn.Linear(nf, c_out)]
		if y_range: layers += [SigmoidRange(*y_range)]
		return nn.Sequential(*layers)   