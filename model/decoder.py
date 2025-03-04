
import os
import re
import gc
import pdb
import copy
import torch
import pickle
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
from pathlib import Path
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score

from model.encoder import Encoder


class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
		super().__init__()
		self.num_layers = num_layers
		h = [hidden_dim] * (num_layers -1)
		self.layers = nn.ModuleList(nn.Linear(n,k) for n,k in zip([input_dim] +h, h + [output_dim]))
		
	def forward(self, x):
		for i, layer in enumerate(self.layers):
			x = F.relu(layer(x)) if i<self.num_layers -1 else layer(x)
		return x



class TransformerDecoder(nn.Module):
	__constants__ = ['norm']

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(TransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, x_, x):
		
		for mod in self.layers:
			x_, x = mod(x_, x)
		x = x * x_
		
		return x #, atn

class TransformerDecoderLayer(nn.Module):
	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False, num_queries=49, 
				 device=None, dtype=None, layers=1) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(TransformerDecoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
											**factory_kwargs)
		self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
												 **factory_kwargs)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)
		self.layers = _get_clones(self, layers)
		
		self.activation = F.relu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)


	def forward(self, e2, e):
		
		# q = e2
		# k, v = e, e
		for layer in self.layers:
			e = self.norm1(e + self._sa_block(e))
			e2 = self.norm1(e2 + self._sa_block(e2))

			e2 = self.norm2(e2 + self._mha_block(e2, e, e))
			e2 = self.norm3(e2 + self._ff_block(e2))

			e = self.norm3(e + self._ff_block(e2))

		e = e*e2
		return e


	# self-attention block
	def _sa_block(self, x):
		x = self.self_attn(x, x, x)[0]
		return self.dropout1(x)

	# multihead attention block
	def _mha_block(self, x,k,v):
		x = self.multihead_attn(x, k, v)[0]
		return self.dropout2(x)

	# feed forward block
	def _ff_block(self, x):
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout3(x)

def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class Decoder(nn.Module):
	def __init__(self, hidden_dim=768, num_queries=16, num_decoder_layers=6, num_class=3):
		super().__init__()

		self.num_queries = num_queries
		self.hidden_dim = hidden_dim
		self.num_decoder_layers = num_decoder_layers
		self.num_class = num_class

		self.decoder_layer = TransformerDecoderLayer(hidden_dim, nhead=8, layers=num_decoder_layers)
		# self.decoder = TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

		self.class_embed = nn.Linear(hidden_dim, num_class+1)
		self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
		self.position_encoding = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
		

	def forward(self, e2, e):
		e += self.position_encoding
		e2 += self.position_encoding
		
		output = self.decoder_layer(e2, e)

		# Generate class and bbox embeddings from the output tensor
		class_output = self.class_embed(output)
		bbox_output = self.bbox_embed(output).sigmoid()
		
		# Reshape the bbox embeddings into the correct format
		class_output = class_output.view(-1, self.num_queries, self.num_class+1)
		bbox_output = bbox_output.view(-1, self.num_queries, 4)

		# class_output = torch.softmax(class_output, dim=-1)
		return class_output, bbox_output




class CSAT_detection(nn.Module):
	def __init__(self, encoder=None, hidden_dim=768, nheads=8,
				num_encoder_layers=6, dropout=0.1,
				num_queries=256, num_decoder_layers=6, num_class=3, multiscale=False):
		super().__init__()
		if encoder is None:
			self.encoder=Encoder(hidden_dim, nheads, num_encoder_layers, dropout, multiscale=multiscale, detector=True)
			
		else:
			self.encoder = encoder

		self.decoder=Decoder(hidden_dim, num_queries, num_decoder_layers, num_class)
		self.pool=nn.MaxPool3d(kernel_size=(3,1,1))


	def forward(self, inputs):	# (3,b,3,h,w)
		e1=self.encoder(inputs[:,0])
		e2=self.encoder(inputs[:,1]) # (B, 49, 768)
		e3=self.encoder(inputs[:,2])

		e=torch.stack([e1,e2,e3]).permute(1,0,2,3)
		print(e.shape)
		# 
		e=self.pool(e).squeeze(1) # (B, 49, 768)
		outputs=self.decoder(e2, e)

		return outputs



if __name__=='__main__':
	abc = torch.rand((3, 1, 3, 224, 224))
	detector = CSAT_detection(multiscale=True)
	res = detector(abc)