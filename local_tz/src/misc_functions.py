import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy

def subsequent_mask(size):
	"""Mask out subsequent positions."""
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0 # sets diagonal and below to True, above to False

def attention(query, key, value, mask=None, dropout=None):
	"""Compute 'Scaled Dot Product Attention' (Equ 1)"""
	d_k = query.size(-1)
	## (Encode:) query, key are both [30, 8, 10, 64], scores is [30, 8, 10, 10]
	scores = torch.matmul(query, key.transpose(-2, -1)) \
			 / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask == 0, -1e9) # sets masked scores to -inf
	p_attn = F.softmax(scores, dim = -1) # computes softmax along last dimension
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

def clones(module, N):
	"""Produce N identical layers."""
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)]) # holds submodules in a list
