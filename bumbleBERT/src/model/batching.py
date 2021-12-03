import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class CustomBatch():
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or maxLen. Unclear to me whether this is instantiated
    at every call to dataloader or if it's instantiated along with dataloader.
    For now, all the tensors exist on CPU and are later pushed to the GPU. Needs
    potentially to be changed.
    """

    def __init__(self, data, dim=0, maxLenModel=100, padValue=0, stackDim=1):
        """
        Input:
            data (dataset)      : a batch of dataset.
            dim (int)           : the dimension to be padded (dimension of time in sequences)
            maxLenModel (int)   : maixmum length of sentence
            padValue (int)      : the value for padding.
            stackDim (int)      : dimension along which to stack the data in tensor.
                                    1 for huggingface, 0 for annotated transformer
        """
        self.dim = dim; self.padValue = padValue

        max_len_seq = np.max( [ x.shape[self.dim] for x in data ] )
        self.maxLen = np.min( [max_len_seq, maxLenModel] )

        # pad according to max_len
        batch = [self.pad_tensor(x[:self.maxLen]) for x in data ]
        #pad_mask = [  ]
        # stack all, change to dim = 0 for annotated transformer?
        self.src = (torch.stack([x[:-1] for x in batch], dim=stackDim)).long()
        self.tgt = (torch.stack([x[1:] for x in batch], dim=stackDim)).long()
        self.src_pad_mask = (self.src == self.padValue)
        #self.src_pad_mask = torch.transpose( (self.src != self.padValue).type(torch.int), 0, 1 )
        #self.tgt_pad_mask = (self.tgt != self.padValue).type(torch.int)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))

    def pad_tensor(self, vec):
        """
        padding a tensor which represents a batch

        Input:
            vec : tensor to pad

        Output:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        padSize = list(vec.shape)
        padSize[self.dim] = self.maxLen - vec.size(self.dim)
        return torch.cat([vec, self.padValue*torch.ones(*padSize)], dim=self.dim)

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self
