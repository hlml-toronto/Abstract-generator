import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


"""
See URL: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

Code notes:
1) PyTorch has data loader classes WikiText2 or WikiText103 (50x bigger) and a Vocab class 
    - Vocab class builds a word level dictionary, with each word having specific index int
    - Data notes: 
        - RAW: WikiText2 training data is a list of ~30,000 "lines" or "paragraphs" (600 articles)
        - STEP 1: This is converted to stream of ~2 million tokens 
        - STEP 2: Divide into batched dataset with 2D shape - (2 million // nbatches, nbatches)
    - When the data is actually passed through the network, it is done in sweeps of length BPTT
        - e.g. BPTT = 5, input could be: 'The dog ran across the', target is the next word.  
    - To make network input output pairs, "get_batch(batched_dataset, batch_idx)" is called
        - returns batch_input, batch_target
        - batch_input has form  (2D) X.shape = (bptt, nbatches), with elements being vocab integers
        - batch_target has form (1D) Y.shape = bptt * nbatches, with elements being vocab integers
            - contains the next token for each input 'sentence' of length **up to** bptt 

- Example: Dataset = alphabet, 26 total tokens.
    - Params: (nbatches = 1), (bptt = 2)
        - batched_dataset = [A, B, ..., Z]  (as 26 x 1 tensor)
        - get_batch(batched_dataset, 0) returns
            X = [A, B] as 2 x 1   (2D tensor)
            Y = [B, C] as 2       (1D tensor)
        - get_batch(batched_dataset, 2) returns
            X = [C, D] as 2 x 1   (2D tensor)
            Y = [D, E] as 2       (1D tensor)
    - Params: (nbatches = 2), (bptt = 4)
          - batched_dataset = [A, N,
                               B, O,  (as 13 x 2 tensor)
                              ..., 
                               M, Z]
          - get_batch(batched_dataset, 0) returns
            X = 4 x 2 tensor, Y = 8 1D tensor
            X = [A, N
                 B, O
                 C, P
                 D, Q],  
            Y_as_2D = [B, O
                       C, P
                       D, Q
                       E, R],
            Y_as_1D = [B, O, C, P, D, Q, E, R]  (using reshape(-1) as in their code).
 
2) Params to choose:
    - "bptt" which sets the window length for input text 
      (e.g. bptt = 7 then each window has 7 words, and we want to predict the 8th word) 
    - batch size (training and evaluation, can be different)
    - model:
        ntokens = len(vocab.stoi)  # the size of vocabulary
        emsize = 200               # embedding dimension
        nhid = 200                 # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2                # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2                  # the number of heads in the multiheadattention models
        dropout = 0.2              # the dropout value
    - optimization:
        lr = 5.0                   # learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

3) model.forward() takes in: data, src_mask
    - src_mask "blocks out next word" (bptt x bptt square matrix of elements either 0 or -inf)
        [0 -inf ... -inf]      [The ??? ... ???]
        [0   0  ... -inf]      [The dog ... ???]
        [0  ... ...  ...] ~~~~ [The dog ... ???]
        [0   0  ...   0 ]      [The dog ... street]
    - When the input is a  2D tensor X = bptt x nbatch, 
           the output is a 3D tensor Z = bptt x nbatch x ntokens
    - The output represents the unnormalized probability of the next word given the input,
      for each element of the input tensor (properly masked using src_mask)

4) Loss function ("criterion"): nn.CrossEntropyLoss
    - input A is target class (i.e. the integer of the target word)
    - input B is the model output, vector of class probabilities
"""


def pad_tensor(vec, pad_size, dim, pad):
    # TODO why is pad_size unused?
    """
    Input:
        vec : tensor to pad
        pad_size : the size to pad to
        dim : dimension to pad
        pad : value of pad

    Output:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad_size - vec.size(dim)
    return torch.cat([vec, pad * torch.ones(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or max_len
    """

    def __init__(self, dim=0, max_len=100, pad_value=0):
        """
        Input:
            dim : the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.maxLen = max_len
        self.padValue = pad_value

    def pad_collate(self, batch):
        """
        Input:
            batch : list of (tensor, label)

        Output:
            xs : a tensor of all examples in 'batch' after padding
            ys : a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len_seq = np.max( [ x.shape[self.dim] for x in batch ] )
        max_len = np.min( [max_len_seq, self.maxLen] )

        # pad according to max_len
        batch = [pad_tensor(x[:max_len], pad_size=max_len, pad=self.padValue
                            , dim=self.dim) for x in batch ]
        # stack all
        data = torch.stack([x[:-1] for x in batch], dim=1) # change to dim = 0 for annotated transformer?
        target = torch.stack([x[1:] for x in batch], dim=1)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))
        return [data.long(), target.long()]

    def __call__(self, batch):
        return self.pad_collate(batch)


class PadCollateMemoryOnGPU:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or max_len
    """

    def __init__(self, dim=0, max_len=100, pad_value=0):
        """
        Input:
            dim : the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim
        self.maxLen = max_len
        self.padValue = pad_value

    def pad_collate(self, batch):
        """
        Input:
            batch : list of (tensor, label)

        Output:
            xs : a tensor of all examples in 'batch' after padding
            ys : a LongTensor of all labels in batch
        """
        # find longest sequence
        print(batch[0])
        max_len_seq = np.max( [ x.shape[self.dim] for x in batch ] )
        max_len = np.min( [max_len_seq, self.maxLen] )

        # pad according to max_len
        batch = [pad_tensor(x[:max_len], pad_size=max_len, pad=self.padValue
                            , dim=self.dim) for x in batch ]
        # stack all
        data = torch.stack([x[:-1] for x in batch], dim=1) # change to dim = 0 for annotated transformer?
        target = torch.stack([x[1:] for x in batch], dim=1)
        #ys = torch.LongTensor(map(lambda x: x[1], batch))
        return [data.long(), target.long()]

    def __call__(self, batch):
        return self.pad_collate(batch)


class TransformerModel(nn.Module):
    """
    Notes on semantics:
    - This appears to be defined as an "Encoder only" transformer
    - It is however defined very similar to GPT models, where they instead say: "stack of decoders"
      https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
        - They call their network a stack of decoders (seemingly because they use subsequent masks)
        - From Attention is All You Need (2017):
            - Encoder = self-attention with no "subsequent text masking"
            - Decoder =      attention with    "subsequent text masking" + input from encoder side
    - The "TransformerEncoder" class supports masking and that's how it's used here
    - The "TransformerDecoder" class is more complex, supports additional inputs (e.g. from Encoder)
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # used with subsequent mask its more of a decoder?
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, src_key_padding_mask=None): # should I add a padding mask here?
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # TODO: jeremy line below -- as it is, transformer_encoder only has two arguments...
        #  output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def save_model(model, fpath, as_pickle=True):
    if as_pickle:
        # approach 1: save model (class) entirely (uses pickle)
        torch.save(model, fpath)
    else:
        # approach 2: save model weights
        torch.save(model.state_dict(), fpath)


def load_model(model_path, device, as_pickle=True, vocab=None):
    """
    If not using as_pickle:
        TODO this is risky way to load -- what if they were trained differently? refactor
        Need to supply vocab class (for ntokens argument in model instantiation)
    """
    if as_pickle:
        model = torch.load(model_path, map_location=device)
    else:
        ntokens = len(vocab.stoi)  # the size of vocabulary
        emsize = 200  # embedding dimension
        nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2  # the number of heads in the multiheadattention models
        dropout = 0.2  # the dropout value
        # instantiate + fill in weights
        model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model