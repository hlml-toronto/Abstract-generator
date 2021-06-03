import os
import torch

from model import TransformerModel, PositionalEncoding  # need all class definitions for un-pickle
from model_utils import gen_tokenizer_and_vocab
from settings import DIR_MODELS

# specify path to model or model_weights
model_path_A = DIR_MODELS + os.sep + 'model_epoch3.pth'
model_path_B = DIR_MODELS + os.sep + 'model_weights_epoch3.pth'

# device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load method A
model_A = torch.load(model_path_A, map_location=device)
model_A.eval()

# load method B
tokenizer, vocab = gen_tokenizer_and_vocab()
# TODO this is risky way to load -- what if they were trained differently? refactor
ntokens = len(vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
# instantiate + fill in weights
model_B = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
model_B.load_state_dict(torch.load(model_path_B, map_location=device))
model_B.eval()

# inspect both models
print('model_A info...\n', model_A)
print('\nmodel_B info...\n', model_B)

print('model_A == model_B:', model_A == model_B)


def gen_some_text(model, first_word='The', tokens_to_gen=10):
    model.eval()
    model.forward()
    return


generated_text = gen_some_text(model_A)
print(generated_text)
