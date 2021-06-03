import os
import torch
from torchtext.datasets import WikiText2

from model import TransformerModel, PositionalEncoding  # need all class definitions for un-pickle
from model import load_model
from model_utils import gen_tokenizer_and_vocab
from settings import DIR_MODELS

# device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify path to model or model_weights
model_path_A = DIR_MODELS + os.sep + 'model_epoch3.pth'
model_path_B = DIR_MODELS + os.sep + 'model_weights_epoch3.pth'

# load dataset, tokenizer, vocab
tokenizer, vocab = gen_tokenizer_and_vocab()
train_iter, val_iter, test_iter = WikiText2()

# load method A and B
model_A = load_model(model_path_A, device, as_pickle=True, vocab=None)
model_B = load_model(model_path_B, device, as_pickle=True, vocab=vocab)

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
