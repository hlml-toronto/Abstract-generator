import os
import math

import torch.nn as nn
import torch
from torchtext.datasets import WikiText2

from model import TransformerModel, PositionalEncoding  # need all class definitions for un-pickle
from model import load_model, evaluate
from model_utils import gen_tokenizer_and_vocab, data_process, batchify
from settings import DIR_MODELS, BPTT

if __name__ == '__main__':
    """
      3 epochs: | End of training | test loss  6.54 | test ppl   692.65
     20 epochs:
    100 epochs: | End of training | test loss  5.48 | test ppl   239.20
    """
    # device settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # specify path to model or model_weights
    model_path_A = DIR_MODELS + os.sep + 'model_epoch20.pth'
    model_path_B = DIR_MODELS + os.sep + 'model_weights_epoch20.pth'

    # load dataset, tokenizer, vocab
    tokenizer, vocab = gen_tokenizer_and_vocab()
    train_iter, val_iter, test_iter = WikiText2()

    # load method A and B
    model_A = load_model(model_path_A, device, as_pickle=True, vocab=None)
    model_B = load_model(model_path_B, device, as_pickle=False, vocab=vocab)
    
    # data loading
    train_data = data_process(train_iter, vocab, tokenizer)
    val_data = data_process(val_iter, vocab, tokenizer)
    test_data = data_process(test_iter, vocab, tokenizer)

    batch_size = 20
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size, device)
    val_data = batchify(val_data, eval_batch_size, device)
    test_data = batchify(test_data, eval_batch_size, device)

    # report model error on test set
    ntokens = len(vocab.stoi)
    test_loss = evaluate(model_A, test_data, device, ntokens, nn.CrossEntropyLoss())
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
    print(ntokens)
