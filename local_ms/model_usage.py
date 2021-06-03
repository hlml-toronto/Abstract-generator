import os
import torch
from torchtext.datasets import WikiText2

from model import TransformerModel, PositionalEncoding  # need all class definitions for un-pickle
from model import load_model
from model_utils import gen_tokenizer_and_vocab
from settings import DIR_MODELS, BPTT


def tokenize_some_text(text='The dog ran across the'):
    """
    Current vanilla pytorch tokenizer
        1) Split by spaces into word tokens
        2) Use vocab to creat integer rep of each token
    """
    tokenized_text = tokenizer(text)
    tokenized_text_ints = torch.tensor([vocab[token] for token in tokenized_text], dtype=torch.long)
    return tokenized_text, tokenized_text_ints


def gen_some_text(model, vocab, device, text_prompt='The dog ran across the', tokens_to_gen=10):
    """
    dummy_token: if text_prompt is < BPTT tokens, need to add dummy tokens and mask them
    1) tokenize the text prompt
    """
    total_text_string = text_prompt  # this will be extended by tokens_to_gen

    def process_prompt(dummy_token=0):
        # Two cases:
        # - if less than BPTT (context length), need to add dummy tokens
        # - if longer than BPTT (context length), truncate to BPTT
        text_split, tokenized_text = tokenize_some_text(text=text_prompt)
        nn = tokenized_text.shape[0]

        print(tokenized_text)
        print(nn)

        if nn > BPTT:  # take last BPTT elements
            input_slice = tokenized_text[nn-BPTT:]
            src_mask = model.generate_square_subsequent_mask(BPTT).to(device)
        else:
            input_slice = tokenized_text[0:nn]
            src_mask = model.generate_square_subsequent_mask(nn).to(device)

        #processed_tokenized_text = torch.zeros(BPTT, dtype=torch.long) + dummy_token
        #processed_tokenized_text[0:nn] = input_slice
        src = torch.zeros((nn,1), dtype=torch.long)
        src[0:nn, 0] = tokenized_text
        src.to(device)

        return text_split, src, src_mask

    # 1) tokenize the text prompt and prepare associated src_mask for model.forward()
    prompt_split, src, src_mask = process_prompt()  # src should be in form ntokens x nbatches
    nn = src_mask.shape[0]
    src.reshape((nn, 1))
    print(src)
    print(src_mask)

    # 2)
    model.eval()
    for idx in range(tokens_to_gen):
        out = model.forward(src, src_mask)
        print(out.shape)

        """
        for idx in range(5):
            print('Input %d:' % idx, prompt_split[0:idx+1])
            qq = torch.argmax(out[idx,0,:])
            print('Guess %d:' % idx, vocab.itos[qq])
            print(qq)
            print(vocab.itos[qq])
        """

        best_guess_int = torch.argmax(out[nn-1, 0])  # care batch dimension, nn vs BPTT
        best_guess_string = vocab.itos[best_guess_int]
        print('best_guess_int, best_guess_string:', best_guess_int, best_guess_string)

        # update total_text_string by adding best guess
        total_text_string += ' %s' % best_guess_string

        # update src and src mask for next pass of model.forward()
        if nn < BPTT:
            # extend and shift the running window of input data
            src_updated = torch.zeros((nn + 1,1), dtype=torch.long)  # extend src to nn
            src_updated[0:nn, 0] = src[:, 0]
            src_updated[nn, 0] = best_guess_int
            src = src_updated.to(device)
            # increment mask dimension
            src_mask = model.generate_square_subsequent_mask(nn + 1).to(device)
            nn += 1
        else:
            src_orig = src.clone()
            src[0:BPTT-1, 0] = src_orig[1:, 0]
            src[-1, 0] = best_guess_int

    return total_text_string


if __name__ == '__main__':
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
    model_B = load_model(model_path_B, device, as_pickle=False, vocab=vocab)

    # inspect both models
    print('model_A info...\n', model_A)
    print('\nmodel_B info...\n', model_B)

    print('model_A == model_B:', model_A == model_B)

    # Text generation example
    prompt = 'Text generation is easier than you think , however'
    ngen = 30
    generated_text = gen_some_text(model_B, vocab, device, text_prompt=prompt, tokens_to_gen=ngen)
    print("Text prompt:\n", prompt)
    print("Number of tokens to generate:", ngen)
    print("Generated_text:\n", generated_text)
