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
    return tokenized_text_ints


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
        raw_tokenized_text = tokenize_some_text(text=text_prompt)
        nn = raw_tokenized_text.shape[0]

        print(raw_tokenized_text)
        print(nn)

        if nn > BPTT:
            input_slice = raw_tokenized_text[nn-BPTT:]
            init_mask = model.generate_square_subsequent_mask(BPTT).to(device)
        else:
            input_slice = raw_tokenized_text[0:nn]
            init_mask = model.generate_square_subsequent_mask(BPTT).to(device)

        processed_tokenized_text = torch.zeros(BPTT, dtype=torch.long) + dummy_token
        processed_tokenized_text[0:nn] = input_slice
        processed_tokenized_text.to(device)

        return processed_tokenized_text, init_mask

    # 1) tokenize the text prompt and prepare associated src_mask for model.forward()
    processed_tokenized_text, initial_mask = process_prompt()
    print(processed_tokenized_text)
    print(initial_mask)

    # 2)
    model.eval()
    for idx in range(tokens_to_gen):
        out = model.forward(processed_tokenized_text, initial_mask)
        print(out.shape)
        best_guess = torch.argmax(out)
        print(len(vocab.itos), best_guess)
        best_string = vocab.itos[best_guess]
        print(best_guess, best_string)

        # update total_text_string by adding best guess
        total_text_string += ' %s' % best_string

        # update 35 long running window

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
    prompt = 'The dog ran across the'
    ngen = 10
    generated_text = gen_some_text(model_A, vocab, device, text_prompt=prompt, tokens_to_gen=ngen)
    print("Text prompt:", prompt)
    print("Number of tokens to generate:", ngen)
    print("Generated_text:")
    print(generated_text)
