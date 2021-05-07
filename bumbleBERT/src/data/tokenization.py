import os

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import NFD, NFKD, NFC, NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, WhitespaceSplit, Punctuation, Metaspace,\
                                        CharDelimiterSplit
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from pathlib import Path

#from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

def train_custom_tokenizer(token_model, data_iterator, token_filename, token_dir, vocab_size=30000
                                    , vocab=None, max_input_chars_per_word=None, eos_token=None
                                    , bos_token=None, pad_token=None, mask_token=None
                                    , unk_token=None):
    """
    Building a Tokenizer using HuggingFace library. The pipeline seems to be:

        - Model           : algorithm that tokenizes, it is a mandatory component. There are
                            only 4 models implemented (BPE, Unigram, WordLevel, WordPiece)
        - Normalizer      : some preprocessing that could happen before, but doesn't necessarily
        - Pre-Tokenizer   : splitting the input according to some rules
        - Post-Processing : needing to add some tokens/input after (mostly seems to be eos
                            , bos tokens)
        - Decoder         : certain previous pipeline steps need to be reversed for proper
                            decoding
        - Trainer         : The corresponding training algorithm for the model

    Note : Some pre-processing might need to happen beforehand in previous functions (might
            be easier using pandas)

    Input
        token_model              : algorithm to use for tokenization
        data_iterator            : a python iterator that goes through the data to be used for
                                    training
        token_dir                : directory with tokenizers
        vocab_size               : size of the vocabulary to use
        token_filename           : filename of particular token we want to train. Will overwrite
                                    previously save files.
        vocab                    : models other than BPE can use non-mandatory vocab as input
        max_input_chars_per_word : used for WordPiece

    Output
        tokenizer                : huggingFace Tokenizer object, our fully trainer tokenizer

    """
    special_token_lst = [unk_token, bos_token, eos_token, pad_token, mask_token]

    normalizer_lst = [NFKC()]; pre_tokenizer_lst = [ByteLevel()]; decoder_lst = []

    bos_idx = special_token_lst.index(bos_token); eos_idx = special_token_lst.index(eos_token)

    if token_model == 'BPE':
        model   = BPE(unk_token=unk_token)
        Trainer = BpeTrainer
    elif token_model == 'Unigram':
        model   = Unigram(vocab=vocab)
        Trainer = UnigramTrainer
    elif token_model == 'WordLevel':
        model   = WordLevel(unk_token=unk_token,vocab=vocab)
        Trainer = WordLevelTrainer
    elif token_model == 'WordPiece':
        model   = WordPiece(unk_token=unk_token,vocab=vocab, max_input_chars_per_word=max_input_chars_per_word)
        Trainer = WordPieceTrainer
    else:
        error_msg = f'Error: token_model ({token_model}) not an algorithm in [BPE, Unigram, WordLevel, WordPiece]'
        raise SystemExit(error_msg)

    # instantiation
    tokenizer = Tokenizer(model)

    # trainer
    trainer = Trainer(vocab_size=vocab_size, show_progress=True, special_tokens=special_token_lst)

    # normalizer
    tokenizer.normalizer = normalizers.Sequence( normalizer_lst )

    # pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence( pre_tokenizer_lst )

    # post-processing
    tokenizer.post_processor = processors.TemplateProcessing( single=bos_token+" $A "+eos_token
                                                    #, pair=bos_token+" $A "+eos_token" $B:1 "+eos_token+":1"
                                                    , special_tokens=[(bos_token, bos_idx),(eos_token, eos_idx)]
                                                    )

    # decoder
    if ByteLevel() in pre_tokenizer_lst: tokenizer.decoder = decoders.ByteLevel()
    if Metaspace() in pre_tokenizer_lst: tokenizer.decoder = decoders.Metaspace()
    if token_model == 'WordPiece' : tokenizer.decoder = decoders.WordPiece()

    tokenizer.train_from_iterator(trainer=trainer, iterator=data_iterator)

    if not os.path.exists( token_dir ):
        os.makedirs( token_dir )
    if os.path.exists( token_dir + os.sep + token_filename ):
        print(f"Warning : overwriting previously save tokenizer with same filename ( {token_filename} ).")
    tokenizer.save( token_dir + os.sep + token_filename + '_' + token_model + '.json' )

    # TODO : Should I add PreTrained and Fast Tokenizer here? Seems like it might be appropriate.
    transformer = False; fast = False
    function_from_transformer_todo = None
    if transformer:
        raise SystemExit("HuggingFace transformers library not yet implemented here!")
        if fast: tokenizer = function_from_transformer_todo
        else: tokenizer = function_from_transformer_todo

    return tokenizer


def load_custom_tokenizer(token_dir, token_filename, transformer=False, fast=False):
    """
    Input
        token_dir      : directory with tokenizers saved
        token_filename : trained tokenizer that we want to load
        transformer    : (bool) whether to use HuggingFace transformers library implementation
        fast           : (bool) whether to use HuggingFace transformers fast implementation
    Output
        tokenizer      : tokenizer from Tokenizer class to be passed to rest of algorithm
    """
    tokenizer = Tokenizer.from_file(token_dir + os.sep + token_filename)

    function_from_transformer_todo = None
    if function_from_transformer != None:
        if transformer:
            raise SystemExit("HuggingFace transformers library not yet implemented here!")
            if fast: tokenizer = function_from_transformer_todo
            else: tokenizer = function_from_transformer_todo

    return tokenizer
