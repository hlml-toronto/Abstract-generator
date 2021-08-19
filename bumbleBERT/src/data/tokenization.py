import os
import numpy as np
from pathlib import Path

from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders\
                                , processors
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import NFD, NFKD, NFC, NFKC, Lowercase, StripAccents
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, WhitespaceSplit\
                                                , Punctuation, Metaspace\
                                                , CharDelimiterSplit
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer\
                                            , WordLevelTrainer
from transformers import PreTrainedTokenizerFast

#from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

def train_custom_tokenizer(token_model, dataset, token_filename, token_dir
                                    , vocab_size, vocab=None
                                    , max_input_chars_per_word=None
                                    , eos_token=None, bos_token=None
                                    , pad_token=None, mask_token=None
                                    , unk_token=None):
    """
    Building a Tokenizer using HuggingFace library. The pipeline seems to be:

        - Model           : algorithm that tokenizes, it is a mandatory
                            component. There are only 4 models implemented
                            (BPE, Unigram, WordLevel, WordPiece)
        - Normalizer      : some preprocessing that could happen before, but
                            doesn't necessarily have to
        - Pre-Tokenizer   : splitting the input according to some rules
        - Post-Processing : needing to add some tokens/input after (mostly seems
                            to be eos, bos tokens)
        - Decoder         : certain previous pipeline steps need to be reversed
                            for proper decoding
        - Trainer         : The corresponding training algorithm for the model

    Note : Some pre-processing might need to happen beforehand in previous
            functions (might be easier using pandas before)

    Input
        token_model (str)        : algorithm to use for tokenization
        dataset (class)          : a python iterator that goes through the data
                                    to be used for training
        token_dir (str)          : directory with tokenizers
        vocab_size (int)         : size of the vocabulary to use
        token_filename (str)     : filename of particular token we want to
                                    train. Will overwrite previously save files.
        vocab (list of str)      : models other than BPE can use non-mandatory
                                    vocab as input
        max_input_chars_per_word : used for WordPiece

    Output
        tokenizer                : huggingFace Tokenizer object, our fully
                                    trainer tokenizer

    """
    special_token_lst = [pad_token, bos_token, eos_token, mask_token, unk_token]

    #NFKC
    normalizer_lst = []; pre_tokenizer_lst = [Whitespace, ByteLevel];
    decoder_lst = []

    bos_idx = special_token_lst.index(bos_token);
    eos_idx = special_token_lst.index(eos_token)

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
        model   = WordPiece(unk_token=unk_token,vocab=vocab
                            , max_input_chars_per_word=max_input_chars_per_word)
        Trainer = WordPieceTrainer
    else:
        error_msg = f'Error: token_model ({token_model}) not an algorithm in\
                        [BPE, Unigram, WordLevel, WordPiece]'
        raise SystemExit(error_msg)

    # instantiation
    tokenizer = Tokenizer(model)

    # trainer
    if vocab_size == None:
        trainer = Trainer(show_progress=True, special_tokens=special_token_lst)
    else:
        trainer = Trainer(vocab_size=vocab_size, show_progress=True
                                            , special_tokens=special_token_lst)

    # normalizer
    tokenizer.normalizer = normalizers.Sequence(
                                    [fcn() for fcn in normalizer_lst] )

    # pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
                                    [fcn() for fcn in pre_tokenizer_lst] )

    # post-processing
    tokenizer.post_processor = processors.TemplateProcessing(
                    single=bos_token+" $A "+eos_token
                    #, pair=bos_token+" $A "+eos_token" $B:1 "+eos_token+":1"
                    , special_tokens=[(bos_token, bos_idx),(eos_token, eos_idx)]
                    )

    # decoder
    if ByteLevel in pre_tokenizer_lst:
        tokenizer.decoder = decoders.ByteLevel()
    if Metaspace in pre_tokenizer_lst: tokenizer.decoder = decoders.Metaspace()
    if token_model == 'WordPiece' : tokenizer.decoder = decoders.WordPiece()

    # creating iterator
    def batch_iterator():
        for i in np.arange(0,len(dataset)):
            yield dataset[i]

    # train call
    tokenizer.train_from_iterator(trainer=trainer, iterator=batch_iterator()
                                    , length=len(dataset))

    # save file
    if not os.path.exists( token_dir ):
        os.makedirs( token_dir )
    tknzrfile = token_dir + os.sep + token_filename +'_'+ token_model + '.json'
    if os.path.exists( tknzrfile ):
        print(f"Warning : overwriting previously save tokenizer with\
                        same filename ( {token_filename} ).")
    tokenizer.save( tknzrfile )

    return tokenizer


def load_tokenizer(tknzrFile, eos_token=None, bos_token=None
                            , pad_token=None, mask_token=None, unk_token=None):
    """
    Interestingly, HuggingFace does not allow the base tokenizer to be called.
    This is a bizarre choice, but accordingly we have to look for something else
    , which is why I use the PreTrainedTokenizerFast to wrap the base tokenizer.
    Written in Rust, it's faster than the base tokenizer class, but also lets
    you call the tokenizer as tknzr('text to be tokenized').

    Input
        tknzrFile (str) : .json file of the tokenizer trained previously
        *_tokens (str)  : tokens that are to be used in the corresponding context
                            Some of them are not implemented yet...
    Output
        tknzr     : tokenizer as PreTrainedTokenizerFast class to be passed on
    """
    tknzr = PreTrainedTokenizerFast(tokenizer_file=tknzrFile)
    tknzr.pad_token = pad_token
    tknzr.mask_token = mask_token

    return tknzr
