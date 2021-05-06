import re
import csv, os
import spacy
import pandas as pd
from spacy.tokenizer import Tokenizer
from tokenizers import BertWordPieceTokenizer
from spacy.tokens import Doc

DATA_DIR    = "datasets"
filename    = "raw_arxiv_10.csv"

raw_data = pd.read_csv(DATA_DIR + os.sep + filename)

remove_lst = ['\r\n']
remove = lambda x: ' '.join([item for item in x.split() if item not in remove_lst])
raw_data['summary'] = raw_data['summary'].apply(remove)

ex_abstract = raw_data['summary'][3]

#### OPTION 1 ####

special_cases = {"\\": [{"ORTH": "\\"}]}
prefix_re = re.compile(r'''^[\$"'\{\(\[]''')
suffix_re = re.compile(r'''[\$"'\}\)\]]$''')
infix_re = re.compile(r'''[-~_/\{\}\[\]\\]''')
simple_url_re = re.compile(r'''^https?://''')

def custom_tokenizer(nlp):
    return Tokenizer(nlp.vocab, rules=special_cases,
                                prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                url_match=simple_url_re.match)

nlp = spacy.load( "en_core_web_sm" )
nlp.tokenizer = custom_tokenizer( nlp )
doc = nlp( ex_abstract )
print([token.text for token in doc]) # ['hello', '-', 'world.', ':)']
print(''.join(token.text_with_ws for token in doc))
# TODO : Is there a way to give it a list of tokens and have it recreate the

"""
### OPTION 2 ####

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

class BertTokenizer:
    def __init__(self, vocab, vocab_file, lowercase=True):
        self.vocab = vocab
        self._tokenizer = BertWordPieceTokenizer(vocab_file, lowercase=lowercase)

    def __call__(self, text):
        tokens = self._tokenizer.encode(text)
        words = []
        spaces = []
        for i, (text, (start, end)) in enumerate(zip(tokens.tokens, tokens.offsets)):
            words.append(text)
            if i < len(tokens.tokens) - 1:
                # If next start != current end we assume a space in between
                next_start, next_end = tokens.offsets[i + 1]
                spaces.append(next_start > end)
            else:
                spaces.append(True)
        return Doc(self.vocab, words=words, spaces=spaces)



txt_file = ""
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab, )
doc = nlp( ex_abstract )
print([token.text for token in doc])
"""
