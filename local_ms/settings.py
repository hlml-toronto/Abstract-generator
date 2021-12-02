import os

DIR_DATA = 'data'
DIR_TOKENIZERS = 'tokenizers'
DIR_MODELS = 'models'

DEFAULT_DATASET = 'raw_arxiv_10.csv'

# determines sequence length for get_batch(); used elsewhere e.g. train()
BPTT = 35  # constant used by model.py and training notebook (acts as a max token context length)

for core_dir in [DIR_DATA, DIR_TOKENIZERS, DIR_MODELS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)
