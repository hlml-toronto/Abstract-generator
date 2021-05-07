from src import default
from src.data import download as dl, data_preprocessing as dpp, tokenization as tkn


# download data
filename = dl.arxiv_api( default.RAW_DATA_DIR )
print(f'>> Using {filename} for training <<')

# preprocessing
proc_data = dpp.arxiv_preprocess_abstract(default.RAW_DATA_DIR
                                , default.PROC_DATA_DIR, filename, True )

# convert to list/iterator
data_iter = dpp.arxiv_abstract_iterator( proc_data )
fname_strip_csv = filename[:-4]
print(fname_strip_csv)
tkn.train_custom_tokenizer('BPE', data_iter, fname_strip_csv, default.TOK_DIR
                                , **default.special_token_lst)
