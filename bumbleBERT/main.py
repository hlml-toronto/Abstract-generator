from src import default
import os, torch, time, math
import numpy as np
from src.data import download as dl, data_preprocessing as dpp, tokenization as tkn\
                        , custom_dataset as cd
from torch.utils.data import DataLoader
from src.model.transformer_hf import TransformerModel, PadCollate
#from src.model.transformer import make_gpt_model # imports don't work

# PARAMETERS
maxLen  = 250 # maximumsentence length
bsz     = 3 # batch size
vocabSize = None # None if you want to let tokenizer do its thing
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in torch.nn.TransformerEncoder
nlayers = 2 # the number of torch.nn.TransformerEncoderLayer in torch.nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
tknzerType = 'BPE' # type of tokenizing algorithm
trainTokenizer = False
download = False
nbrResults = 1000
best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download data
filename = dl.arxiv_api( default.RAW_DATA_DIR, max_results=nbrResults )
print(f'>> Using {filename} for training <<')
fnameStrip = filename[:-4] # remove .csv
tknzrFile = default.TOK_DIR + os.sep + fnameStrip + '_' + tknzerType + '.json'

# create dataset
dataset = cd.ArxivDataset(default.RAW_DATA_DIR + os.sep + filename, maxLen)

# create tokenizer
if trainTokenizer:
    _ = tkn.train_custom_tokenizer(tknzerType, dataset, fnameStrip
                                            , default.TOK_DIR
                                            , vocabSize
                                            , **default.special_token_lst)

# load PreTrainedTokenizerFast, for __call__. __call__ not implemented in
# the base Tokenizer class... that sounds silly, but it is what it is
tknzr = tkn.load_tokenizer(tknzrFile, **default.special_token_lst)

# set vocab size to the one of the tokenizer
if vocabSize is None: vocabSize = tknzr.vocab_size
ntokens = len(tknzr.get_vocab()) # the size of vocabulary

print(ntokens,vocabSize)

# set tknzr as the transform
dataset.set_transform( tknzr )

# separate dataset into train, test valid TODO : make into a function
fracTrain, fracTest, fracVal = ( 0.7, 0.2, 0.1)
trainTestVal = [ np.floor(fracTrain*len(dataset))\
                    , np.floor(fracTest*len(dataset))\
                    , len(dataset) - ( np.floor( fracTrain*len(dataset) ) +
                    np.floor( fracTest*len(dataset) ) )
                    ]

trainDataset, testDataset, valDataset =\
        torch.utils.data.random_split(dataset, [int(x) for x in trainTestVal]
                                , generator=torch.Generator().manual_seed(42) )

# create dataloaders
# uses collate function to transform batch to correct dimensions
trainDataLoader = DataLoader(trainDataset, batch_size=bsz, shuffle=True
                                        , collate_fn = PadCollate(dim=0,
                                            maxLen=maxLen,
                                            padValue=tknzr.get_vocab()["<pad>"])
                                        )
valDataLoader = DataLoader(valDataset, batch_size=bsz, shuffle=True
                                        , collate_fn = PadCollate(dim=0,
                                            maxLen=maxLen,
                                            padValue=tknzr.get_vocab()["<pad>"])
                                        )

# training function - same as in hugging face
def train( model, maxLen, dataLoader, nbrSamples, optimizer_, scheduler_
                , criterion_, device_ ):

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(maxLen).to(device_)
    for i, batch in enumerate(dataLoader):
        data = batch[0]; targets = batch[1]
        optimizer_.zero_grad()
        if data.size(0) != maxLen:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        output = model(data, src_mask)
        loss = criterion_(output.view(-1, ntokens), targets.reshape(-1))
        loss.backward()
        torch.torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer_.step()

        total_loss += loss.item()
        log_interval = 200
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, nbrSamples // maxLen,
                            scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# evaluation function outside of training - same as hugging face
def evaluate(eval_model, maxLen, dataLoader, nbrSamples):

    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(maxLen).to(device)
    with torch.no_grad():
        for batch in dataLoader:
            data = batch[0]; targets = batch[1]
            if data.size(0) != maxLen:
                src_mask = model.generate_square_subsequent_mask(
                                                    data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat
                                                , targets.reshape(-1)).item()
    return total_loss / (nbrSamples - 1)

# transformer from huggingface
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

# transformer from illustrated transformer
#model = make_gpt_model(vocabSize, vocabSize, nlayers, emsize, nhid, nhead, dropout)

criterion = torch.nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train( model, maxLen, trainDataLoader, len(trainDataset), optimizer
                , scheduler, criterion, device)
    val_loss = evaluate(model, maxLen, valDataLoader, len(valDataset))
    print('-' * 89)
    print(val_loss)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
                                     # Why is math.exp so large????
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

# decoding a tokenized tensor
"""
print(tknzr.decode(dataset[0]['src']))
nbr = 5
print(dataset[nbr]['src'])
print(dataset[nbr]['trg'])
"""
