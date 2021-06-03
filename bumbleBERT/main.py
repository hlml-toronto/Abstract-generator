from src import default
import os, torch, time
import numpy as np
from src.data import download as dl, data_preprocessing as dpp, tokenization as tkn\
                        , custom_dataset as cd
from torch.utils.data import DataLoader
from src.model.transformer_hf import TransformerModel

# PARAMETERS
maxLen  = 250 # maximumsentence length
bsz     = 3 # batch size
vocabSize = None # None if you want to let tokenizer do its thing
emsize = 200 # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
tknzerType = 'BPE' # type of tokenizing algorithm

# download data
filename = dl.arxiv_api( default.RAW_DATA_DIR )
print(f'>> Using {filename} for training <<')
fnameStrip = filename[:-4] # remove .csv
tknzrFile = default.TOK_DIR + os.sep + fnameStrip + '_' + tknzerType + '.json'

# create dataset
dataset = cd.ArxivDataset(default.RAW_DATA_DIR + os.sep + filename, maxLen)

# create tokenizer
_ = tkn.train_custom_tokenizer(tknzerType, dataset, fnameStrip
                                            , default.TOK_DIR
                                            , vocabSize
                                            , **default.special_token_lst)

# load PreTrainedTokenizerFast, for __call__. __call__ not implemented in
# the base Tokenizer class... that sounds silly, but it is what it is
tknzr = tkn.load_tokenizer(tknzrFile, **default.special_token_lst)

# set vocab size to the one of the tokenizer
if vocabSize is None: vocabSize = tknzr.vocab_size

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

# create data_loader
trainDataLoader = DataLoader(trainDataset, batch_size=bsz, shuffle=True
                                            #, collate_fn=lambda x: x
                                            )

#print("Trying second version")
#print(tknzr.decode(dataset[0]['src']))
nbr = 5
print(dataset[nbr]['src'])
print(dataset[nbr]['trg'])
"""
for batch in trainDataLoader:
    print('hue',len(batch))
    print(batch)
"""
criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

"""
def train( model, dataLoader, optimizer_, scheduler_, device_ ):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(maxLen).to(device)
    for batch in dataLoader:
        data = batch['input_ids'], targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

best_val_loss = float("inf")
epochs = 3 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train( model )
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()
"""
