import time
# training function - same as in hugging face
def train( model, maxLen, dataLoader, device, optimizer_, scheduler_, criterion_ ):
    """
    Training loop that takes batches from dataLoader and pushes them to device
    to train. Will check if they're the same size of maxLen: if shorter, will
    reduces to longest length in batch. then trains according to optimizer,
    criterion and schedule.

    Input
        model (instance)        : model that is being trained
        maxLen (int)            : maximum sentence length
        dataLoader (instance)   : dataloader that batches data into tensors
        optimizer (instance)    : Not sure what type optimizers are
        criterion               :
        device (str)            : gpu or cpu
    Output
        None
    """

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(maxLen).to(device_)
    for i, batch in enumerate(dataLoader):
        #print((batch.src).is_pinned())
        src = (batch.src).to(device); tgt = (batch.tgt).to(device)

        optimizer_.zero_grad()
        if src.size(0) != maxLen:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)

        output = model(src, src_mask)
        loss = criterion_(output.view(-1, vocabSize), tgt.reshape(-1))
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
                    epoch, i, len(dataLoader),
                            scheduler.get_last_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# evaluation function outside of training - same as hugging face
def evaluate(eval_model, maxLen, dataLoader, nbrSamples):
    """
    Takes a trained model, puts it in evaluation mode to see how well it
    performs on another set of data.

    Input
        eval_model (instance)   : model to be evaluated
        maxLen (int)            : maximum length possible/trained on
        dataLoader (instance)   : dataloader of the dataset that is evaluate on
        nbrSamples (int)        : Supposed to be number of samples, not sure I need
    Output
        loss of evaluated set
    """
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(maxLen).to(device)
    with torch.no_grad():
        for batch in dataLoader:
            src = (batch.src).to(device); tgt = (batch.tgt).to(device)
            if src.size(0) != maxLen:
                src_mask = model.generate_square_subsequent_mask(
                                                    src.size(0)).to(device)
            output = eval_model(src, src_mask)
            output_flat = output.view(-1, vocabSize)
            print("length of src :",len(src))
            total_loss += len(src) * criterion(output_flat
                                                , tgt.reshape(-1)).item()
    return total_loss / (nbrSamples - 1) # nbrSamples -x-> len(dataLoader)