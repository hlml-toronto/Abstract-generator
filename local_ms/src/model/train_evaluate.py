import __init__  # For now, needed for all the relative imports

import math
import time

import numpy as np
import torch

from src.model.model_utils import get_batch
from src.settings import BPTT


def train(model, device, train_data, ntokens, optimizer, scheduler, criterion, epoch):
    """
    scheduler: either an int/float (fixed learning rate) or a scheduler torch object
    """
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(BPTT).to(device)

    batch_indices = np.arange(0, train_data.size(0) - 1, BPTT)
    loss_per_batch = 0.0 * batch_indices  # record the training loss for each batch

    for batch, i in enumerate(batch_indices):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != BPTT:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # TODO check if scaling is correct
        loss_per_batch[batch] = loss

        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            if isinstance(scheduler, int) or isinstance(scheduler, float):
                last_lr = scheduler
            else:
                last_lr = scheduler.get_last_lr()[0]

            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:2e} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // BPTT, last_lr,
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return loss_per_batch


def train_version_jeremy(model, dataLoader, device, vocabSize, epoch, optimizer_, scheduler_, criterion_, maxLen=None):
    """
    Training loop that takes batches from dataLoader and pushes them to device
    to train. Will check if they're the same size of maxLen: if shorter, will
    reduces to longest length in batch. then trains according to optimizer,
    criterion and schedule.

    Input
        model (instance)        : model that is being trained
        dataLoader (instance)   : dataloader that batches data into tensors
        optimizer (instance)    : Not sure what type optimizers are
        criterion               :
        device (str)            : gpu or cpu
        maxLen (int)            : maximum sentence length if not None
    Output
        None
    """

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    if maxLen is not None:
        src_mask = model.generate_square_subsequent_mask(maxLen).to(device)
    for i, batch in enumerate(dataLoader):
        #print((batch.src).is_pinned())
        src = (batch.src).to(device); tgt = (batch.tgt).to(device)
        src_pad_mask = (batch.src_pad_mask).to(device)
        #tgt_pad_mask = (batch.tgt_pad_mask).to(device)

        optimizer_.zero_grad()
        if src.size(0) != maxLen:
            src_mask = model.generate_square_subsequent_mask(src.size(0)).to(device)

        output = model(src, src_mask, src_pad_mask.T)
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
                scheduler_.get_last_lr()[0],
                elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return None


def evaluate(eval_model, data_source, device, ntokens, criterion):
    """
    Note: changed pytorch URL code to eval_model from model in a few lines
    """
    eval_model.eval()  # set to: evaluation mode
    total_loss = 0.
    src_mask = eval_model.generate_square_subsequent_mask(BPTT).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, BPTT):
            data, targets = get_batch(data_source, i)
            if data.size(0) != BPTT:
                src_mask = eval_model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def evaluate_version_jeremy(eval_model, dataLoader, device, vocabSize, criterion_, maxLen, nbrSamples):
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
    src_mask = eval_model.generate_square_subsequent_mask(maxLen).to(device)
    with torch.no_grad():
        for batch in dataLoader:
            src = (batch.src).to(device); tgt = (batch.tgt).to(device)
            if src.size(0) != maxLen:
                src_mask = eval_model.generate_square_subsequent_mask(
                    src.size(0)).to(device)
            output = eval_model(src, src_mask)
            output_flat = output.view(-1, vocabSize)
            total_loss += len(src) * criterion_(output_flat,
                                                tgt.reshape(-1) ).item()
    return total_loss / (nbrSamples - 1) # nbrSamples -x-> len(dataLoader)
