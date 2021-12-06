import __init__  # For now, needed for all the relative imports

import math
import time

import numpy as np
import torch

from src.model.model_utils import get_batch
from settings import BPTT


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
