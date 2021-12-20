import numpy as np
import torch
from torch.utils.data import DataLoader


class CustomBatch():
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences or max_len. Unclear to me whether this is instantiated
    at every call to dataloader or if it's instantiated along with dataloader.
    For now, all the tensors exist on CPU and are later pushed to the GPU. Needs
    potentially to be changed.
    """

    def __init__(self, data, dim=0, pad_value=0, stack_dim=1, max_len_model=None):
        """
        Input:
            data (dataset)      : a batch of dataset.
            dim (int)           : the dimension to be padded (dimension of time
                                    in sequences)
            max_len_model (int)  : maximum length of sentence, if any
            pad_value (int)      : the value for padding.
            stack_dim (int)      : dimension along which to stack the data tensor.
                                     1 - huggingface, 0 - annotated transformer
        """
        self.dim = dim
        self.padValue = pad_value

        max_len_seq = np.max([x.shape[self.dim] for x in data])
        self.maxLen = np.min([x for x in [max_len_seq, max_len_model]
                              if x is not None])
        # pad according to max_len
        batch = [self.pad_tensor(x[:self.maxLen]) for x in data]
        # stack all, change to dim = 0 for annotated transformer?
        self.src = (torch.stack([x[:-1] for x in batch], dim=stack_dim)).long()
        self.tgt = (torch.stack([x[1:] for x in batch], dim=stack_dim)).long()
        self.src_pad_mask = (self.src == self.padValue)
        #  self.tgt_pad_mask = (self.tgt != self.pad_value).type(torch.int)
        #  ys = torch.LongTensor(map(lambda x: x[1], batch))

    def pad_tensor(self, vec):
        """
        Padding a tensors to the max length in the batch.
        Input:
            vec : tensor to pad
        Output:
            a new tensor padded to 'pad' in dimension 'dim'
        """
        pad_size = list(vec.shape)
        pad_size[self.dim] = self.maxLen - vec.size(self.dim)
        return torch.cat([vec, self.padValue*torch.ones(*pad_size)], dim=self.dim)

    def pin_memory(self):
        self.src = self.src.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


class CustomDataloader():
    def __init__(self, dataset, batch_size, max_len,
                 dim=0,
                 num_workers=2,
                 pin_memory=True,
                 split_train_test_val=(0.7, 0.2, 0.1)):

        split_train_test_val = [np.floor(split_train_test_val[0] * len(dataset)),
                                np.floor(split_train_test_val[1] * len(dataset)),
                                len(dataset) - (np.floor(split_train_test_val[0] * len(dataset)) +
                                                np.floor(split_train_test_val[1] * len(dataset)))
                                ]
        self.dataset_train, self.dataset_test, self.dataset_valid = \
            torch.utils.data.random_split(
                dataset,
                [int(x) for x in split_train_test_val],
                generator=torch.Generator().manual_seed(42)
            )

        tknzr = dataset.transform

        def collate_wrapper(batch):
            return CustomBatch(batch, dim=dim, max_len_model=max_len,
                               pad_value=tknzr.get_vocab()["<pad>"])

        self.train = DataLoader(self.dataset_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=collate_wrapper,
                                pin_memory=pin_memory
                                )
        self.test = DataLoader(self.dataset_test, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers,
                               collate_fn=collate_wrapper,
                               pin_memory=pin_memory
                               )
        self.valid = DataLoader(self.dataset_valid, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers,
                                collate_fn=collate_wrapper,
                                pin_memory=pin_memory
                                )
