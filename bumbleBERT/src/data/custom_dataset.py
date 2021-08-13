from torch.utils.data import Dataset
import pandas as pd
import torch

# TODO create CustomDataset with self.transform and correct transform functions

class ArxivDataset(Dataset):
    """
    This Dataset takes the Arxiv data downloaded into a '.csv' files and, when
    called, returns a summary from the list of summaries. You can choose to set
    a transform (generally a tokenizer) to transform what is returned into
    another form of data more suitable (generally a tensor of tokens).

    To change what a sample is, need only change the method :
        get_instance_pretransform
    """
    def __init__(self, csvfile, device=None, maxLength=None, transform=None):
        """
        Loads all of the data and cleans it up slightly. Might wants to call it
        something else instead of 'raw', but for now will do. Sets a transform
        of the data (which is how data is presented if fetched).

        Input
            csvfile (str)               : csv file containing data
            device (str, opt)                : Unused for now.
            maxLength (int, opt)        : Unused for now.
            transform (function, opt)   : a transform of the data if one already
                                            exists
        """
        self.data_raw = pd.read_csv(csvfile)
        # last one r'\s+|\\n' seems to be the only one that works
        remove_lst = ['\r\n','\n','\ n',r'\\n',r'\n',r'\s+|\\n']
        self.data_raw.replace(remove_lst,' ',regex=True, inplace=True)
        self.transform = transform  # HuggingFace tokenizer
        self.maxLen = maxLength
        self.get_instance = self.get_instance_pretransform # how to get a sample
                                                        # from the dataset
        if self.transform is not None:
            self.get_instance = self.get_instance_transformed

    def __len__(self):
        # gives number of samples in dataset
        return len(self.data_raw)

    def __getitem__(self, idx):
        # uses function as defined by self.get_instance
        return self.get_instance(idx)

    def set_transform(self, transform):
        # can set transform and will change how we get an instance
        self.transform = transform
        self.get_instance = self.get_instance_transformed

    def get_instance_pretransform(self, idx):
        # returns some form of the text which will be our sample
        return self.data_raw['summary'][idx]

    def get_instance_transformed(self, idx):
        """
        Once tranform is defined, can get item already transformed

        Input
            idx (int) : index of sample to fetch and transform
        Return
            transformed sample
        """

        instance = self.get_instance_pretransform(idx)
        # tokenize on-the-fly
        instance = self.transform( instance
                                    #, max_length=self.maxLen+1
                                    #, padding='max_length'
                                    #, truncation=True
                                    , return_tensors='pt'
                                    )

        return instance['input_ids'][0]

    """
    # This is some code that was copied over from some class I based this on.
    # I'm not entirely sure what it was used for, but could be interesting to
    # figure out.
    @staticmethod
    def collate_fn(batches: List[Dict[str, int]]):
        return {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batches[0].items()
        }
    """


class WikiTextCustom(Dataset):
    def __init__(self, csvfile):
        self.dataRaw = pd.read_csv(csvfile)
        self.arraySamples = []
        for i in np.arange(0,len(self.dataRaw)//bptt):
            self.arraySamples[i] = self.dataRaw[i*bptt:(i+1)*bptt]

    def __init__(self):
        return 0


class ArxivDatasetMemoryOnGPU(Dataset):
    """
    NOTE: Copy of above class that I wanted to change to have samples on GPU,
            however I think this won't work because of the dataloader (it should
            probably happen at the level of batches)

    This Dataset takes the Arxiv data downloaded into a '.csv' files and, when
    called, returns a summary from the list of summaries. You can choose to set
    a transform (generally a tokenizer) to transform what is returned into
    another form of data more suitable (generally a token).
    """
    def __init__(self, csvfile, maxLength=None,transform=None, device=None):
        self.dataRaw = pd.read_csv(csvfile)
        # last one r'\s+|\\n' seems to be the only one that works
        remove_lst = ['\r\n','\n','\ n',r'\\n',r'\n',r'\s+|\\n']
        self.dataRaw.replace(remove_lst,' ',regex=True, inplace=True)
        self.transform = transform  # HuggingFace tokenizer
        self.maxLen = maxLength
        self.device = device
        self.get_instance = self.get_instance_pretransform
        self.dataRawSummary = self.dataRaw['summary']
        if self.transform is not None:
            self.get_instance = self.get_instance_transformed

    def __len__(self):
        return len(self.dataRaw)

    def __getitem__(self, idx):
        # uses function as defined by self.get_instance
        return self.get_instance(idx)

    def set_transform(self, transform):
        # can set transform and will change how we get an instance
        self.transform = transform
        self.get_instance_transformed
        self.dataToken = ((self.dataRawSummary).transform( self.transform
                                    #, max_length=self.maxLen+1
                                    #, padding='max_length'
                                    #, truncation=True
                                    #, return_tensors='pt'
                                    ).tolist())#.to(device)
        self.dataToken = [x['input_ids'] for x in self.dataToken]
        print(self.dataToken)
        self.dataToken = torch.tensor(self.dataToken).to(self.device)
        print(self.dataToken)

    def get_instance_pretransform(self, idx):
        # returns simply the text of summary
        return self.dataRawSummary[idx]

    def get_instance_transformed(self, idx):
        # once tranform is defined, can get item already transformed

        return self.dataToken[idx]

    """
    @staticmethod
    def collate_fn(batches: List[Dict[str, int]]):
        return {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batches[0].items()
        }
    """
