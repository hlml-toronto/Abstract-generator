from torch.utils.data import Dataset
import pandas as pd

class ArxivDataset(Dataset):
    """
    NOTE: Preprocessing can happen in here!

    This Dataset takes the Arxiv data downloaded into a '.csv' files and, when
    called, returns a summary from the list of summaries. You can choose to set
    a transform (generally a tokenizer) to transform what is returned into
    another form of data more suitable (generally a token).
    """
    def __init__(self, csvfile, maxLength=None, transform=None):
        self.data = pd.read_csv(csvfile)
        # last one r'\s+|\\n' seems to be the only one that works
        remove_lst = ['\r\n','\n','\ n',r'\\n',r'\n',r'\s+|\\n']
        self.data.replace(remove_lst,' ',regex=True, inplace=True)
        self.transform = transform  # HuggingFace tokenizer
        self.maxLen = maxLength
        self.get_instance = self.get_instance_pretransform
        if self.transform is not None:
            self.get_instance = self.get_instance_transformed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # uses function as defined by self.get_instance
        return self.get_instance(idx)

    def set_transform(self, transform):
        # can set transform and will change how we get an instance
        self.transform = transform
        self.get_instance = self.get_instance_transformed

    def get_instance_pretransform(self, idx):
        # returns simply the text of summary
        return self.data['summary'][idx]

    def get_instance_transformed(self, idx):
        # once tranform is defined, can get item already transformed
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
    @staticmethod
    def collate_fn(batches: List[Dict[str, int]]):
        return {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batches[0].items()
        }
    """
