import csv, os
import pandas as pd
import csv
import src.default as default

def arxiv_abstract_preprocess(raw_data_dir, proc_data_dir, filename, save = False ):
    """
    For now, simply removes newline. Might make more elaborate eventually

    Input
        raw_data_dir  : directory of raw data
        proc_data_dir : directory of processed data
        filename      : which file to process
        save          : whether to save processed data.
    Output
        processed_data : transformed data
    """
    # read data as panda dataframe
    raw_data = pd.read_csv(raw_data_dir + os.sep + filename)

    # remove newlines
    remove_lst = ['\r\n','\n','\ n']
    remove = lambda x: ' '.join([item for item in x.split() if item not in remove_lst])
    #raw_data['summary'].apply(remove)
    raw_data.replace(['\r\n','\n','\ n'],' ',regex=True)

    raw_data.to_csv(proc_data_dir +os.sep + filename)

    return raw_data

def arxiv_abstract_load():

    return 0

def arxiv_abstract_iterator( processed_data ):
    return processed_data['summary'].to_list()

def wiki_preprocess():

    return 0

def wiki_iterator( file_dir ):
    files = [default.RAW_DATA_DIR + os.sep + file_dir + os.sep + 'wiki.%s.raw'
                % a for a in ["test", "train", "valid"]]

    return files

class DataLoading():
    def __init__(self, listFiles):
        if isinstance(filenames, list):
            self.isList = True
        else: self.isList = False
        self.filenames = filenames

    def process():

        return 0

    def processed_iterator():
        """
        must return an iterator
        """
        default.PROC_DATA_DIR

        return 0
