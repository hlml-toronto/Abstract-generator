import csv, os
import pandas as pd

def arxiv_preprocess_abstract(raw_data_dir, proc_data_dir, filename, save = False ):
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
    remove_lst = ['\r\n']
    remove = lambda x: ' '.join([item for item in x.split() if item not in remove_lst])
    processed_data = raw_data['summary'].apply(remove)

    raw_data.to_csv(proc_data_dir +os.sep + filename)

    return raw_data

def arxiv_abstract_iterator( processed_data ):

    return iter( processed_data.summary.to_list() )
