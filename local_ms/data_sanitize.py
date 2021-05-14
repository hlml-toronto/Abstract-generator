import csv
import os

from settings import DIR_DATA, DEFAULT_DATASET


def read_data_raw():
    """
    Returns:
         - fields: list of headings for each data column
         - rows:   list of lists, each containing the data for given row
    """
    datapath = DIR_DATA + os.sep + DEFAULT_DATASET
    with open(datapath, mode='r') as csv_file:
        csvdata = list(csv.reader(csv_file))
        fields = csvdata[0]
        datalines = csvdata[1:]
    return fields, datalines


def clean_list_of_abstracts(to_remove=('\n')):
    """
    TODO cleaning, dealing with strange characters and punctuation
    Uses read_data_raw() to return a list of abstracts from the dataset
    Cleaning options:
        - remove newlines (maybe we should keep them so the LM learns the line spacing too...)
    Args:
        to_remove: tuple of strings to be replaced by a space ' '
    """
    fields, datalines = read_data_raw()
    assert fields[-1] == 'summary'
    ndata = len(datalines)
    list_of_abstracts = [0] * ndata
    for idx in range(ndata):
        abstract = datalines[idx][-1]
        for removal in to_remove:
            abstract = abstract.replace(removal, ' ')
        list_of_abstracts[idx] = abstract

    return list_of_abstracts


if __name__ == '__main__':
    list_of_abstracts = clean_list_of_abstracts()
    abstract_idx = 0
    print('Example of abstract entry %d:' % abstract_idx)
    print(list_of_abstracts[abstract_idx])
    print(list_of_abstracts[abstract_idx].split(' '))
