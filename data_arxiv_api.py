import urllib.request
import feedparser

import csv


if __name__ == '__main__':
    # Base api query url
    base_url = 'http://export.arxiv.org/api/query?'

    # Search parameters
    search_query = 'all:electron' # search for electron in all fields
    start = 0                     # retreive the first 5 results
    max_results = 10

    query = 'search_query=%s&start=%i&max_results=%i' % (search_query,
                                                         start,
                                                         max_results)

    # perform a GET request using the base_url and query
    response = urllib.request.urlopen(base_url+query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)

    # # print out feed information
    # print('Feed title: %s' % feed.feed.title)
    # print('Feed last updated: %s' % feed.feed.updated)

    # # print opensearch metadata
    # print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
    # print('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
    # print('startIndex for this query: %s'   % feed.feed.opensearch_startindex)

    abstract_list = []

    # Run through each entry, and print out information
    for entry in feed.entries:
        #print(entry.keys())
        pc = entry.arxiv_primary_category['term']
        tags = [entry.tags[i].term for i in range(len(entry.tags))]
        data_row = [
            entry.id,
            entry.published_parsed,
            entry.published,
            entry.title,
            pc,
            tags,
            entry.summary]
        abstract_list.append(data_row)

    fields = ['id', 'published_parsed', 'published', 'title', 'arxiv_primary_category', 'tags', 'summary']
    with open('raw_arxiv_%d.csv' % max_results, mode='w') as csv_file:
        write = csv.writer(csv_file, lineterminator='\n')
        write.writerow(fields)
        write.writerows(abstract_list)
