from sickle import Sickle
import pandas as pd

BASE = 'http://export.arxiv.org/oai2'
search = Sickle(BASE,max_retries=10)
records = search.ListRecords(
             **{'metadataPrefix': 'arXiv',
             'from': '2021-01-01',
             'until': '2022-01-01',
             'ignore_deleted':False,
             'set':'physics:astro-ph'
            })

col=['title', 'abstract']

master_df = pd.DataFrame(columns=col)

for record in records:
    data_df =  pd.DataFrame([[record.metadata['title'][0], record.metadata['abstract'][0]]],columns=col)
    if master_df.empty:
        master_df = data_df
    else:
        master_df = master_df.append(data_df, ignore_index=True)
