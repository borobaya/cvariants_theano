from __future__ import print_function
import os
import sqlalchemy
import numpy as np
import pandas as pd
from urllib import urlopen
from cStringIO import StringIO

p = "" # Parts removed for security
connection_string = "postgresql+psycopg2://data_team:"+p+"@" # Parts removed for security

engine = sqlalchemy.create_engine(connection_string)

def data_mirror(sql):
    result = pd.read_sql_query(sql, engine)
    return result

def get_image_urls(retailer_id, product_id, limit=10):
    data = data_mirror("""
    select i.*
    from api_product p
    inner join api_link l
    on l.product_id = p.id
    inner join api_link_images li
    on li.link_id = l.id
    inner join api_image i
    on li.image_id = i.id
    where p.id = '{i}' and l.retailer_id = '{r}'
    limit {l}
    """.format(i=product_id, r=retailer_id, l=limit))

    juices = data.juice
    phashes = data.phash
    data['url'] = '' + juices # Parts removed for security
    urls = dict(zip(data.phash.values, data.url.values))

    return list(urls.values())

def load_ids():
    merged = pd.read_csv('merge_ids.csv', header=None, names=['retailer_id', 'product_code', 'product_ids'])
    merged.product_ids = merged.product_ids.str.strip('][]').str.split()
    return merged

if __name__=="__main__":
    df = load_ids()
    urls = []

    for index, row in df.iterrows():
        retailer_id = row.retailer_id
        product_ids = row.product_ids
        
        for i, product_id in enumerate(product_ids):
            image_urls = get_image_urls(retailer_id, product_id)

            for j, url in enumerate(image_urls):
                row = (index, i, j, url)
                print(row)
                urls.append(row)

        if index%100==0:
            print("At index", index)

    np.save("image_urls", urls)
        
