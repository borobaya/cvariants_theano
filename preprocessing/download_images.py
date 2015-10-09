from __future__ import print_function
import os
import sys
import numpy as np
from urllib import urlopen
import yaml
from cStringIO import StringIO
from PIL import Image

train_ratio = 0.8
minsize = 256

def save_product_image(path, i, j, url):
    # Set path and filename
    filename = "%02d_%02d.jpg" % (i, j)
    dest = path + filename # Includes start of filename

    try:
        # Download image
        f = StringIO(urlopen(url).read())
        img = Image.open(f)

        # Resize file to appropriate dimensions
        ratio = float(img.size[0])/float(img.size[1]) \
                if img.size[0]>img.size[1] \
                else float(img.size[1])/float(img.size[0])
        size = int(minsize * ratio)
        img.thumbnail((size, size), Image.ANTIALIAS)

        # Save file
        img.save(dest)
    except Exception as e:
        print("Exception raised for [%d, %d, %d, %s]" % (index, i, j, url))

if __name__=="__main__":
    with open('paths.yaml', 'r') as f:
        paths = yaml.load(f)

    train_img_dir = paths['train_img_dir']
    val_img_dir = paths['val_img_dir']

    urls = np.load("image_urls.npy")
    val_index_start = int(train_ratio * int(urls[-1][0]))

    count = 0
    for index, i, j, url in urls:
        index, i, j = int(index), int(i), int(j)

        # Create folder for the group of colour variants
        DIR = train_img_dir if index<val_index_start else val_img_dir
        superpath = DIR + str("%05d" % int(index/1500)) + "/"
        path = superpath + str("%04d" % int(index%1500)) + "/" # This is the path for each group of colour variants
        if not os.path.exists(path):
            os.makedirs(path)

        save_product_image(path, i, j, url)
        
        count += 1
        if count%10==0:
            sys.stdout.write("Downloaded %d images\r" % count)
            sys.stdout.flush()

