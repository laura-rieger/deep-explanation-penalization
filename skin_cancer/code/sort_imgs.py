# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:48:46 2019

@author: lauri
"""

import csv
import numpy as np
from scipy.misc import imresize
import scipy.misc
img_path = "./ISICArchive"
meta_file = "ISIC_y.csv"
mal_folder = "mal_folder"
list_of_meta = []
from PIL import Image
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list_of_meta.append(row)    
#%%
list_mal_files = []
for line in list_of_meta[1:]:
    list_mal_files.append(line[0] + ".jpg")
from shutil import copyfile
from os.path import join as oj
#%%
for i,file_name in enumerate(list_mal_files):
    if i%500 ==0:
        print(i)
        try:
            img = Image.open(oj(img_path, file_name))
            test = np.asarray(img)
            if test.shape != (450,600, 3):
                test_new = imresize(test, (450, 600, 3))
                scipy.misc.imsave(oj(img_path, file_name), test_new)
        except:
            print(file_name)
    