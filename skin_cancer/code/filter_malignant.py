# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:48:46 2019

@author: lauri
"""

import csv
img_path = "./ISICArchive"
meta_file = "ISIC_y.csv"
mal_folder = "good_folder"
list_of_meta = []
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list_of_meta.append(row)    
#%%
list_mal_files = []
for line in list_of_meta:
    if line[1] == 'benign':
        list_mal_files.append(line[0] + ".jpg")
from shutil import copyfile
from os.path import join as oj
#%%
for file_name in list_mal_files:
    try:
    
        copyfile(oj(img_path, file_name),oj (mal_folder, file_name))
    except:
        print(file_name)