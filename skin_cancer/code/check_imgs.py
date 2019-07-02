# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:19:36 2019

@author: lauri
"""

img_path = "./ISICArchive"
meta_file = "meta.csv"
#%%
import csv
list_of_meta = []
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    
    for row in spamreader:
        list_of_meta.append(row)    

#%%
list_of_meta = [x for x in list_of_meta if len(x) >0]
#%%
list_of_class = [(x[0], x[3]) for x in list_of_meta[1:]]
#%%
with open('ISIC_y.csv', mode='w', newline='') as out_file:
    csv_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for line in list_of_class:
        csv_writer.writerow(line)
