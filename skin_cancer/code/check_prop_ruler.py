# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 00:00:55 2019

@author: lauri
"""
import csv
import numpy as np
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from shutil import copyfile
from os.path import join as oj
from PIL import Image
from skimage.color import rgb2gray
from tqdm import tqdm
def has_line(img):
    bw_img = rgb2gray(img) #(img.sum(axis = 2)/ img.sum(axis=2).max())
    lines = probabilistic_hough_line(canny(bw_img, sigma=2), threshold=40, line_length=200,
                                 line_gap=10)
    lines = [line for line in lines if line[0][0] != line[1][0] and line[0][1] != line[1][1]  ]
    
    return len(lines) >0
    
    
    
meta_file = "ISIC_y.csv"
mal_folder = "cancer"
list_of_meta = []
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list_of_meta.append(row)    
#%%
list_mal_files = []
for line in list_of_meta[1:]:
    if line[1] == 'malignant':
        list_mal_files.append(line[0] + ".jpg")

#%%
num_lines = 0
num_total = 0
for i,file_name in tqdm(enumerate(list_mal_files)):

    try:
        img = Image.open(oj(mal_folder, file_name))
        num_total +=1
        if has_line(np.asarray(img)):
            
            num_lines +=1
            
    except:
        pass
print("Cancer" , num_lines/num_total)

# not cancer now
mal_folder = "not_cancer"
list_of_meta = []
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list_of_meta.append(row)    
#%%
list_mal_files = []
for line in list_of_meta[1:]:
    if line[1] == 'benign':
        list_mal_files.append(line[0] + ".jpg")

num_lines = 0
num_total = 0
for i,file_name in tqdm(enumerate(list_mal_files)):
    try:
        img = Image.open(oj(mal_folder, file_name))
        num_total +=1
        if has_line(np.asarray(img)):
            num_lines +=1
    except:
        pass
print("Cnot ancer" , num_lines/num_total)