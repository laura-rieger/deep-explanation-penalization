
import csv
import numpy as np
from tqdm import tqdm
from scipy.misc import imresize
import scipy.misc
from shutil import copyfile
from os.path import join as oj
img_path = "./not_cancer"
meta_file = "ISIC_y.csv"
list_of_meta = []
from PIL import Image
with open(meta_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        list_of_meta.append(row)    
#%%
list_mal_files = []
for line in list_of_meta[1:]:
    if line[1] == 'benign':
        list_mal_files.append(line[0] + ".jpg")

#%%
for i,file_name in tqdm(enumerate(list_mal_files)):
    if i%500 ==0:
        print(i)
    try:
        img = Image.open(oj(img_path, file_name))
        test = np.asarray(img)
        if test.shape != (299,299, 3):
            test_new = imresize(test, (299, 299, 3))
            scipy.misc.imsave(oj(img_path, file_name), test_new)
    except:
        print(file_name)
    