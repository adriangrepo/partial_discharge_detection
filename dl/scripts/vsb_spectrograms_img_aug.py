
# coding: utf-8

import logging
import time
import datetime
import uuid
import pyageng
from scipy import fftpack, signal
from skimage import util
import copy
from multiprocessing import Process
import multiprocessing as mp
import PIL.Image as pil_image

import pyarrow.parquet as pq
import os
import seaborn as sns
import numpy.fft as fft

import matplotlib
import matplotlib.pyplot as plt

from fastai import *
from fastai.tabular import *
from fastai.utils import *

import fastai
print(fastai.__version__)



IMG_SIZE=300


DATE = datetime.datetime.today().strftime('%Y%m%d')
UID=str(uuid.uuid4())[:8]

print(f'DATE: {DATE}')
print(f'uID: {UID}')


# ## Data preparation

path = Path('../../input/')
train_path = path/'train/'
test_path = path/'test/'
train_aug_path = path/'train_300_bp_500Hz-40MHz_aug/'
#note used
test_aug_path = path/'test_aug/'



# 800000 samples every 20 millisecond



def read_meta():
    meta_train = pd.read_csv(path/'metadata_train.csv')
    features = meta_train.columns
    meta_test = pd.read_csv(path/'metadata_test.csv')
    return meta_train, features, meta_test


def flip_image(img_name, in_path, out_path):
    img = pil_image.open(in_path/f'{img_name}.jpg')
    rotated_image = img.transpose(pil_image.FLIP_LEFT_RIGHT)
    rotated_image.save(out_path/f'{img_name}_flip_lr.jpg', format='JPEG', subsampling=0, quality=100)
    #rotated_image = img.transpose(pil_image.FLIP_TOP_BOTTOM)
    #rotated_image.save(out_path/f'{img_name}_flip_tb.jpg', format='JPEG', subsampling=0, quality=100)
    img.close()
    rotated_image.close()
    

        

if __name__ == '__main__':
    meta_train, features, meta_test = read_meta()
    train_imgs = meta_train['signal_id'].values
    print('train')
    for img_name in train_imgs:
        flip_image(img_name, train_path, train_aug_path)
    test_imgs = meta_test['signal_id'].values
    '''
    #no need to extra test images
    for img_name in test_imgs:
        flip_image(img_name, test_path, test_aug_path)
    '''

    


