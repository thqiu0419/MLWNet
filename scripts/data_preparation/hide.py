# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_gopro

def main():

    create_lmdb_for_gopro()



if __name__ == '__main__':
    main()
