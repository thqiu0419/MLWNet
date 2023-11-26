#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 1/03/2023 9:14 pm
# @Author  : 邱天衡
# @FileName: run_once.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/Mr_Clutch
import time

import torch
from torch import nn
from network import SRNDeblurNet
import numpy as np




if __name__ == '__main__':
    timings = np.zeros((2000, 1))
    for i in range(2000):
        x3 = torch.zeros(1, 3, 160, 160).cuda()
        # down_m = Downsample(32)
        # o = down_m(x3)
        # print(o.shape)
        # up_m = Upsample(32)
        # o = up_m(x3)
        # print(o.shape)
        x2 = torch.zeros(1, 3, 320, 320).cuda()
        x1 = torch.zeros(1, 3, 640, 640).cuda()
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        model = SRNDeblurNet().cuda()
        starter.record()
        out = model(x1, x2, x3)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time
        print("time consume", curr_time)
        for o in out:
            print(o.shape)
    print(np.mean(timings))
