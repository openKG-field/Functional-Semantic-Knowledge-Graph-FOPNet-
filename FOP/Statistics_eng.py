# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/21
# @LastUpdateTime: 2021/6/21
# @Author: Yingtong Hu

"""

Statistic Methods: Mean, Mode, Medium...

"""

import numpy as np


def mean(data):
    meanArr = []
    for numArr in data:
        meanArr.append(np.mean(numArr))
    return meanArr


def median(data):
    medianArr = []
    for numArr in data:
        medianArr.append(np.median(numArr))
    return medianArr


def mode(data):
    modeArr = []
    for numArr in data:
        tempArr = [float(format(i, '.1f')) for i in numArr]
        d = {}
        for i in tempArr:
            d[i] = tempArr.count(i)
        try:
            modeArr.append(max(d, key=d.get))
        except:
            modeArr.append(0)
    return modeArr


def std(data):
    stdArr = []
    for numArr in data:
        stdArr.append(np.std(numArr))
    return stdArr
