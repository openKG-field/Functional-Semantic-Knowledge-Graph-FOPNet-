# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/21
# @LastUpdateTime: 2021/6/24
# @Author: Yingtong Hu

"""

Top level file for Patent Matching Project

"""
import time

from FOP_eng import init, main, common_main
from MAP_MRR_NDCG_eng import MAP_MRR_NDCG_main

SAOExtracted = '../FOPdata/data_FOP_eng/extraction_method_1_eng.txt'
WordVector = '../FOPdata/data_FOP_eng/vector.txt'
# WordVector = 'vector_eng/vector_common_new.txt'
vectorLen = 300
#
init(SAOExtracted, WordVector, vectorLen, 0, 1, 1)
# common_main('arrayipc')
main()
# MAP_MRR_NDCG_main()


