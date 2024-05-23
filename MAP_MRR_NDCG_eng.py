# coding=utf-8
# @CreateTime: 2021/6/22
# @LastUpdateTime: 2021/6/23
# @Author: Yingtong Hu

import time
import datetime
import numpy as np
from FileProcess_eng import get_data
from FOP_eng import new_all, init, allMethods
from sklearn.metrics import average_precision_score
from sklearn.metrics._ranking import ndcg_score
from copy import deepcopy


now = datetime.datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

in_file = "data_SAO_eng/extraction_method_1_dict_eng.txt"

relation, all_ID = {}, {}


def mean_reciprocal_rank(rs):
    """Refer to https://www.programcreek.com/python/?CodeExample=mean+reciprocal+rank
    Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def MAP_main():
    record = {}
    total_AP_scores = []
    total_NDCG_scores = []
    MRR_scores = [[], [], [], [], [], [], [], [], [], [],[],[]]
    t1 = time.time()
    total = len(relation) * len(all_ID)
    index_begin = 0
    index_stop = 2
    ind_start =0
    ind_stop = 10

    for index, main_ID in enumerate(relation):
        if index + 1 < index_begin: continue
        if index + 1 == index_stop: break
        print('#' + str(index+1) + ', ' + str(len(relation)-index-1) + ' left')
        print("main ID:", main_ID)
        scores = [[], [], [], [], [], [], [], [], [], [],[],[]]
        true_relevance = [[], [], [], [], [], [], [], [], [], [],[],[]]
        scores_w_ID = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{},{}]
        # print(len(all_ID))
        for ind, ID in enumerate(all_ID):
            t2 = time.time()
            current = index * len(all_ID) + ind + 1
            # print('------ #' + str(index+1) + '.' + str(ind+1) + ' ----- ',
            #        format(current / total * 100, '.2f'), '% done --------')
            print('------ #' + str(index + 1) + '.' + str(ind + 1) + ' ----- ')
            if ind + 1 < ind_start: continue
            if ind + 1 == ind_stop: break
            if ID == main_ID:
                continue

            sim, record = calculate(ID, main_ID, record)
            if sim is not None:
                # for i in range(len(allMethods)):
                for i in range(3):
                    scores[i].append(sim[i])
                    if ID in relation[main_ID]:
                        true_relevance[i].append(1)
                    else:
                        true_relevance[i].append(0)
                    scores_w_ID[i][ID] = sim[i]

            # 记录运行超过2分钟的数据
            cost_time_solo = str(datetime.timedelta(seconds=time.time()-t2))
            tt = time.time() - t1
            print("Cost time: "+cost_time_solo+'   '+"Total cost time: " + str(datetime.timedelta(seconds=tt)))
            # seconds = tt / current * (total - current)
            # print("Estimated time left: " + str(datetime.timedelta(seconds=seconds)))


        NDCG_scores = []
        AP_scores = []

        for i in range(len(allMethods)):
            # calculate NDCG
            NDCG_scores.append(ndcg_score(np.asarray([true_relevance[i]]), np.asarray([scores[i]])))

            # calculate AP
            AP_scores.append(average_precision_score(np.array(true_relevance[i]), np.array(scores[i])))

            # collect MRR
            for target_ID in relation[main_ID]:
                temp = deepcopy(scores_w_ID[i])
                MRR_scores[i].append(get_MRR_score(temp, target_ID))


        total_NDCG_scores.append(NDCG_scores)
        total_AP_scores.append(AP_scores)
        print(total_NDCG_scores)
        print('\n',total_AP_scores)

    # write
    print('\n\n\n'+vectype+' '+str(index_begin)+'.'+str(ind_start)+'-'+str(index_stop)+'.'+str(ind_stop))
    file = open("MAP_MRR_NDCG_"+vectype+"_result_" + str(index_begin)+'_'+str(index_stop-1)+
                '_'+str(ind_stop-1) + ".txt", 'w', encoding='utf-8')

    print("--------------MAP result------------\n")
    file.write("--------------MAP result------------\n")
    mean_AP_scores = get_mean_score(total_AP_scores)
    for i in range(len(allMethods)):
        print(allMethods[i] + ': ' + str(mean_AP_scores[i]))
        file.write(allMethods[i] + ': ' + str(mean_AP_scores[i]) + '\n')

    print("--------------MRR result------------\n")
    file.write("\n--------------MRR result------------\n")
    for i in range(len(allMethods)):
        print(allMethods[i]+': '+str(mean_reciprocal_rank(MRR_scores[i])))
        file.write(allMethods[i] + ': ')
        file.write(str(mean_reciprocal_rank(MRR_scores[i])))


    print("--------------NDCG result------------\n")
    file.write("\n--------------NDCG result------------\n")
    mean_NDCG_scores = get_mean_score(total_NDCG_scores)
    for i in range(len(allMethods)):
        print(allMethods[i] + ': ' + str(mean_NDCG_scores[i]) )
        file.write(allMethods[i] + ': ' + str(mean_NDCG_scores[i]) + '\n')


def calculate(ID, main_ID, record):
    if (main_ID, ID) in record.keys():
        sim = record[(main_ID, ID)]
    elif (ID, main_ID) in record.keys():
        sim = record[(ID, main_ID)]
    else:
        pair = {main_ID: all_ID[main_ID], ID: all_ID[ID]}
        sim = new_all(pair, 12)
        record[(main_ID, ID)] = sim
    return sim, record


def get_mean_score(total):
    mean = []
    for i in range(len(allMethods)):
        try:
            mean.append(np.mean(np.array(total)[:, i]))
        except:
            mean.append(0)
    return mean


def get_MRR_score(simResult, target_ID):
    MRR_score = []
    while simResult:
        closest_patent = max(simResult, key=simResult.get)
        if closest_patent == target_ID:
            MRR_score.append(1)
            MRR_score += [0] * (len(simResult) - 1)
        else:
            MRR_score.append(0)
        simResult.pop(closest_patent)
    return MRR_score


def MAP_MRR_NDCG_main():
    global relation, all_ID
    global vectype
    relation, all_ID = get_data(in_file)

    vectype = 'array'
    in_vector ='vector_eng/vector_'+vectype+'_new.txt'

    init(-1, in_vector, 300, 0, 0, 0)
    MAP_main()

if __name__ == '__main__':
    MAP_MRR_NDCG_main()