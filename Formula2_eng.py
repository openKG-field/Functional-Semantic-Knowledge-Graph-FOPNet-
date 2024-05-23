# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/18
# @Author: Yingtong Hu

"""

Calculation Formulas for different methods
including: Dice coefficient, inclusion index, Jaccard coefficient, Euclidean distance, Pearson coefficient,
            Spearman coefficient, Arcosine distance and different methods of concept hierarchy

"""

import numpy as np
from pandas.core.frame import DataFrame
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from numpy import dot
from numpy.linalg import norm
# from gloveServer import get_wordArray


brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
errorpos = ['a', 'r', 's']


def dice(set1, set2):
    """dice coefficient 2nt/(na + nb)."""
    overlapLen = len(set1 & set2)
    similarity = overlapLen * 2.0 / (len(set1) + len(set2))
    return similarity


def inclusionIndex(set1, set2):
    """inclusion index nt/min(na, nb)."""
    overlap = set1 & set2
    similarity = len(overlap) / min(len(set1), len(set2))
    return similarity


def jaccard(set1, set2):
    """Jaccard coefficient nt/n(aUb)."""
    similarity = len(set.intersection(set1, set2)) / len(set.union(set1, set2))
    return similarity


def euclidean(vector1, vector2):
    """Euclidean Distance """
    d = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return 1 / (1 + d)


def pearson(vector1, vector2):
    """Pearson Coefficient"""
    data = DataFrame({'x': vector1, 'y': vector2})
    similarity = abs(data.corr(method='pearson')['x']['y'])
    if np.isnan(similarity):
        return 0.5
    return similarity


def spearman(vector1, vector2):
    """Spearman Coefficient"""
    data = DataFrame({'x': vector1, 'y': vector2})
    similarity = abs(data.corr(method='spearman')['x']['y'])
    if np.isnan(similarity):
        return 0.5
    return similarity


def arcosine(v1, v2):
    """Arcosine Distance"""
    a = np.sqrt(sum([np.square(x) for x in v1]))
    b = np.sqrt(sum([np.square(x) for x in v2]))
    if a == 0.0 or b == 0.0:
        return 0.0
    d = np.dot(v1, v2) / (a * b)
    return d
    # d = dot(v1,v2)/(norm(v1)*norm(v2))
    # return d


def hierarchy(word1, word2):
    """Concept Hierarchy top-level function"""
    temp_word1 = word1.split(' ')
    temp_word2 = word2.split(' ')

    sim_max = [0.0,0.0, 0.0, 0.0,0.0]
    for t1 in temp_word1:
        for t2 in temp_word2:
            sim_max = hierarchyCalculation(t1, t2, sim_max)
    return sim_max
            # # average value of concept similarity
            # for i in range(3):
            #     d[i] += result[i]
            #     if result[i] != 0:
            #         count[i] += 1
    # return divide(d, count)


def divide(sim, count):
    if len(sim) != len(count):
        print("wrong length matched")
        return -1
    arr = []
    for i in range(len(sim)):
        if sim[i] > 0 and count[i] > 0:
            arr.append(sim[i] / count[i])
        else:
            arr.append(sim[i])
    return arr


def hierarchyCalculation(w1, w2, sim_max):
    """Concept Hierarchy Calculation"""
    if w1 == w2:
        return [1.0, 1.0, 1.0, 1.0, 1.0]
    maxx = [0.0, 0.0, 0.0, 0.0, 0.0]


    synsets1, synsets2 = wordnet.synsets(w1), wordnet.synsets(w2)
    if not synsets1 or not synsets2:
        return maxx

    for s1 in synsets1:
        for s2 in synsets2:
            if s1._pos == s2._pos and s1._pos not in errorpos and s2._pos not in errorpos:
                maxx = [lin(s1, s2, maxx[0]), resnik(s1, s2, maxx[1]), jiang(s1,s2,maxx[2]), leacock(s1,s2,maxx[3]),wu(s1,s2,maxx[4])]
                # maxx = [lin(s1, s2, maxx[0]), resnik(s1, s2, maxx[1]), leacock(s1, s2, maxx[2]), wu(s1,s2, maxx[3])]
            if maxx[0] > sim_max[0]: sim_max[0]=maxx[0]
            if maxx[1] > sim_max[1]: sim_max[1] = maxx[1]
            if maxx[2] > sim_max[2]: sim_max[2] = maxx[2]
            if maxx[3] > sim_max[3]: sim_max[3] = maxx[3]
            if maxx[4] > sim_max[4]: sim_max[4] = maxx[4]
    return sim_max


def wu(s1, s2, val):
    """Wu & Palmer's"""
    sim_wu = s1.wup_similarity(s2)
    if sim_wu is not None and sim_wu > val:
        return sim_wu
    return val


def lin(s1, s2, val):
    """Lin's"""
    try:
        sim_lin = s1.lin_similarity(s2, semcor_ic)
    except:
        return val
    if sim_lin is not None and sim_lin > val:
        return sim_lin
    return val


def leacock(s1, s2, val):
    """ Leacock & Chodorow's"""
    sim1_leacock = s1.lch_similarity(s1)
    sim2_leacock = s2.lch_similarity(s2)
    sim_leacock = s1.lch_similarity(s2) / max(sim1_leacock, sim2_leacock)
    if sim_leacock > val:
        return sim_leacock
    return val


def resnik(s1, s2, val):
    """Resnik's"""
    sim1_resnik = s1.res_similarity(s1, brown_ic)
    sim2_resnik = s2.res_similarity(s2, brown_ic)
    try:
        sim_resnik = s1.res_similarity(s2, brown_ic) / max(sim1_resnik, sim2_resnik)
    except:
        return val
    if sim_resnik is not None and sim_resnik > val:
        return sim_resnik
    return val


def jiang(s1, s2, val):
    """Jiang & Conrath's"""
    sim1_jiang = s1.jcn_similarity(s1, brown_ic)
    sim2_jiang = s2.jcn_similarity(s2, brown_ic)
    sim_jiang = s1.jcn_similarity(s2, brown_ic) / max(sim1_jiang, sim2_jiang)
    if sim_jiang > val:
        return sim_jiang
    return val




def extended_sd(set1,set2):
    tt_dict = {}

    # synsets1, synsets2 = wordnet.synsets(word1), wordnet.synsets(word2)
    # k = 0.0
    # if not synsets1 or not synsets2:
    #     k = inclusionIndex(set1,set2)
    # for s1 in synsets1:
    #     for s2 in synsets2:
    #         if s1._pos == s2._pos and s1._pos not in errorpos and s2._pos not in errorpos:
    #             k = leacock(s1, s2, k)
    # tt_dict[(word1, word2)] = k



    # for word1 in set1:
    #     for word2 in set2:
    #         if word1 == word2:
    #             k = 1
    #         if word1 == ' ' or word2 == ' ':
    #             k = 0
    #         elif word1 == '' or word2 == '':
    #             k = 0
    #         else:
    #             # calculate term-to-term similarity
    #             synsets1, synsets2 = wordnet.synsets(word1), wordnet.synsets(word2)
    #             k = 0.0
    #             if not synsets1 or not synsets2:
    #                 k = inclusionIndex({word1},{word2})
    #                 tt_dict[(word1, word2)] = k
    #                 continue
    #             for s1 in synsets1:
    #                 for s2 in synsets2:
    #                     if s1._pos == s2._pos and s1._pos not in errorpos and s2._pos not in errorpos:
    #                         k = leacock(s1, s2, k)
    #         tt_dict[(word1, word2)] = k



    # term sorting
    list1 = list(set1)
    list2 = list(set2)
    for wd1 in list1:
        for wd2 in list2:
            try:
                vector1 = get_wordArray(wd1)
                vector2 = get_wordArray(wd2)
                k = pearson(vector1,vector2)
            except:
                k=0
            tt_dict[(wd1, wd2)] = k

    E1, E2 = term_sorting(list1, list2, tt_dict)
    # entity-to-entity(S-S, O-S, A-A, S-O, O-O) similarity using extended_sd
    ESD = 0
    for j in range(min(len(E1), len(E2))):
        ESD += F(E1[j], E2[j], tt_dict)
    ESD = 2 * ESD / (len(E1) + len(E2))
    return ESD


# 2. Extended SD algorithm
# calculate differences between first patent and other following patents
#     input_patent_SAO - dict format of patents
#                        key = patent ID
#                        value = SAO
#     return: sim - a list contains sorted patents according to similarity with first patent,
#                   from smallest to biggest
def extended_sd_old(cleaned_sao):
    a1, a2, a3 = 0.3, 0.4, 0.3  # weighting parameters
    first_ID = next(iter(cleaned_sao))
    # first_SAO = cleaned_sao.pop(first_ID)
    first_SAO = cleaned_sao[first_ID]
    score = {}
    total = 0
    for i in cleaned_sao:
        if i == first_ID:continue
        s = {}
        first_SAO = str(first_SAO).replace('(','').replace(')','').split(';')
        cleaned_sao[i] = str(cleaned_sao[i]).replace('(','').replace(')','').split(';')
        # print(first_SAO)
        # print(cleaned_sao[i])
        for n, l1 in enumerate(first_SAO):
            for m, l2 in enumerate(cleaned_sao[i]):
                if n == 1 and m != 1:
                    continue
                if m == 1 and n != 1:
                    continue
                l11 = str(l1).split(', ',2)
                l21 = str(l2).split(', ',2)
                tt_dict = {}
                for word1 in l11:
                    for word2 in l21:
                        if word1 == ' ' or word2 == ' ':
                            k = 0
                        elif word1 == '' or word2 == '':
                            k = 0
                        else:
                            # calculate term-to-term similarity


                            # set1 = set(word1.split(' '))
                            # set2 = set(word2.split(' '))
                            # # k = dice(set1,set2)  # similarity between word1 and word2
                            # k = inclusionIndex(set1,set2)

                            # wdlist1 = word1.split(' ')
                            # wdlist2 = word2.split(' ')
                            # k = 0.0
                            # for wd1 in wdlist1:
                            #     for wd2 in wdlist2:
                            #         try:
                            #             v1 = get_wordArray(wd1)
                            #             v2 = get_wordArray(wd2)
                            #             tmp = pearson(v1,v2)
                            #         except:
                            #             tmp=0
                            #
                            #         if tmp>k : k=tmp




                            wdlist1 = word1.split(' ')
                            wdlist2 = word2.split(' ')
                            for wd1 in wdlist1:
                                for wd2 in wdlist2:
                                    synsets1, synsets2 = wordnet.synsets(wd1), wordnet.synsets(wd2)
                                    k = 0.0
                                    if not synsets1 or not synsets2:
                                        continue
                                    for s1 in synsets1:
                                        for s2 in synsets2:
                                            if s1._pos == s2._pos and s1._pos not in errorpos and s2._pos not in errorpos:
                                                k = leacock(s1,s2,k)

                        tt_dict[(word1, word2)] = k
                # term sorting
                E1, E2 = term_sorting(l11, l21, tt_dict)
                # entity-to-entity(S-S, O-S, A-A, S-O, O-O) similarity using extended_sd
                ESD = 0
                for j in range(min(len(E1), len(E2))):
                    ESD += F(E1[j], E2[j], tt_dict)
                ESD = 2 * ESD / (len(E1) + len(E2))
                s[(n, m)] = ESD
        # weighted average
        comb1 = a1 * s[(0, 0)] + a2 * s[(1, 1)] + a3 * s[(2, 2)]
        comb2 = a1 * s[(0, 2)] + a2 * s[(1, 1)] + a3 * s[(2, 0)]
        score[i] = max(comb1, comb2)
        total += score[i]
    sim = []
    total = 1
    while score:
        closest_patent = max(score, key=score.get)
        sim.append(closest_patent + '(' + str(score[closest_patent] / total) + ', ' + str(score[closest_patent]) + ')')
        score.pop(closest_patent)
    return sim


# Greedy algorithm for term sorting
# input: E_1 - entity 1
#        E_2 - entity 2
# output: entity 1 and entity 2 with updated term order
def term_sorting(E_1, E_2, d):
    flag_i = 0
    flag_j = 0
    m = len(E_1)
    n = len(E_2)
    for k in range(min(m, n)):
        max_temp = -1
        # search for the k-th maximum matching
        for i in range(k, m):
            for j in range(k, n):
                sim_temp = d[(E_1[i], E_2[j])]  # similarity between term i and term j
                if sim_temp > max_temp:
                    flag_i = i
                    flag_j = j
                    max_temp = sim_temp
        # reorder the terms in E1 and E2
        # swap Term k with Term flag_i for E1
        temp = E_1[k]
        E_1[k] = E_1[flag_i]
        E_1[flag_i] = temp
        # swap Term k with Term flag_j for E2
        temp = E_2[k]
        E_2[k] = E_2[flag_j]
        E_2[flag_j] = temp
    return E_1, E_2


# extended sd function
def F(x, y, d):
    R = [0, 0.2, 0.4, 0.6, 0.8, 1]
    s = d[(x, y)]
    for i in range(len(R) - 1):
        if R[i] <= s <= R[i + 1]:
            return R[i]
    return -1