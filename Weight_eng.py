# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/23
# @Author: Yingtong Hu

"""

Different weighting methods for FOP similarity calculation

"""

import numpy as np
import re
from BM25_eng import BM25
from sklearn.cluster import KMeans, SpectralClustering

import networkx as nx
import FOP_eng


class Weight:
    """
    Combining different weighting methods: BM25, KMeans, Spectral Clustering, Graph and TFIDF
    """
    km_labels = []
    sc_labels = []
    graph_degrees = []
    bm25_matrix = []
    bm25_mean = 0
    IDF_count = {}
    doc_num = 0

    def __init__(self, patents, input_patent_FOP, vec, FOP_, methods):
        self.patents = patents
        self.input_patent_FOP = input_patent_FOP
        self.vec = vec
        self.FOP = FOP_
        self.methods = methods
        self.sc_label_define = 1

    def set_up(self):
        """
        getting weight calculation required variables ready
        :return: None
        """
        self.km_labels = self.__KMeans_set_up()
        self.sc_labels = self.__SpectralClustering_set_up()
        self.graph_degrees = self.__Graph_set_up()
        self.bm25_matrix, self.bm25_mean = self.__bm25_set_up()
        self.IDF_count, self.doc_num = self.__tfidf_set_up()

    def __bm25_set_up(self):
        """
        set up bm25
        :return: matrix of bm25 values; mean of all bm25 values
        """

        values = self.input_patent_FOP.values()
        FOPs = self.__to_FOP(list(values)[0])
        test = []
        for FOPComb in FOPs:
            test.append(self.__to_word(FOPComb))
        s = BM25(test)
        input_s = self.__to_FOP(list(values)[1])
        matrix = []
        for i in input_s:
            matrix.append(s.similarity((self.__to_word(i))))
        return matrix, np.mean(matrix)

    @staticmethod
    def __to_FOP(sentence):
        """
        For BM25 calculation
        split string of all FOPs to array of FOPs
        eg. "^_^(aaa, bbb, ccc); (aaa, bbb, ccc)" --> ["(aaa, bbb, ccc)", "(aaa, bbb, ccc)"]
        """

        if type(sentence) == str:
            sentence = sentence.replace('^_^', '').replace(';', '')
            FOPs = sentence.split(')')
        else:
            FOPs = sentence
        while '' in FOPs:
            FOPs.remove('')
        for i in range(len(FOPs)):
            FOPs[i] = FOPs[i]+')'
        return FOPs

    @staticmethod
    def __to_word(FOPComb):
        """
        For BM25 calculation
        split FOPs to single word combination
        eg. (aaa, bbb, ccc) --> [aaa, bbb, ccc]
        """
        word = FOPComb.split(', ')
        for i in range(len(word)):
            word[i] = word[i].replace('(', '').replace(')', '')
        if '' in word:
            word.remove('')
        return word

    def __KMeans_set_up(self):
        """
        set up KMeans
        :return: KMeans labels
        """
        cluster = KMeans(n_clusters=2).fit(self.vec)
        return cluster.labels_

    def __SpectralClustering_set_up(self):
        """
        set up Spectral Clustering
        :return: Spectral Clustering labels
        """
        try:
            cluster = SpectralClustering(n_clusters=2, assign_labels='discretize').fit(self.vec)
            self.sc_label_define = 1
        except:
            print("Spectral Clustering fail")
            self.sc_label_define = -1
            return []
        return cluster.labels_

    def __Graph_set_up(self):
        """
        set up Graph
        :return: graph degrees
        """
        degrees = []
        for i in self.FOP:
            degrees.append(self.__FOPGraph(self.FOP[i][0], self.FOP[i][1], self.FOP[i][2]))
        return degrees

    @staticmethod
    def __FOPGraph(F, O, P):
        """
        create graph
       
        :return: normalized degree
        """

        # return list([0]*len(S))

        g =nx.Graph()
        g.add_nodes_from(list(set(F + O + P)))

        for i in range(len(F)):
            if F[i] != '' and P[i] != '' and O[i] != '' and not g.has_edge((F[i], P[i])):
                g.add_edge((F[i], P[i]))
            if F[i] != '' and O[i] != '' and not g.has_edge((F[i], O[i])):
                g.add_edge((F[i], O[i]))
            if P[i] != '' and O[i] != '' and not g.has_edge((O[i], P[i])):
                g.add_edge((O[i], P[i]))
        degrees = np.array([])
        for i in range(len(F)):
            degrees = np.append(degrees, len(g.neighbors(F[i])) + len(g.neighbors(O[i])) + len(g.neighbors(P[i])))
        if len(degrees)<1:
            return list([0]*len(F))
        if max(degrees) == 0:
            return degrees
        return list(degrees / max(degrees))

    def __tfidf_set_up(self):
        """
        set up TFIDFS
        :return: each word's IDF value; total number of documents
        """
        word_IDF = {}  # word - [patent]
        patent_ID = []
        for input_patent_FOP in self.patents:
            for i in input_patent_FOP:
                if i not in patent_ID:
                    patent_ID.append(i)

                if type(input_patent_FOP[i])==list:
                    text = input_patent_FOP[i]
                else:
                    text = re.split(';', input_patent_FOP[i])
                for n, j in enumerate(text):
                    j = FOP_eng.FOPLabel(j)[1]
                    s, a, o = FOP_eng.splitknowledge(j)
                    # IDF
                    if (s, a, o) in word_IDF:
                        if i not in word_IDF[(s, a, o)]:
                            word_IDF[(s, a, o)].append(i)
                    else:
                        word_IDF[(s, a, o)] = [i]
        return word_IDF, len(patent_ID)

    def bm25(self, FOP_index, similarity, similarity_w_weight, count):
        """
        calculate bm25 weight
        :param FOP_index: current FOP pair
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """

        try:
            if self.bm25_matrix[FOP_index[1]][FOP_index[0]] > self.bm25_mean:
                for index in range(len(self.methods)):
                    if index == len(self.methods)-1:
                        if similarity[index] > 0:
                            count[index] += 1
                        if 'bm25' in similarity_w_weight[index].keys():
                            similarity_w_weight[index]['bm25'] += similarity[index]
                        else:
                            similarity_w_weight[index]['bm25'] = similarity[index]
                    else:
                        if similarity[index][FOP_index] > 0:
                            count[index] += 1
                        if 'bm25' in similarity_w_weight[index].keys():
                            similarity_w_weight[index]['bm25'] += similarity[index][FOP_index]
                        else:
                            similarity_w_weight[index]['bm25'] = similarity[index][FOP_index]
        except:
            pass
        return similarity_w_weight, count

    def KMeans(self, FOP_index, source, similarity, similarity_w_weight, count):
        """
        calculate KMeans weight
        :param FOP_index: current FOP pair
        :param source: source patent
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        km_weight = 1 if self.km_labels[FOP_index[0]] == self.km_labels[len(source[0]) + FOP_index[1]] else 0
        km_score = similarity
        if km_weight == 1:
            km_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        for index in range(len(self.methods)):
            if km_score[index] > 0:
                count[index] += 1

            if 'km' in similarity_w_weight[index].keys():
                similarity_w_weight[index]['km'] += km_score[index]
            else:
                similarity_w_weight[index]['km'] = km_score[index]
        return similarity_w_weight, count

    def SpectralClustering(self, FOP_index, source, similarity, similarity_w_weight, count):
        """
        calculate Spectral Clustering weight
        :param FOP_index: current FOP pair
        :param source: source patent
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        if self.sc_label_define == -1:
            sc_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0]
        else:
            sc_weight = 1 if self.sc_labels[FOP_index[0]] == self.sc_labels[len(source[0]) + FOP_index[1]] else 0
            sc_score = similarity
            if sc_weight == 1:
                sc_score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0]
        for index in range(len(self.methods)):
            if sc_score[index] > 0:
                count[index] += 1
            if 'sc' in similarity_w_weight[index].keys():
                similarity_w_weight[index]['sc'] += sc_score[index]
            else:
                similarity_w_weight[index]['sc'] = sc_score[index]
        return similarity_w_weight, count

    def Graph(self, FOP_index, similarity, similarity_w_weight):
        """
        calculate Graph weight
        :param FOP_index: current FOP pair
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        :return: updated similarity_w_weight
        """
        d1 = self.graph_degrees[0][FOP_index[0]]
        d2 = self.graph_degrees[1][FOP_index[1]]
        if 'graph' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                similarity_w_weight[index]['graph'] += similarity[
                                             index] * (d1 + d2) / 2
        else:
            for index in range(len(self.methods)):
                similarity_w_weight[index]['graph'] = similarity[
                                            index] * (d1 + d2) / 2
        return similarity_w_weight

    def tfidf(self, FOP_index, source, target, TF_count, similarity, similarity_w_weight):
        """
        calculate TFIDF weight
        :param FOP_index: current FOP pair
        :param source: source patent FOP list
        :param target: target patent FOP list
        :param TF_count: total TF count
        :param similarity: raw similarity
        :param similarity_w_weight: similarity with weight added
        :return: updated similarity_w_weight
        """
        w1 = self.__get_tfidf_val(FOP_index[0], source, FOP_eng.sourceID, TF_count)
        w2 = self.__get_tfidf_val(FOP_index[1], target, FOP_eng.targetID, TF_count)
        if 'tfidf' in similarity_w_weight[0].keys():
            for index in range(len(self.methods)):
                if similarity[index] * (w1 + w2) / 2 > 1:
                    similarity_w_weight[index]['tfidf'] += 1
                else:
                    similarity_w_weight[index]['tfidf'] += similarity[index] * (w1 + w2) / 2
        else:
            for index in range(len(self.methods)):
                if similarity[index] * (w1 + w2) / 2 > 1:
                    similarity_w_weight[index]['tfidf'] = 1
                else:
                    similarity_w_weight[index]['tfidf'] = similarity[index] * (w1 + w2) / 2
        return similarity_w_weight

    def __get_tfidf_val(self, ind, FOPList, ID, TF_count):
        """
        get TFIDF value
        :param ind: FOP index
        :param FOPList: patent's FOP list
        :param ID: patent ID
        :param TF_count: total TF count
        :return: TFIDF value for this FOP
        """
        if FOPList[0][ind] == ' ':
            word = (FOPList[1][ind], FOPList[2][ind])
        else:
            word = (FOPList[0][ind], FOPList[1][ind], FOPList[2][ind])
        tf = TF_count[ID][word] / len(TF_count[ID])
        idf = np.log(self.doc_num / (len(self.IDF_count[word]) + 1))
        if idf == 0.0:
            idf = 0.0001
        tfidf_val = tf * idf
        return tfidf_val
