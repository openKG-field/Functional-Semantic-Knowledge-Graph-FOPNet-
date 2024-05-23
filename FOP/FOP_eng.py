
"""

Calculate FOP using string, concept and vector comparison

"""

import re
import time

# import jieba
# jieba.load_userdict(r"D:\Semantic_Search\data\lexicon.txt")
import numpy as np
import datetime
from pandas.core.frame import DataFrame
from Statistics_eng import mean, median, mode, std
from FileProcess_eng import to_dict, to_vec, to_ipcid
from Weight_eng import Weight
from Formula2_eng import dice, inclusionIndex, jaccard, euclidean, pearson, spearman, hierarchy,arcosine


sourceID = ""
targetID = ""
weight_m = ['km', 'sc', 'graph', 'tfidf', 'bm25']
thre = [0.2, 0.4, 0.6, 0.8]
weight = [0.2, 0.4, 0.5, 0.6,  0.8]
allMethods = ['dice', 'inclusion', 'jaccard', 'euclidean', 'pearson', 'spearman', 'cosine', 'Lin', 'resnik',
              'jiang','wu','leacock']
wordComb, allLabel, phraseVector, simFnP = {}, {}, {}, {}
weightScore = {}
threScore = {}
textWeightScore = {}
score = []
ifWeight, ifThreshold, ifTextWeight, vectorLen = 0, 0, 0, 300
words_vector, vsm_index = {}, {}
FOPDict = {}
ipcDict= {}

def load_embedding(embedding_path):
    '''convert file to dict'''
    embedding_list = list()
    count = 0
    for line in open(embedding_path,encoding='utf-8'):
        line = line.strip()
        embedding_list.append(line)
        count += 1
    # print('loaded %s word embedding, finished' % count, )
    myDict = np.array(embedding_list)
    return myDict


def FOPLabel(phrase):
    if phrase[:3] == '^_^':
        updatePhrase = phrase[3:]
        return 1, updatePhrase
    return 0, phrase


def splitFOP(phrase):
    updatePhrase = phrase.split(', ')
    return updatePhrase[0][1:], updatePhrase[1], updatePhrase[2][:-1]


def splitknowledge(phrase):
    updatePhrase = phrase.split(', ')
    if len(updatePhrase)==3:
        return updatePhrase[0].replace('(', ''), updatePhrase[1], updatePhrase[2].replace(')', '')
    else:
        return ['','','']

def init(FOPExtracted, WordVector, INvectorLen, INifWeight, INifThreshold, INifTextWeight):
    global words_vector, vsm_index, FOPDict, ifWeight, ifThreshold, ifTextWeight, vectorLen,ipcDict
    ipcDict = to_ipcid('../FOPdata/data_FOP_eng/eng_ipc_fop.txt')
    words_vector, vsm_index = to_vec(WordVector)    # 获得单词向量
    if FOPExtracted != -1:
        FOPDict = to_dict(FOPExtracted)
    ifWeight = INifWeight   # 是否考虑不同权重方法
    ifThreshold = INifThreshold     # 是否考虑不同阈值
    ifTextWeight = INifTextWeight   # 是否考虑前后部分不同权重
    vectorLen = INvectorLen         # 初始化向量长度
    words_vector[''] = [0] * vectorLen  # 设置空字符的向量值

def initipc(ipcdictfile):
    global ipcDict
    ipcDict = to_ipcid(ipcdictfile)

# setter - 当外部文件需要设置计算方法时使用
def set_ifWeight(INifWeight):
    global ifWeight
    ifWeight = INifWeight


def set_ifThreshold(INifThreshold):
    global ifThreshold
    ifThreshold = INifThreshold


def set_ifTextWeight(INifTextWeight):
    global ifTextWeight
    ifTextWeight = INifTextWeight


def set_vectorLen(INvectorLen):
    global vectorLen
    vectorLen = INvectorLen



# id - {S, O, A lists}
def format_2(pair):
    global allLabel, phraseVector
    out_dict, allLabel, word_TF, phraseVector = {}, {}, {}, {}  # word_TF: patent_id - word - count
    vec_dict_all = []

    for i in pair:
        word_TF[i], label = {}, {}
        phraseVector[i] = []
        text = re.split(';', pair[i])
        F, O, P = [], [], []
        for n, j in enumerate(text):
            label[n], j = FOPLabel(j)
            # tmp = j.split(',',2)
            # if len(tmp)<3:continue
            f, o, p = splitknowledge(j)
            F.append(f)
            O.append(o)
            P.append(p)

            if f not in phraseVector.keys():
                phraseVector[f] = __vector(f)

            if o not in phraseVector.keys():
                phraseVector[o] = __vector(o)

            if p not in phraseVector.keys():
                phraseVector[p] = __vector(p)

            v = [phraseVector[f][t] + phraseVector[o][t] + phraseVector[p][t] for t in
                 range(vectorLen)]
            # v = [phraseVector[s] + phraseVector[a] + phraseVector[o]]

            vec_dict_all.append(v)

            # TF
            if (f, o, p) in word_TF[i]:
                word_TF[i][(f, o, p)] += 1
            else:
                word_TF[i][(f, o, p)] = 1

        out_dict[i] = [F, O, P]
        allLabel[i] = label

    return out_dict, word_TF, vec_dict_all


def __vector(word):
    v = [0.0] * vectorLen
    if word != '':
        temp_wordS = word.split(' ')
        # S/A/O 任意一个中大于一个词
        if len(temp_wordS) > 1:
            try:
                tmp = '$'.join(temp_wordS)
                # print(tmp)
                v = words_vector(tmp)
                return v
            except:
                for t1 in temp_wordS:
                    if t1 != '' and t1 in words_vector:
                        v = [v[t] + words_vector[t1][t] for t in range(vectorLen)]
                    elif t1 !='' and t1 not in words_vector:
                        try:
                            tmp = words_vector(t1)
                        except:
                            tmp = [0.0] * vectorLen
                        v = [v[t] + tmp[t] for t in range(vectorLen)]
        elif temp_wordS[0] in words_vector:
            v = words_vector[temp_wordS[0]]
        else:
            return [0.0] * vectorLen
        return v
    return [0.0] * vectorLen


ifsplit = 0
ifipc = 0
splitfile ='catvectoripc_result.txt'
def new_all(pair, MethodNumber,split=''):
    global sourceID, targetID, simFnP

    cleaned_fop, TF_count, vec_dict_all = format_2(pair)
    sourceID = list(cleaned_fop.keys())[0]
    first_fop = cleaned_fop[sourceID]
    targetID = list(cleaned_fop.keys())[1]
    second_fop = cleaned_fop[targetID]

    # F: 0, 0   P: 1, 1
    simFnP = {(0, 0): [], (1, 1): []}
    A = []
    for SOType in simFnP:
        for i in range(MethodNumber):
            simFnP[SOType].append({})
            A.append({})

    if ifsplit == 1:
        fop1list = []
        for i in range(len(first_fop[0])):
            tmplist = [first_fop[0][i],first_fop[1][i],first_fop[2][i]]
            fop1list.append(tmplist)

        fop2list = []
        for i in range(len(second_fop[0])):
            tmplist = [second_fop[0][i], second_fop[1][i], second_fop[2][i]]
            fop2list.append(tmplist)
        scorelist1=[]
        scorelist2 = []
        scorelist3 = []
        scorelist4 = []
        for i in range(len(fop1list)):
            for j in range(len(fop2list)):
                tmps = fop1list[i][0]
                tmpa = fop1list[i][1]
                tmpo = fop1list[i][2]
                vector1 = phraseVector[tmps] +phraseVector[tmpa]+phraseVector[tmpo]

                tmps = fop2list[j][0]
                tmpa = fop2list[j][1]
                tmpo = fop2list[j][2]
                vector2 = phraseVector[tmps] + phraseVector[tmpa] + phraseVector[tmpo]

                if ifipc == 1:
                    try:v1_ipc = ipcDict[sourceID]
                    except:v1_ipc = [0] * 14
                    try:v2_ipc = ipcDict[targetID]
                    except:v2_ipc = [0] * 14
                    vector1 += v1_ipc
                    vector2 += v2_ipc
                    if len(vector1) != 914: vector1 = vector1[:914]
                    if len(vector2) != 914: vector2 = vector2[:914]

                vector1 = np.array(vector1)
                vector2 = np.array(vector2)
                scorelist1.append(euclidean(vector1, vector2))
                scorelist2.append(pearson(vector1, vector2))
                scorelist3.append(spearman(vector1, vector2))
                scorelist4.append(arcosine(vector1, vector2))
        fout = open(splitfile,'a',encoding='utf-8')
        fout.write(str(np.mean(scorelist1))+' '+
                   str(np.mean(scorelist2))+' '+
                   str(np.mean(scorelist3))+' '+
                   str(np.mean(scorelist4))+'\n')
        print('split Score',np.mean(scorelist1),np.mean(scorelist2),np.mean(scorelist3),np.mean(scorelist4))




    if MethodNumber == 4:
        methods = allMethods[3:7]
    elif MethodNumber == 3:
        methods = allMethods[:3]
    elif MethodNumber == 5:
        methods = allMethods[-5:]
    else:
        methods = allMethods

    first_FP = [first_fop[1], first_fop[2]]
    second_FP = [second_fop[1], second_fop[2]]

    for i, l1 in enumerate(first_FP):
        for j, l2 in enumerate(second_FP):
            if i == 0 and j == 1:continue
            if i == 1 and j == 0: continue
            for ind1, word1 in enumerate(l1):
                for ind2, word2 in enumerate(l2):
                    if word2 == '' or word1 == '':
                        simFnP[(i, j)][0][(ind1, ind2)] = False
                        continue
                    d = similarity_FOP(word1, word2, ind1, ind2, MethodNumber,sourceID,targetID)
                    for index in range(len(methods)):
                        simFnP[(i, j)][index][(ind1, ind2)] = d[index]

    for ind1, word1 in enumerate(first_fop[0]):
        for ind2, word2 in enumerate(second_fop[0]):
            d = similarity_FOP(word1, word2, ind1, ind2, MethodNumber,sourceID,targetID, wordType="a")
            for index in range(len(methods)):
                A[index][(ind1, ind2)] = d[index]

    FOPSim, count = get_FOP_sim(A, MethodNumber)


    # if MethodNumber != 4:
    #     sim_esd = extended_sd(pair)
    #     sim_esdd = sim_esd[0].split(', ')[1]
    #     FOPSim[len(methods)-1] = (float(sim_esdd[:-1]))
    #     if FOPSim[len(methods)-1]>0:
    #         count[len(methods)-1]+=1


    if ifWeight:
        get_sim_w_weight(FOPSim, vec_dict_all, cleaned_fop, first_fop, second_fop, TF_count, pair, count)

    if ifThreshold:
        get_sim_w_thre(FOPSim, count)

    if ifTextWeight:
        get_sim_w_textWeight(FOPSim, count)

    patent_sim = get_patent_sim(FOPSim, count, MethodNumber)

    return patent_sim


def similarity_FOP(word1, word2, ind1, ind2, MethodNumber,sourceID,targetID,wordType=""):
    d = []

    for i in range(MethodNumber):
        d.append(0.0)
    if allLabel[sourceID][ind1] == allLabel[targetID][ind2]:
        if (word1 == word2) or (wordType == "a" and (word1 in word2 or word2 in word1)):
            d = []
            for i in range(MethodNumber):
                d.append(1.0)
        elif (word1, word2) in wordComb:
            d = wordComb[(word1, word2)]
        elif (word2, word1) in wordComb:
            d = wordComb[(word2, word1)]
        else:
            d = similarity_2words(word1, word2, MethodNumber,sourceID,targetID)
    wordComb[(word1, word2)] = d
    return d


def similarity_2words(word1, word2, MethodNumber,sourceID,targetID):
    v1, v2 = phraseVector[word1], phraseVector[word2]

    if ifipc == 1:
        try:v1_ipc = ipcDict[sourceID]
        except:v1_ipc = [0]*14
        try:v2_ipc = ipcDict[targetID]
        except:v2_ipc = [0]*14
        v1 = v1 + v1_ipc
        v2 = v2 + v2_ipc
        if len(v1) != 314: v1 = v1[:314]
        if len(v2) != 314: v2 = v2[:314]
        # print('mean+ipc Score',euclidean(vv1, vv2), pearson(vv1, vv2), spearman(vv1, vv2), arcosine(vv1, vv2))


    set1 = set(word1.split(' '))
    set2 = set(word2.split(' '))
    vector1 = np.array(v1)
    vector2 = np.array(v2)
    if MethodNumber == 4:       # 只用于向量相似度计算
        return [euclidean(vector1, vector2), pearson(vector1, vector2), spearman(vector1, vector2), arcosine(vector1,vector2)]
    elif MethodNumber == 5:
        return hierarchy(word1,word2)
    elif MethodNumber == 3:
        return [dice(set1, set2), inclusionIndex(set1, set2), jaccard(set1, set2)]

    return [dice(set1, set2), inclusionIndex(set1, set2), jaccard(set1, set2),
            euclidean(vector1, vector2),pearson(vector1, vector2), spearman(vector1, vector2), arcosine(vector1,vector2)] + \
           hierarchy(word1, word2)

def get_FOP_sim(A, MethodNumber):
    global score
    FOPSim, count, score = [], [], []
    for i in range(MethodNumber):
        FOPSim.append({})
        count.append(0)
        score.append(0.0)

    if MethodNumber == 4:
        methods = allMethods[3:7]
    elif MethodNumber == 5:
        methods = allMethods[-5:]
    elif MethodNumber == 3:
        methods = allMethods[:3]
    else:
        methods = allMethods

    for j in simFnP[(0, 0)][0]:
        for index in range(len(methods)):
            if simFnP[(0, 0)][0][j] is not False:
                FOPSim[index][j] = 0.5 * simFnP[(0, 0)][index][j] + 0.5 * A[index][j]
            elif simFnP[(1, 1)][0][j] is not False:
                FOPSim[index][j] = 0.5 * A[index][j] + 0.5 * simFnP[(1, 1)][index][j]
            else:
                FOPSim[index][j] = A[index][j]
            if FOPSim[index][j] != 0:
                count[index] += 1
        for index in range(len(methods)):
            score[index] += FOPSim[index][j]
    return FOPSim, count


def get_patent_sim(FOPSim, count, MethodNumber):
    global score
    score = []
    for i in range(MethodNumber):
        score.append(0.0)

    if MethodNumber == 4:
        methods = allMethods[3:7]
    elif MethodNumber ==3:
        methods = allMethods[:3]
    elif MethodNumber == 5:
        methods = allMethods[-5:]
    else:
        methods = allMethods

    for j in simFnP[(0, 0)][0]:
        for index in range(len(methods)):
            score[index] += FOPSim[index][j]
    for index in range(len(methods)):

        if count[index] > 0:
            score[index] = score[index] / count[index]
        else:
            score[index] = 0.0
    return score


def get_sim_w_weight(FOPSim, vec_dict_all, cleaned_fop, first_fop, second_fop, TF_count, pair, count):
    global weightScore
    weightSys = Weight(FOPDict, pair, vec_dict_all, cleaned_fop, allMethods)
    weightSys.set_up()
    weightScore = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}]
    c = {}
    for m in weight_m:
        c[m] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in simFnP[(0, 0)][0]:
        temp_score = []
        for index in range(len(score)):
            temp_score.append(FOPSim[index][j])

        # bm25
        weightScore, c['bm25'] = weightSys.bm25(j, FOPSim, weightScore, c['bm25'])

        # km
        weightScore, c['km'] = weightSys.KMeans(j, first_fop, temp_score, weightScore, c['km'])

        # sc
        weightScore, c['sc'] = weightSys.SpectralClustering(j, first_fop, temp_score, weightScore, c['sc'])

        # # graph
        weightScore = weightSys.Graph(j, temp_score, weightScore)

        # tfidf
        weightScore = weightSys.tfidf(j, first_fop, second_fop, TF_count, temp_score, weightScore)

    for index in range(len(allMethods)):
        for m in weight_m:
            if (m == 'bm25' or m == 'km' or m == 'sc') and m in weightScore[index].keys() and c[m][index] > 0:
                weightScore[index][m] = weightScore[index][m] / c[m][index]
            elif m in weightScore[index].keys() and count[index] > 0:
                weightScore[index][m] = weightScore[index][m] / count[index]
            else:
                weightScore[index][m] = 0.0


def get_sim_w_thre(FOPSim, count):
    global threScore
    threScore = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for j in simFnP[(0, 0)][0]:
        for index in range(len(allMethods)):
            for t in thre:
                if FOPSim[index][j] > t:
                    s = 1.0
                else:
                    s = FOPSim[index][j]
                if str(t) not in threScore[index].keys():
                    threScore[index][str(t)] = s
                else:
                    threScore[index][str(t)] += s
    for index in range(len(allMethods)):

        for m in thre:
            if count[index] > 0:
                threScore[index][str(m)] = threScore[index][str(m)] / count[index]
            else:
                threScore[index][str(m)] = 0.0


def get_sim_w_textWeight(FOPSim, count):
    global textWeightScore
    textWeightScore = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    for j in simFnP[(0, 0)][0]:
        if allLabel[targetID][j[1]] == 1:
            for index in range(len(allMethods)):
                for w in weight:
                    if str(w) not in textWeightScore[index].keys():
                        textWeightScore[index][str(w)] = w * FOPSim[index][j]
                    else:
                        textWeightScore[index][str(w)] += w * FOPSim[index][j]
        else:
            for index in range(len(allMethods)):
                for w in weight:
                    if str(w) not in textWeightScore[index].keys():
                        textWeightScore[index][str(w)] = (1 - w) * FOPSim[index][j]
                    else:
                        textWeightScore[index][str(w)] += (1 - w) * FOPSim[index][j]
    for index in range(len(allMethods)):
        for m in weight:
            if count[index] > 0:
                textWeightScore[index][str(m)] = textWeightScore[index][str(m)] / count[index]
            else:
                textWeightScore[index][str(m)] = 0.0



def common_main(filename=''):
    rawScoreFile = [[], [], [], []]
    compare, target, id_list = [], [], []
    t1 = time.time()
    start = 0
    end = 100
    # fout = open('fop_text_matching_array_fopipc_' + str(start) + '_' + str(end) + '.txt', 'w', encoding='utf-8')
    for ind, pair in enumerate(FOPDict):
        # len(FOPDict)
        start_t = time.time()
        print('\n------ #' + str(ind + 1) + ' ----- ', format(ind / len(FOPDict) * 100, '.2f'),
              '% done --------')


        if ind + 1 == end: break
        if ind + 1 < start: continue


        id_ = list(pair.keys())
        if id_ in id_list:
            print('id error: already calculated\n')
            continue

        if len(id_) != 2:
            print('id error: not a pair')
            continue

        id_list.append(id_)
        compare.append(id_[0])
        target.append(id_[1])
        # print('compare:' + str(id_[0]))
        # print('target:' + str(id_[1]))


        #Method number can be 4 or other
        rawScore = new_all(pair, 4)
        # writescore = str(rawScore).replace('[', '').replace(']', '').replace(', ', ',')
        # fout.write(str(ind + 1) + ',' + str(id_[0]) + ',' + str(id_[1]) + ',' + writescore + '\n')
        print('mean Score:' + str(rawScore))
        for i in range(len(rawScore)):
            rawScoreFile[i].append(rawScore[i])

        cost_t = time.time() - start_t
        # print('cost time: '+  str(datetime.timedelta(seconds=cost_t)))


        tt = time.time() - t1
        # print('Total Cost time: ' + str(datetime.timedelta(seconds=tt)))
        seconds = tt / (ind + 1) * (len(FOPDict) - (ind + 1))
        # print("still need: " + str(datetime.timedelta(seconds=seconds)))

    compare += ["", "mean", "median", "mode", "standard deviation"]

    meanData, medianData, modeData, stdData = mean(rawScoreFile), median(rawScoreFile), mode(rawScoreFile), \
                                              std(rawScoreFile)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    for i in range(len(rawScoreFile)):
        rawScoreFile[i] += ["", meanData[i], medianData[i], modeData[i], stdData[i]]


    data = [compare, target + [""]] + rawScoreFile

    data = DataFrame(data)
    data = data.T
    data.rename(
        columns={0: 'compare_ID', 1: 'target_ID', 2: 'Euclidean', 3: 'Pearson', 4: 'Spearman', 5: 'Cosine',
                 }, inplace=True)
    data.to_csv("ResultFiles/SAOrawScore_"+filename + "_"+dt_string + ".csv", encoding='utf_8_sig')
    # data.to_csv("ResultFiles/SAOrawScore_array3_" + dt_string + ".csv", encoding='utf_8_sig')


def main():
    rawScoreFile = [[], [], [], [], [], [], [], [], [], [], [], []]
    file = {}
    threFile = {}
    textFile = {}
    for w in weight_m:
        file[w] = [[], [], [], [], [], [], [], [], [], [], [], []]
    for m in thre:
        threFile[str(m)] = [[], [], [], [], [], [], [], [], [], [], [], []]
    for m in weight:
        textFile[str(m)] = [[], [], [], [], [], [], [], [], [], [], [], []]
    compare, target, id_list = [], [], []
    t1 = time.time()

    start = 0
    end = 1000
    fout = open('fop_text_matching_array_saoipc_'+str(start)+'_'+str(end)+'.txt','a',encoding='utf-8')
    for ind, pair in enumerate(FOPDict):
        start_t =time.time()
        print('\n------ #' + str(ind + 1) + ' ----- ', format(ind / len(FOPDict) * 100, '.2f'),
              '% done --------')

        if ind + 1 == end: break
        if ind + 1 < start:continue
        # if ind + 1 in skip: continue

        id_ = list(pair.keys())
        if id_ in id_list:
            print('id error: already calculated\n')
            continue

        if len(id_) != 2:
            print('id error: not a pair')
            continue

        id_list.append(id_)
        compare.append(id_[0])
        target.append(id_[1])
        print('compare:' + str(id_[0]))
        print('target:' + str(id_[1]))

        # Methodnumber can be 4 or 10 or other
        rawScore = new_all(pair, 12)
        # rawScore[-2:] =[0,0]
        print(rawScore)
        writescore = str(rawScore).replace('[','').replace(']','').replace(', ',',')
        fout.write(str(ind+1)+','+str(id_[0])+','+str(id_[1])+','+writescore+'\n')
        for i in range(len(rawScore)):
            rawScoreFile[i].append(rawScore[i])

        if (ifWeight):
            for i in range(len(weightScore)):
                for key in weightScore[i].keys():
                    file[key][i].append(weightScore[i][key])

        if (ifThreshold):
            for i in range(len(threScore)):
                for key in threScore[i].keys():
                    threFile[key][i].append(threScore[i][key])

        if (ifTextWeight):
            for i in range(len(textWeightScore)):
                for key in textWeightScore[i].keys():
                    textFile[key][i].append(textWeightScore[i][key])

        tt = time.time() - start_t
        ttt = time.time() - t1
        seconds = tt / (ind + 1) * (len(FOPDict) - (ind + 1))
        print('Cost time: '+ str(datetime.timedelta(seconds=tt)))
        print('Total Cost time: ' + str(datetime.timedelta(seconds=ttt)))
        # print("Estimated time left: " + str(datetime.timedelta(seconds=seconds)))


    compare += ["", "mean", "median", "mode", "standard deviation"]

    meanData, medianData, modeData, stdData = mean(rawScoreFile), median(rawScoreFile), mode(rawScoreFile), \
                                              std(rawScoreFile)

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

    for i in range(len(rawScoreFile)):
        rawScoreFile[i] += ["", meanData[i], medianData[i], modeData[i], stdData[i]]
    data = [compare, target + [""]] + rawScoreFile

    data = DataFrame(data)
    data = data.T
    data.rename(
        columns={0: 'compare_ID', 1: 'target_ID', 2: 'Dice', 3: 'Inclusion', 4: 'Jac', 5: 'Euclidean',
                 6: 'Pearson',
                 7: 'Spearman', 8: 'Cosine', 9: 'Lin', 10: 'Resnik',
                 11: 'Leacock', 12: 'Jiang', 13: 'Extend_SD'
                 }, inplace=True)
    # data.to_csv("TestFiles/KnowledgerawScore_" + dt_string + ".csv", encoding='utf_8_sig')
    data.to_csv("ResultFiles/KnowledgerawScore_" + dt_string + ".csv", encoding='utf_8_sig')

    if(ifWeight):
        for f in file:
            meanData, medianData, modeData, stdData = mean(file[f]), median(file[f]), mode(file[f]), std(file[f])

            for i in range(len(file[f])):
                file[f][i] += ["", meanData[i], medianData[i], modeData[i], stdData[i]]
            data = [compare, target + [""]] + file[f]

            data = DataFrame(data)
            data = data.T
            data.rename(
                columns={0: 'compare_ID', 1: 'target_ID', 2: 'Dice', 3: 'Inclusion', 4: 'Jac', 5: 'Euclidean',
                         6: 'Pearson',
                         7: 'Spearman', 8: 'Arccos', 9: 'Lin', 10: 'Resnik',
                         11: 'Leacock', 12: 'Jiang', 13: 'Extend_SD'
                         }, inplace=True)
            # data.to_csv("TestFiles/Knowledge_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')
            data.to_csv("ResultFiles/Knowledge_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')

    if(ifThreshold):
        for f in threFile:
            meanData, medianData, modeData, stdData = mean(threFile[f]), median(threFile[f]), mode(threFile[f]), \
                                                      std(threFile[f])

            for i in range(len(threFile[f])):
                threFile[f][i] += ["", meanData[i], medianData[i], modeData[i], stdData[i]]
            data = [compare, target + [""]] + threFile[f]

            data = DataFrame(data)
            data = data.T
            data.rename(
                columns={0: 'compare_ID', 1: 'target_ID', 2: 'Dice', 3: 'Inclusion', 4: 'Jac', 5: 'Euclidean',
                         6: 'Pearson',
                         7: 'Spearman', 8: 'Arccos', 9: 'Lin', 10: 'Resnik',
                         11: 'Leacock', 12: 'Jiang', 13: 'Extend_SD'
                         }, inplace=True)
            # data.to_csv("TestFiles/Knowledge_thre_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')
            data.to_csv("ResultFiles/Knowledge_thre_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')

    if(ifTextWeight):
        for f in textFile:
            meanData, medianData, modeData, stdData = mean(textFile[f]), median(textFile[f]), mode(textFile[f]), \
                                                      std(textFile[f])

            for i in range(len(textFile[f])):
                textFile[f][i] += ["", meanData[i], medianData[i], modeData[i], stdData[i]]
            data = [compare, target + [""]] + textFile[f]

            data = DataFrame(data)
            data = data.T
            data.rename(
                columns={0: 'compare_ID', 1: 'target_ID', 2: 'Dice', 3: 'Inclusion', 4: 'Jac', 5: 'Euclidean',
                         6: 'Pearson',
                         7: 'Spearman', 8: 'Arccos', 9: 'Lin', 10: 'Resnik',
                         11: 'Leacock', 12: 'Jiang', 13: 'Extend_SD'
                         }, inplace=True)
            # data.to_csv("TestFiles/Knowledge_weight_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')
            data.to_csv("ResultFiles/Knowledge_weight_" + f + '_' + dt_string + ".csv", encoding='utf_8_sig')



def sim_tmp(word1, word2):
    set1 = set(word1.split(', '))
    set2 = set(word2.split(', '))
    return [dice(set1, set2), inclusionIndex(set1, set2), jaccard(set1, set2)]

def cal_keyword_sim():
    fin = open('ipc/sao/sao_eng_pairs_SAO.txt','r',encoding='utf-8')
    fout = open('ResultFiles\sim_keywords','w',encoding='utf-8')
    text = fin.readline().strip()
    flag=0
    pair_list = []
    sim_list1 = []
    sim_list2 = []
    sim_list3 = []
    while text:
        if flag == 7:break
        if '====' in text:
            wd1 = pair_list[1]
            wd2 = pair_list[3]
            sim = sim_tmp(wd1,wd2)
            sim_list1.append(sim[0])
            sim_list2.append(sim[1])
            sim_list3.append(sim[2])
            s1 = np.mean(sim_list1)
            s2 = np.mean(sim_list2)
            s3 = np.mean(sim_list3)
            print(pair_list[0],pair_list[2],s1,s2,s3,'\n')
            pair_list = []
            sim_list1 = []
            sim_list2 = []
            sim_list3 = []
            text = fin.readline().strip()
            flag+=1
        id, keyword = str(text).split(': (',1)
        # keyword = str(keyword).split(', ')
        keyword = str(keyword).replace(');(',', ').replace(')','')
        # print(keyword)
        pair_list.append(id)
        pair_list.append(keyword)
        text = fin.readline().strip()
        flag+=1


def get_ipcdict():
    fin = open('DWSAO/old/cleanedIpc.txt','r',encoding='utf-8')
    fout = open('DWSAO/old/cleanedIpcDict.txt','w',encoding='utf-8')
    t =  fin.readline().strip()
    while t:
        while len(t)<18:
            t = fin.readline().strip()
        try:
            id,ipc = str(t).split(': ')
            number = ipc2num(ipc)
            fout.write(str(id)+':'+str(number)+'\n')
            print(id,ipc, number)
            ipcDict[id] = number
        except:
            pass
        t = fin.readline().strip()

def ipc2num(ipc):
    number = ''
    number += str(ord(ipc[0]))+ ' '
    number += str(int(ipc[1])) + ' '
    number += str(int(ipc[2])) + ' '
    number += str(ord(ipc[3]))+ ' '
    left,right = str(ipc).split('/',1)
    left = left[4:]
    left = str(int(left))
    right = str(int(right))
    number += left+' '
    number += right

    return number


if __name__ == '__main__':
    get_ipcdict()
    # cal_keyword_sim()
    # a = 'ber physical system, comprising, exchanger, search ability determination, visualizing, accentuated graphical user academic user,'
    # print(a)
    # b = set(a.split(', '))
    # print(b)
