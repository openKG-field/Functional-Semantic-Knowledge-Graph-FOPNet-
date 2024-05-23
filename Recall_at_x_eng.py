# coding=utf-8

import re
import datetime
import time
import random
from FOP_eng import new_all, init, allMethods,initipc
from Statistics_eng import mean
from FileProcess_eng import to_dict_w_pair


def generate_random_100():
    patent_ID = []
    while len(patent_ID) < 100:
        patent_index = random.randint(0, len(SAO_dic)-1)
        temp_id = list(SAO_dic.keys())[patent_index]
        if temp_id not in patent_ID:
            patent_ID.append(temp_id)
    to_compare = patent_ID[0]
    target = pairs[to_compare]
    if target in patent_ID:
        patent_ID.remove(target)
        while len(patent_ID) < 100:
            patent_index = random.randint(0, len(SAO_dic)-1)
            temp_id = list(SAO_dic.keys())[patent_index]
            if temp_id not in patent_ID and temp_id != target:
                patent_ID.append(temp_id)
    dataset = [{to_compare: SAO_dic[to_compare], target: SAO_dic[target]}]
    for index in patent_ID[1:]:
        temp_dic = {to_compare: SAO_dic[to_compare], index: SAO_dic[index]}
        dataset.append(temp_dic)
    return dataset, to_compare, target


def write_dataset():
    file.write(compare_ID + '\n')
    file.write(target_ID + '\n')
    for pair in data_SAO:
        keys = list(pair.keys())
        for key in keys:
            file.write(key + ', ')
        file.write('\n')
    file.write("=================================================\n")


def format_field_tech(patent_pair):
    info = {}
    word_TF = {}
    for id_ in patent_pair:
        word_TF[id_] = {}
        text = re.split('#', patent_pair[id_])
        text = text[2] + text[4]
        words = text.split(',')
        words.remove('')
        info[id_] = words
        for word in words:
            if word in word_TF[id_]:
                word_TF[id_][word] += 1
            else:
                word_TF[id_][word] = 1
    return info, word_TF


def all_main(patent_list, round_number, total):
    dataFile = open('Recall_at_10_' + dt_string + '.txt', 'w', encoding='utf-8')
    id_list = []
    method_result = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

    for ind, pair in enumerate(patent_list):
        number = (ind + 1) + round_number * 100
        # if number%5 == 0:break
        if ind +1== 20:break
        print('------ #' + str(number) + ' ----- ', format((number-1) / total * 100, '.2f'),
              '% done --------')

        id_ = list(pair.keys())
        if id_ in id_list:
            print('id error: already calculated\n')
            continue

        if len(id_) != 2:
            print('id error: not a pair')
            continue

        id_list.append(id_)

        sim = new_all(pair, 12)
        for index in range(len(sim)):
            method_result[index][id_[1]] = sim[index]

        tt = time.time() - start_time
        seconds = tt / number * (total - number)
        print("Estimated time left: " + str(datetime.timedelta(seconds=seconds)))

    dataFile.write('compare_ID: ' + compare_ID + '\ntarget_ID: ' + target_ID + '\n')
    termRank = []
    for index, method in enumerate(allMethods):
        dataFile.write(method + ': \n')
        result = method_result[index]
        count = 0
        while result:
            closest_patent = max(result, key=result.get)
            count += 1
            dataFile.write(str(count) + ' ' + closest_patent + '(' + str(result[closest_patent]) + ')\n')
            if closest_patent == target_ID:
                termRank.append(count)
                break
            result.pop(closest_patent)
        dataFile.write('=============================================\n')
    return termRank


now = datetime.datetime.now()
dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")

in_file = "./FOPdata/data_FOP_eng/extraction_method_1_eng.txt"
in_file_ipc = './FOPdata/data_FOP_eng/eng_ipc_fop.txt'
input_SAO, pairs, SAO_dic = to_dict_w_pair(in_file)

SAOExtracted = "./FOPdata/data_FOP_eng/extraction_method_1_eng.txt"
WordVector = './FOPdata/data_FOP_eng/vector.txt'
vectorLen = 300

ipcdict = './FOP/ipc/ipc_old_data/eng_ipc.txt'
generate_file = 'recall_dataset' + dt_string + '.txt'
file = open(generate_file, 'w', encoding='utf-8')
init(SAOExtracted, WordVector, vectorLen, 0, 0, 0)
# init(-1, WordVector, vectorLen, 0, 0, 0)
initipc(ipcdict)

print("after data processing")
rank = [[], [], [], [], [], [], [], [], [], [], [], []]
start_time = time.time()
for i in range(20):
    # if i==5:break
    data_SAO, compare_ID, target_ID = generate_random_100()
    write_dataset()
    r = all_main(data_SAO, i, 10*100)
    for j in range(len(allMethods)):
        rank[j].append(r[j])


mean = mean(rank)
f = open('recall_10_mean_knowledge_' + dt_string + '.txt', 'w', encoding='utf-8')
for i in range(len(allMethods)):
    f.write(allMethods[i] + ': ' + str(mean[i]) + '\n')

