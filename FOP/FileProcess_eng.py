# -*- coding: UTF-8 -*-
# @CreateTime: 2021/6/18
# @LastUpdateTime: 2021/6/23
# @Author: Yingtong Hu

"""

Process different types of file

"""


def to_dict(extractFile):
    """
    Convert SAOs from txt to dictionary data type
    [{patent ID1: SAO string, patent ID2: SAO string}, {}...]
    """
    inp = open(extractFile, 'r', encoding='utf-8')
    text_line = inp.readline()
    out_list = []
    temp_dic = {}
    while text_line:
        if text_line.strip() == "":
            text_line = inp.readline()
            continue
        if "===========" in text_line:
            out_list.append(temp_dic)
            temp_dic = {}
            text_line = inp.readline()
            continue
        key, info = text_line.split(":", 1)
        if temp_dic.__contains__(key):
            temp_dic[key] += ";" + info.strip()
        else:
            temp_dic[key] = info.strip()
        text_line = inp.readline()
    return out_list


def to_vec(vectorFile):
    """
    Convert vectors from txt to array data type
    [[vec1], [vec2]...]
    """
    f = open(vectorFile, 'r', encoding='utf-8')
    line = f.readline()
    vector = {}
    vsm_ind = {}
    count = 0
    while line:
        while line.startswith(':') or line.count(':') > 1:
            line = f.readline()
        line = line.strip().split(':')
        if line[0] not in vector.keys():
            vector[line[0]] = [float(i) for i in line[1].split(" ")]
            vsm_ind[line[0]] = count
            count += 1
        line = f.readline()
    return vector, vsm_ind

def to_ipcid(vectorFile):
    """
    Convert vectors from txt to array data type
    [[vec1], [vec2]...]
    """
    f = open(vectorFile, 'r', encoding='utf-8')
    line = f.readline()
    vector = {}
    count = 0
    while line:
        while line.startswith(':') or line.count(':') > 1:
            line = f.readline()
        line = line.strip().split(': ')
        if len(line)==1:
            vector[line[0]] = [0] * 14
        elif line[0] not in vector.keys():
            for str in line[1]:
                if str.isalpha():
                    v = ord(str)
                    a = int(v/10)
                    b = v%10
                    if line[0] not in vector:
                        vector[line[0]] = [a]
                        vector[line[0]] += [b]
                    else:
                        vector[line[0]] += [a]
                        vector[line[0]] += [b]
                if str.isdigit():
                    vector[line[0]] += [int(str)]
        if len(vector[line[0]])>14:vector[line[0]]=vector[line[0]][:14]
        if len(vector[line[0]])<14:
            num = 14 - len(vector[line[0]])
            vector[line[0]] += [0]*num
        # print(line[0],vector[line[0]])
        line = f.readline()
    return vector

def get_grade(gradeFile):
    file = open(gradeFile, 'r', encoding='utf-8')
    line = file.readline()
    dic = {}
    while line:
        info = line.strip().split(' ')
        ID_1 = info[1]
        ID_2 = info[2]
        grade = 2 ** int(info[3]) - 1
        if (ID_1, ID_2) not in dic.keys() and (ID_2, ID_1) not in dic.keys():
            dic[(ID_1, ID_2)] = grade
        line = file.readline()
    return dic


def get_data(dataFile):
    file = open(dataFile, 'r', encoding='utf-8')
    r = {}
    allID = {}
    line = file.readline()
    while line:
        # print(line)
        try:
            line = line.strip().split('$')
        except:
            line = file.readline()
            continue
        try:
            ID_1, info_1 = line[0].split(': ', 1)
        except ValueError:
            ID_1, info_1 = line[0].replace(': ', ''),''
        try:
            ID_2, info_2 = line[1].split(': ', 1)
        except ValueError:
            ID_2, info_2 = line[1].replace(': ', ''),''
        except IndexError:
            continue
        if ID_2 in r.keys():
            r[ID_2].append(ID_1)
        else:
            r[ID_2] = [ID_1]

        if ID_1 not in allID.keys():
            allID[ID_1] = info_1
        if ID_2 not in allID.keys():
            allID[ID_2] = info_2

        line = file.readline()
    return r, allID


def to_dict_w_pair(in_file_name):
    inp = open(in_file_name, 'r', encoding='utf-8')
    text_line = inp.readline()
    out_list = []
    pair = {}
    all_dic = {}
    temp_dic = {}
    while text_line:
        if text_line.strip() == "":
            text_line = inp.readline()
            continue
        if "===========" in text_line:
            out_list.append(temp_dic)
            ids = list(temp_dic.keys())
            if len(ids) == 2:
                pair[ids[0]] = ids[1]
                pair[ids[1]] = ids[0]
                all_dic[ids[1]] = temp_dic[ids[1]]
                all_dic[ids[0]] = temp_dic[ids[0]]
            temp_dic = {}
            text_line = inp.readline()
            continue
        key, info = text_line.split(":", 1)
        if temp_dic.__contains__(key):
            temp_dic[key] += ";" + info.strip()
        else:
            temp_dic[key] = info.strip()
        text_line = inp.readline()
    return out_list, pair, all_dic
