# -*- coding: utf-8 -*-
import codecs
import json
import datetime
import os, shutil
from sklearn.model_selection import train_test_split
import re
import numpy as np
import jieba
import gensim
from utils.data import Data

def split_data(path):
    with codecs.open(path, 'r', 'utf-8') as json_data:
        all_data = json.load(json_data)

    all_data = list(all_data.items())
    print("total data size:", len(all_data))
    train_data, other_data = train_test_split(all_data,  test_size=0.2, random_state=2020)
    _, train_hand_out = train_test_split(train_data, test_size=0.1, random_state=2020)
    valid_data, test_data = train_test_split(other_data,  test_size=0.5, random_state=2020)

    print("train data size: ", len(train_data))
    print("hand out data size: ", len(train_hand_out))
    print("val data size:", len(valid_data))
    print("test data size:", len(test_data))

    train_data_dic, train_hand_out_dic, valid_data_dic, test_data_dic = {}, {}, {}, {}
    for item in train_data:
        train_data_dic[item[0]] = item[1]
    for item in train_hand_out:
        train_hand_out_dic[item[0]] = item[1]
    for item in valid_data:
        valid_data_dic[item[0]] = item[1]
    for item in test_data:
        test_data_dic[item[0]] = item[1]

    with open('./data/chaming-train.json', 'w+', encoding='utf-8') as train_file:
        json.dump(train_data_dic, train_file, indent=4, ensure_ascii=False)

    with open('./data/chaming-handout.json', 'w+', encoding='utf-8') as h_file:
        json.dump(train_hand_out_dic, h_file, indent=4, ensure_ascii=False)

    with open('./data/chaming-dev.json', 'w+', encoding='utf-8') as val_file:
        json.dump(valid_data_dic, val_file, indent=4, ensure_ascii=False)

    with open('./data/chaming-test.json', 'w+', encoding='utf-8') as test_file:
        json.dump(test_data_dic, test_file, indent=4, ensure_ascii=False)


def extract_chaming(path, outpath):

    rule1 = '事实作?(?:如|以)下.*?本院认为'
    rule2 = '(?:审理|本院)查明.*?本院认为'
    rule3 = '(?:如|以)下事实.*?本院认为'
    rule4 = '审(?:查|理|核)(?:认定|认为|查明).*?本院认为'
    rule5 = '事实认定如下.*?本院认为'
    rule6 = '本院认定.*?本院认为'
    rule7 = '(?:以|如)下法律事实.*?本院认为'
    rule8 = '审理(?:认定|查明).*?本院依照《'
    rule9 = '(?:以|如)下案件事实.*?本院认为'
    rule10 = '(?:确认|认定)(?:下列|下述)事实.*?本院认为'
    rule11 = '认定的事实为.*?本院认为'
    rule12 = '(?:确认|认定)(?:下列|下述)案件事实.*?本院认为'
    rule13 = '本案事实确认如下.*?本院认为'

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13]
    patterns = [re.compile(item, re.DOTALL) for item in rules]

    with codecs.open(path, mode='r', encoding='utf-8') as rfile:
        json_data = json.load(rfile)

    match_dic = {}
    chaming_len = []
    count = 0
    for k, v in json_data.items():
        for pattern in patterns:
            match = re.findall(pattern, v['caipanwenshu'])
            if len(match) > 0:
                chaming = ''.join(match[0].split('\n')[:-1])
                v['chaming'] = chaming
                if len(chaming) < 40:
                    print(chaming)
                    count += 1
                    break
                chaming_len.append(len(chaming))
                match_dic[k] = v
                break

    print(count)
    from scipy import stats
    print(stats.describe(np.array(chaming_len)))

    with codecs.open(outpath, mode='w+', encoding='utf-8') as wfile:
        json.dump(match_dic, wfile, ensure_ascii=False, indent=4)


def train_word2vec(path, outpath):
    with codecs.open(path, mode='r', encoding='utf-8') as rfile:
        json_data = json.load(rfile)

    f_w = codecs.open(outpath, mode='w+', encoding='utf-8')
    sentences = []
    config = Data()
    config.build_word_alphabet('data/chaming-train.json')
    config.build_word_alphabet('data/chaming-dev.json')
    config.build_word_alphabet('data/chaming-test.json')
    config.fix_alphabet()

    for k, v in json_data.items():
        chaming = v['chaming']
        chaming_cut_list = list(jieba.cut(chaming))
        chaming_ids_list = [str(config.word_alphabet.get_index(word)) for word in chaming_cut_list]
        sentences.append(chaming_ids_list)

        f_w.write(chaming + '\n')
        claims = v['claims_split']
        for claim in claims:
            claim_cut_list = list(jieba.cut(claim.strip()))
            claim_ids_list = [str(config.word_alphabet.get_index(word)) for word in claim_cut_list]
            sentences.append(claim_ids_list)
            f_w.write(claim.strip() + '\n')

    model = gensim.models.Word2Vec(sentences, min_count=1, size=300)
    model.save('data/word2vec.model')
    model.wv.save_word2vec_format('data/word2vec.txt', binary=False)
    f_w.close()
    return


def save_numpy(path, outpath):
    vectors_dic = {}
    vectors = []
    for line in codecs.open(path, mode='r', encoding='utf-8'):
        fields = line.strip().split()
        if len(fields) != 301:
            print('Error line:', line)

        vectors_dic[int(fields[0])] = [float(item) for item in fields[1:]]

    vectors_dic = sorted(vectors_dic.items(), key=lambda k: k[0])
    for (k, v) in vectors_dic:
        vectors.append(v)

    np.save(outpath, np.array(vectors))


if __name__ == '__main__':
    print(datetime.datetime.now())
    # extract_chaming('data/70487total-1.json', 'data/chaming.json')
    split_data('data/chaming.json')
    # train_word2vec(path='data/chaming.json', outpath='data/chaming.text')
    # save_numpy('data/word2vec.txt', 'data/word2vec.npy')