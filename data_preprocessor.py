# -*- coding: utf-8 -*-
import codecs
import json
import datetime
import os, shutil
from sklearn.model_selection import train_test_split


def read_data(path):
    with codecs.open(path, 'r', 'utf-8') as json_data:
        all_data = json.load(json_data)

    all_data = list(all_data.items())
    print("total data size:", len(all_data))
    train_data, other_data = train_test_split(all_data,  test_size=0.2, random_state=2020, shuffle=True)
    valid_data, test_data = train_test_split(other_data,  test_size=0.5, random_state=2020, shuffle=True)

    print("train data size: ", len(train_data))
    print("val data size:", len(valid_data))
    print("test data size:", len(test_data))

    train_data_dic, valid_data_dic, test_data_dic = {}, {}, {}
    for item in train_data:
        train_data_dic[item[0]] = item[1]
    for item in valid_data:
        valid_data_dic[item[0]] = item[1]
    for item in test_data:
        test_data_dic[item[0]] = item[1]

    with open('./data/70487train-fact.json', 'w', encoding='utf-8') as train_file:
        json.dump(train_data_dic, train_file, indent=4, ensure_ascii=False)

    with open('./data/70487val-fact.json', 'w', encoding='utf-8') as val_file:
        json.dump(valid_data_dic, val_file, indent=4, ensure_ascii=False)

    with open('./data/70487test-fact.json', 'w', encoding='utf-8') as test_file:
        json.dump(test_data_dic, test_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    print(datetime.datetime.now())
    read_data('data/70487total-1.json.fact')