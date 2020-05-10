# -*- coding: utf-8 -*-
# @Author: Leilei Gan
# @Date:   2020-01-12 14:23:06

import sys
import numpy as np
from utils.alphabet import Alphabet
from gensim.models import KeyedVectors
import codecs, os
import torch

NULLKEY = "-null-"

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def read_instance(input_dir, word_alphabet, label_alphabet, number_normalized, max_sent_length):

    total_file_count = len(os.listdir(input_dir))
    instances = []
    for file_name in os.listdir(input_dir):
        in_lines = open(input_dir + '/' + file_name, 'r').readlines()

        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if "chaming" in line:
                cm_text = in_lines[idx + 1].strip()

            elif "suqiu-split" in line:
                suqiu_text = in_lines[idx + 1].strip()

            elif "suqiu-labels" in line:
                suqiu_label = in_lines[idx + 1].strip()
                suqiu_label_list = suqiu_label[2:-2].split("', '")
                suqiu_list = suqiu_text[2:-2].split("', '")
                if len(suqiu_list) != len(suqiu_label_list):
                    print("wrong formate: ", file_name)
                    break

                for suqiu_idx in range(len(suqiu_list)):
                    words = list(cm_text) + list(suqiu_list[suqiu_idx])
                    label = suqiu_label_list[suqiu_idx]
                    word_Ids = [word_alphabet.get_index(word) for word in words]
                    label_Ids = label_alphabet.get_index(label)
                    instances.append((words, label, word_Ids, label_Ids))

    print("finish loading %d files with %d instances" % (total_file_count, len(instances)))
    return instances


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

if __name__=="__main__":
    read_instance('./../data/cases_dev', None, None, False, 512)