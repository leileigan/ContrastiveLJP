# encoding: utf-8
from transformers import BertTokenizer, MT5ForConditionalGeneration, MT5EncoderModel, AutoTokenizer
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch import cuda
import os
import codecs
import json
import jieba
import sys
from tqdm import tqdm
from rouge import Rouge
import pickle
sys.setrecursionlimit(int(1e6))

DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_dataset(path):
    train_path = os.path.join(path, "train_processed_thulac_Legal_basis.pkl")
    valid_path = os.path.join(path, "valid_processed_thulac_Legal_basis.pkl")
    test_path = os.path.join(path, "test_processed_thulac_Legal_basis.pkl")
    
    train_dataset = pickle.load(open(train_path, mode='rb'))
    valid_dataset = pickle.load(open(valid_path, mode='rb'))
    test_dataset = pickle.load(open(test_path, mode='rb'))

    print("train dataset sample len:", len(train_dataset['law_label_lists']))
    return train_dataset, valid_dataset, test_dataset

class CustomDataset(Dataset):

    def __init__(self, data, tokenizer, max_len, id2word_dict):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.id2word_dict = id2word_dict

    def __len__(self):
        return len(self.data['fact_list'])

    def _convert_ids_to_sent(self, fact):
        #fact: [max_sent_num, max_sent_len]
        mask = np.array(fact) == 164672
        mask = ~mask
        seq_len = mask.sum(1) #[max_sent_num]
        sent_num_mask = seq_len == 0
        sent_num_mask = ~sent_num_mask
        sent_num = sent_num_mask.sum(0)
        raw_text = []
        for s_idx in range(sent_num):
            cur_seq_len = seq_len[s_idx]
            raw_text.extend(fact[s_idx][:cur_seq_len])

        return [self.id2word_dict[ids] for ids in raw_text]
    

    def __getitem__(self, index):
        fact_list = self.data['fact_list'][index]
        raw_fact_list = self._convert_ids_to_sent(fact_list) 
        accu_label_lists = self.data['accu_label_lists'][index]
        law_label_lists = self.data['law_label_lists'][index]
        term_lists = self.data['term_lists'][index]
        
        return fact_list, raw_fact_list, accu_label_lists, law_label_lists, term_lists 


def collate_qa_fn(batch):
    
    batch_fact_list, batch_raw_fact_list, batch_law_label_lists, batch_accu_label_lists, batch_term_lists = [], [], [], [], []
    for item in batch:
        batch_fact_list.append(item[0])
        batch_raw_fact_list.append(item[1])
        batch_accu_label_lists.append(item[2])
        batch_law_label_lists.append(item[3])
        batch_term_lists.append(item[4])

    padded_fact_list = torch.LongTensor(batch_fact_list).to(DEVICE)
    padded_accu_label_lists = torch.LongTensor(batch_accu_label_lists).to(DEVICE)
    padded_law_label_lists = torch.LongTensor(batch_law_label_lists).to(DEVICE)
    padded_term_lists = torch.LongTensor(batch_term_lists).to(DEVICE)

    return padded_fact_list, batch_raw_fact_list, padded_accu_label_lists, padded_law_label_lists, padded_term_lists


class NeurJudgeDataset(Dataset):

    def __init__(self, data, tokenizer, max_len, id2word_dict):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.id2word_dict = id2word_dict

    def __len__(self):
        return len(self.data['fact_list'])

    def _convert_ids_to_sent(self, fact):
        #fact: [max_sent_len, ]
        mask = np.array(fact) == 164672
        mask = ~mask
        seq_len = mask.sum(0)
        return [self.id2word_dict[id] for id in fact[:seq_len]]
    

    def __getitem__(self, index):
        fact_list = self.data['fact_list'][index]
        raw_fact_list = self._convert_ids_to_sent(fact_list) 
        accu_label_lists = self.data['accu_label_lists'][index]
        law_label_lists = self.data['law_label_lists'][index]
        term_lists = self.data['term_lists'][index]
        
        return fact_list, raw_fact_list, accu_label_lists, law_label_lists, term_lists 


def collate_neur_judge_fn(batch):
    
    batch_fact_list, batch_raw_fact_list, batch_law_label_lists, batch_accu_label_lists, batch_term_lists = [], [], [], [], []
    for item in batch:
        batch_fact_list.append(item[0])
        batch_raw_fact_list.append(item[1])
        batch_accu_label_lists.append(item[2])
        batch_law_label_lists.append(item[3])
        batch_term_lists.append(item[4])

    padded_fact_list = torch.LongTensor(batch_fact_list).to(DEVICE)
    padded_accu_label_lists = torch.LongTensor(batch_accu_label_lists).to(DEVICE)
    padded_law_label_lists = torch.LongTensor(batch_law_label_lists).to(DEVICE)
    padded_term_lists = torch.LongTensor(batch_term_lists).to(DEVICE)

    return padded_fact_list, batch_raw_fact_list, padded_accu_label_lists, padded_law_label_lists, padded_term_lists


if __name__ == "__main__":
    data_path = "/data/home/ganleilei/law/ContrastiveLJP/"
    tokenizer_path = "/data/home/ganleilei/bert/bert-base-chinese/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_dataset, valid_dataset, test_dataset = load_dataset(data_path)
    custom_dataset = CustomDataset(train_dataset, tokenizer, 512)
