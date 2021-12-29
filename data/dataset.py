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

    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data['fact_list'])

    def __getitem__(self, index):
        fact_list = self.data['fact_list'][index]
        accu_label_lists = self.data['accu_label_lists'][index]
        law_label_lists = self.data['law_label_lists'][index]
        term_lists = self.data['term_lists'][index]
        #raw_fact_lists = self.data['raw_fact_lists'][index]
        #raw_fact_output = self.tokenizer.batch_encode_plus([raw_fact_lists], max_length= self.max_len, padding='max_length',return_tensors='pt', truncation=True)
        #fact_char_ids = raw_fact_output['input_ids'].squeeze()
        #fact_char_mask = raw_fact_output['attention_mask'].squeeze()
        
        return fact_list, accu_label_lists, law_label_lists, term_lists 


def collate_qa_fn(batch):
    """
    max_fact_char_ids = max(x["fact_char_ids"].size(0) for x in batch)
    for field in ["fact_char_ids", "fact_char_mask"]:
        pad_output = torch.full([batch_size, max_fact_char_ids], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.size(0)] = data
        output[field] = pad_output
    """
    batch_fact_list, batch_law_label_lists, batch_accu_label_lists, batch_term_lists = [], [], [], []
    for item in batch:
        batch_fact_list.append(item[0])
        batch_accu_label_lists.append(item[1])
        batch_law_label_lists.append(item[2])
        batch_term_lists.append(item[3])

    padded_fact_list = torch.LongTensor(batch_fact_list).to(DEVICE)
    padded_accu_label_lists = torch.LongTensor(batch_accu_label_lists).to(DEVICE)
    padded_law_label_lists = torch.LongTensor(batch_law_label_lists).to(DEVICE)
    padded_term_lists = torch.LongTensor(batch_term_lists).to(DEVICE)

    return padded_fact_list, padded_accu_label_lists, padded_law_label_lists, padded_term_lists


if __name__ == "__main__":
    data_path = "/data/home/ganleilei/law/ContrastiveLJP/"
    tokenizer_path = "/data/home/ganleilei/bert/bert-base-chinese/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    train_dataset, valid_dataset, test_dataset = load_dataset(data_path)
    custom_dataset = CustomDataset(train_dataset, tokenizer, 512)
