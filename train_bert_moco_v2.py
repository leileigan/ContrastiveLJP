#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn

import argparse
import copy
import datetime
import json
import os
import pickle
import random
import sys
import time
import warnings

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score, confusion_matrix,
                             precision_score, recall_score)
from torch import device, optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from models.model_BERT_MoCo_v2 import LawModel, MoCo
from utils.optim import ScheduledOptim
from utils.config import BertMocoConfig
from data.dataset import load_dataset, CustomDataset, collate_qa_fn
from transformers import AutoTokenizer
from tqdm import tqdm

SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

warnings.filterwarnings("ignore")
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model_decode(model_dir, config, dataset, name, gpu):
    config.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = LawModel(config)
    if config.HP_gpu:
        model = model.cuda()
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    model.load_state_dict(torch.load(model_dir))
    # model = torch.load(model_dir)
    score = evaluate(model, dataset, name)

    return

def str2bool(params):
    return True if params.lower() == 'true' else False

def save_data_setting(data: BertMocoConfig, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data

confused_labels = [1, 3, 4, 5, 6, 9, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 33, 34, 36, 37, 38, 41, 42, 44, 45, 48, 52, 53, 54,
                   55, 58, 60, 61, 69, 71, 72, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 86, 91, 92, 93, 95, 99, 100, 104, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117, 118]
def get_result(accu_target, accu_preds, law_target, law_preds, term_target, term_preds, mode):
    accu_macro_f1 = f1_score(accu_target, accu_preds, average="macro")
    accu_macro_precision = precision_score(accu_target, accu_preds, average="macro")
    accu_macro_recall = recall_score(accu_target, accu_preds, average="macro")

    confused_accu_macro_f1 = f1_score(accu_target, accu_preds, average='macro', labels=confused_labels)
    confused_accu_macro_precision = precision_score(accu_target, accu_preds, average='macro', labels=confused_labels)
    confused_accu_macro_recall = recall_score(accu_target, accu_preds, average='macro', labels=confused_labels)

    law_macro_f1 = f1_score(law_target, law_preds, average="macro")
    law_macro_precision = precision_score(law_target, law_preds, average="macro")
    law_macro_recall = recall_score(law_target, law_preds, average="macro")

    term_macro_f1 = f1_score(term_target, term_preds, average="macro")
    term_macro_precision = precision_score(term_target, term_preds, average="macro")
    term_macro_recall = recall_score(term_target, term_preds, average="macro")

    print("Confused accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (
        confused_accu_macro_f1, confused_accu_macro_precision, confused_accu_macro_recall))
    print("Accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (accu_macro_f1, accu_macro_precision, accu_macro_recall))
    print("Law task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (law_macro_f1, law_macro_precision, law_macro_recall))
    print("Term task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (term_macro_f1, term_macro_precision, term_macro_recall))

    return (accu_macro_f1 + law_macro_f1 + term_macro_f1) / 3


def data_initialization(data, train_file, dev_file, test_file):
    print('begin building word alphabet set')
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.fix_alphabet()
    print('word alphabet size:', data.word_alphabet_size)
    print('label alphabet size:', data.label_alphabet_size)
    print('label alphabet:', data.label_alphabet.instances)
    return data


def evaluate(model, valid_dataloader, config, name):

    model.eval()
    ground_accu_y, ground_law_y, ground_term_y  = [], [], []
    predicts_accu_y, predicts_law_y, predicts_term_y = [], [], []
    #if torch.cuda.device_count() > 1:#判断是不是有多个GPU
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # 就这一行
    #    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    with torch.no_grad():
        for batch_idx, datapoint in tqdm(enumerate(valid_dataloader)):
            fact_list, accu_label_lists, law_label_lists, term_lists = datapoint
            fact_list = x_given_bert(config, fact_list)
            _, _, _,_ , accu_preds, law_preds, term_preds = model(fact_list, accu_label_lists, law_label_lists, term_lists)
            ground_accu_y.extend(accu_label_lists.tolist())
            ground_law_y.extend(law_label_lists.tolist())
            ground_term_y.extend(term_lists.tolist())
            
            predicts_accu_y.extend(accu_preds.tolist())
            predicts_law_y.extend(law_preds.tolist())
            predicts_term_y.extend(term_preds.tolist())
        
    accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
    law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
    term_accuracy = accuracy_score(ground_term_y, predicts_term_y)
    confused_matrix_accu = confusion_matrix(ground_accu_y, predicts_accu_y)
    print("Confused matrix accu:", confused_matrix_accu[1])
    print("Accu task accuracy: %.4f, Law task accuracy: %.4f, Term task accuracy: %.4f" % (accu_accuracy, law_accuracy, term_accuracy)) 
    score = get_result(ground_accu_y, predicts_accu_y, ground_law_y, predicts_law_y, ground_term_y, predicts_term_y, name)

    return score


def x_given_bert(config: BertMocoConfig, fact_list):
    encode_output = config.tokenizer.batch_encode_plus(fact_list, max_length=500, pad_to_max_length=True, truncation=True)
    input_ids, attention_mask = torch.LongTensor(encode_output["input_ids"]).to(config.device), torch.LongTensor(encode_output["attention_mask"]).to(config.device)
    seq_len = attention_mask.sum(-1)
    # print("input ids size:", input_ids.size())
    # print("seq len size:", seq_len)
    # print("attention mask size:", attention_mask.size())
    return input_ids, seq_len, attention_mask


def train(model, dataset, config: BertMocoConfig):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    train_dataloader = DataLoader(train_data_set, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_qa_fn_1, drop_last=True) #config.HP_batch_size
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn_1, drop_last=True)
    test_dataloader = DataLoader(test_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn_1, drop_last=True)

    #if torch.cuda.device_count() > 1:#判断是不是有多个GPU
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # 就这一行
    #    model = nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))
    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    if config.use_warmup_adam:
        optimizer = ScheduledOptim(optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-9), d_model=256, n_warmup_steps=2000)
    elif config.use_sgd:
        optimizer = optim.SGD(parameters, lr=config.HP_lr, momentum=config.HP_momentum)
    elif config.use_adam:
        optimizer = optim.AdamW(parameters, lr=config.HP_lr)
    elif config.use_bert:
        optimizer = optim.Adam(parameters, lr=5e-6)  # fine tuning
    else:
        raise ValueError("Unknown optimizer")
    print('optimizer: ', optimizer)

    best_dev = -1
    no_imporv_epoch = 0
    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        optimizer = lr_decay(optimizer, idx, config.HP_lr_decay, config.HP_lr)
        sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, sample_cl_loss = 0, 0, 0, 0, 0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size
        ground_accu_y, predicts_accu_y = [], []
        ground_law_y, predicts_law_y = [], []
        ground_term_y, predicts_term_y = [], []

        for batch_idx, datapoint in enumerate(train_dataloader):
            
            fact_list, accu_label_lists, law_label_lists, term_lists = datapoint
            fact_list = x_given_bert(config, fact_list)

            contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = model(fact_list, accu_label_lists, law_label_lists, term_lists)
            loss = (1-config.HP_alpha) * (accu_loss + term_loss + law_loss) / batch_size + config.HP_alpha * contra_loss
            sample_loss += loss.data
            sample_accu_loss += accu_loss.data / config.HP_batch_size
            sample_law_loss += law_loss.data / config.HP_batch_size
            sample_term_loss += term_loss.data / config.HP_batch_size
            sample_cl_loss += contra_loss.data

            ground_accu_y.extend(accu_label_lists.tolist())
            ground_law_y.extend(law_label_lists.tolist())
            ground_term_y.extend(term_lists.tolist())

            predicts_accu_y.extend(accu_preds.tolist())
            predicts_law_y.extend(law_preds.tolist())
            predicts_term_y.extend(term_preds.tolist())

            cur_accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
            cur_law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
            cur_term_accuracy = accuracy_score(ground_term_y, predicts_term_y)

            if (batch_idx + 1 ) % 100 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("Instance: %s; Time: %.2fs; loss: %.2f; accu loss %.2f; law loss %.2f; term loss %.2f; contra loss: %.2f, accu acc %.4f; law acc %.4f; term acc %.4f" % 
                ((batch_idx + 1), temp_cost, sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, sample_cl_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0
                sample_cl_loss = 0
            #print('batch_idx', batch_idx)
            #if batch_idx == 101: break
            loss.backward()
            # optimizer.step_and_update_lr()
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_dataloader, config, "Dev")
        #_ = evaluate(model, test_dataloader, config, "Test")
        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = os.path.join(config.save_model_dir, f"epoch{idx}.ckpt")
            torch.save(model.state_dict(), model_name)
            # evaluate test data
            _ = evaluate(model, test_dataloader, config, "Test")
        else:
            no_imporv_epoch += 1
            if no_imporv_epoch >= 10:
                print("early stop")
                break

def collate_qa_fn_1(batch):
    """
    max_fact_char_ids = max(x["fact_char_ids"].size(0) for x in batch)
    for field in ["fact_char_ids", "fact_char_mask"]:
        pad_output = torch.full([batch_size, max_fact_char_ids], 0, dtype=batch[0][field].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field]
            pad_output[sample_idx][: data.size(0)] = data
        output[field] = pad_output
    """
    DEVICE= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_fact_list, batch_law_label_lists, batch_accu_label_lists, batch_term_lists = [], [], [], []
    for item in batch:
        batch_fact_list.append(item[0])
        batch_accu_label_lists.append(item[1])
        batch_law_label_lists.append(item[2])
        batch_term_lists.append(item[3])

    padded_accu_label_lists = torch.LongTensor(batch_accu_label_lists).to(DEVICE)
    padded_law_label_lists = torch.LongTensor(batch_law_label_lists).to(DEVICE)
    padded_term_lists = torch.LongTensor(batch_term_lists).to(DEVICE)

    return batch_fact_list, padded_accu_label_lists, padded_law_label_lists, padded_term_lists

if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Contrastive Legal Judgement Prediction')
    parser.add_argument('--data_path', default="/data/home/ganleilei/law/ContrastiveLJP/bert/")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="/data/home/ganleilei/law/ContrastiveLJP/models/bert_moco_v2/")
    parser.add_argument('--savedset', default="/data/home/ganleilei/law/ContrastiveLJP/models/bert_moco_v2/data.dset")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=250)
    parser.add_argument('--hops', default=3)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--max_decoder_step', default=100)

    parser.add_argument('--HP_iteration', default=30)
    parser.add_argument('--HP_batch_size', default=16, type=int)
    parser.add_argument('--HP_hidden_dim', default=256)
    parser.add_argument('--HP_dropout', default=0.2)
    parser.add_argument('--HP_lstmdropout', default=0.5)
    parser.add_argument('--HP_lstm_layer', default=1)
    parser.add_argument('--HP_lr', default=5e-5, type=float)
    parser.add_argument('--HP_lr_decay', default=0.05, type=float)

    parser.add_argument('--use_warmup_adam', default='False')
    parser.add_argument('--use_adam', default='True')

    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)

    args = parser.parse_args()
    print(args)
    status = args.status

    if status == 'train':
        if os.path.exists(args.loadmodel) and os.path.exists(args.savedset):
            print('Load model path:', args.loadmodel)
            print('Load save dataset:', args.savedset)
            config: BertMocoConfig = load_data_setting(args.savedset)
            model = LawModel(config)
            model.load_state_dict(torch.load(args.loadmodel))
        else:
            print('New config....')
            config = BertMocoConfig()
            config.HP_iteration = args.HP_iteration
            config.HP_batch_size = args.HP_batch_size
            config.HP_hidden_dim = args.HP_hidden_dim
            config.HP_dropout = args.HP_dropout
            config.HP_lstm_layer = args.HP_lstm_layer
            config.HP_lr = args.HP_lr
            config.MAX_SENTENCE_LENGTH = args.MAX_SENTENCE_LENGTH
            config.HP_lr_decay = args.HP_lr_decay
            config.save_model_dir = args.savemodel
            config.save_dset_dir = args.savedset
            config.use_warmup_adam = str2bool(args.use_warmup_adam)
            config.use_adam = str2bool(args.use_adam)
            config.HP_temperature = args.temperature
            config.HP_alpha = args.alpha

            save_data_setting(config, config.save_dset_dir)
            model = MoCo(config)

        config.show_data_summary()
        model.to(DEVICE)

        print("\nLoading data...")
        #tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        train_dataset = CustomDataset(train_data, config.tokenizer, 512)
        valid_dataset = CustomDataset(valid_data, config.tokenizer, 512)
        test_dataset = CustomDataset(test_data, config.tokenizer, 512)

        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        data_dict = {
            "train_data_set": train_dataset,
            "test_data_set": test_dataset,
            "valid_data_set": valid_dataset
        }
        
        print("\nTraining...")
        train(model, data_dict, config)

    elif status == 'test':
        if os.path.exists(args.loadmodel) is False or os.path.exists(args.savedset) is False:
            print('File path does not exit: %s and %s' % (args.loadmodel, args.savedset))
            exit(1)

        config = load_data_setting(args.savedset)
        print("\nLoading data...")
        test_data = load_data(args.test, config)
        decode_results = load_model_decode(args.loadmodel, config, test_data, 'Test', True)