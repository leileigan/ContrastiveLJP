#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn

import argparse
from cProfile import label
import copy
import datetime
import json
import os
import pickle
import random
import sys
import time
from typing import List

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, confusion_matrix)
from torch import optim, threshold
from torch.utils.data.dataloader import DataLoader

from models.model_HARNN import LawModel
from utils.config import Config, seed_rand
from utils.functions import load_data
from utils.optim import ScheduledOptim
from data.dataset import load_dataset, CustomDataset, collate_qa_fn

from transformers import AutoTokenizer

np.set_printoptions(threshold=np.inf)

SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


def load_model(model_dir, config, gpu):
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

    return model

def str2bool(params):
    return True if params.lower() == 'true' else False

def save_data_setting(data: Config, save_file):
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
    print("Accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" %
          (accu_macro_f1, accu_macro_precision, accu_macro_recall))
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


def decode_id2sentence(fact_ids, id2word_dict, fact_sent_num: int, fact_sent_len: List[int]):
    fact = []
    for s_num in range(fact_sent_num):
        s_len = fact_sent_len[s_num] 
        fact.extend([id2word_dict[id.item()] for id in fact_ids[s_num][:s_len]])
    return ' '.join(fact)


def evaluate(model, valid_dataloader, name, config: Config):
    id2word_dict = {i: w for (w, i) in config.word2id_dict.items()}
    model.eval()
    ground_accu_y, ground_law_y, ground_term_y  = [], [], []
    predicts_accu_y, predicts_law_y, predicts_term_y = [], [], []

    for batch_idx, datapoint in enumerate(valid_dataloader):
        fact_list, accu_label_lists, law_label_lists, term_lists = datapoint
        fact_mask = ~fact_list.eq(config.word2id_dict["BLANK"])
        fact_sent_len = torch.sum(fact_mask, dim=-1) # [batch_size, sent_num]
        fact_doc_mask = fact_sent_len.bool()
        fact_sent_num = torch.sum(fact_doc_mask, dim=-1) #[batch_size]
        # print('fact len:', fact_doc_len.size())
        _, _, _, accu_preds, law_preds, term_preds = model.neg_log_likelihood_loss(fact_list, accu_label_lists,law_label_lists, term_lists)

        ground_accu_y.extend(accu_label_lists.tolist())
        ground_law_y.extend(law_label_lists.tolist())
        ground_term_y.extend(term_lists.tolist())

        predicts_accu_y.extend(accu_preds.tolist())
        predicts_law_y.extend(law_preds.tolist())
        predicts_term_y.extend(term_preds.tolist())

        """
        for idx, accu_label in enumerate(accu_label_lists):
            if accu_label == 1 and accu_preds[idx] != accu_label:
                fact_ = fact_list[idx]
                decode_fact = decode_id2sentence(fact_, id2word_dict, fact_sent_num[idx].item(), fact_sent_len[idx])
                print(decode_fact)
                print(f"accu label: {accu_label}, predict label: {accu_preds[idx]}")
        """

    accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
    law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
    term_accuracy = accuracy_score(ground_term_y, predicts_term_y)
    confused_matrix_accu = confusion_matrix(ground_accu_y, predicts_accu_y)
    confused_matrix_law = confusion_matrix(ground_law_y, predicts_law_y)
    confused_matrix_term = confusion_matrix(ground_term_y, predicts_term_y)
    
    print("Confused matrix accu:", confused_matrix_accu[1])
    print("Accu task accuracy: %.4f, Law task accuracy: %.4f, Term task accuracy: %.4f" % (accu_accuracy, law_accuracy, term_accuracy)) 
    score = get_result(ground_accu_y, predicts_accu_y, ground_law_y, predicts_law_y, ground_term_y, predicts_term_y, name)
    # print("accu classification report:", classification_report(ground_accu_y, predicts_accu_y))
    # print("law classification report:", classification_report(ground_law_y, predicts_law_y))
    # print("term classification report:", classification_report(ground_term_y, predicts_term_y))

    return score


def train(model, dataset, config: Config):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    train_dataloader = DataLoader(train_data_set, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_qa_fn)
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)
    test_dataloader = DataLoader(test_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)

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
        sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss = 0, 0, 0, 0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size

        ground_accu_y, predicts_accu_y = [], []
        ground_law_y, predicts_law_y = [], []
        ground_term_y, predicts_term_y = [], []

        for batch_idx, datapoint in enumerate(train_dataloader):
            fact_list, accu_label_lists, law_label_lists, term_lists = datapoint
            accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = model.neg_log_likelihood_loss(fact_list, 
                                                                                                              accu_label_lists, 
                                                                                                              law_label_lists, 
                                                                                                              term_lists
                                                                                                              )
            loss = (accu_loss + term_loss + law_loss) / batch_size
            sample_loss += loss.data
            sample_accu_loss += accu_loss.data
            sample_law_loss += law_loss.data
            sample_term_loss += term_loss.data

            ground_accu_y.extend(accu_label_lists.tolist())
            ground_law_y.extend(law_label_lists.tolist())
            ground_term_y.extend(term_lists.tolist())

            predicts_accu_y.extend(accu_preds.tolist())
            predicts_law_y.extend(law_preds.tolist())
            predicts_term_y.extend(term_preds.tolist())

            if (batch_idx + 1 ) % 100 == 0:
                cur_accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
                cur_law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
                cur_term_accuracy = accuracy_score(ground_term_y, predicts_term_y)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("Instance: %s; Time: %.2fs; loss: %.4f; accu loss %.4f; law loss %.4f; term loss %.4f; accu acc %.4f; law acc %.4f; term acc %.4f" % 
                ((batch_idx + 1), temp_cost, sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0

            loss.backward()
            # optimizer.step_and_update_lr()
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_dataloader, "Dev", config)

        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = config.save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            # evaluate test data
            _ = evaluate(model, test_dataloader, "Test", config)
        else:
            no_imporv_epoch += 1
            if no_imporv_epoch >= 10:
                print("early stop")
                break


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Contrastive Legal Judgement Prediction')
    parser.add_argument('--data_path', default="/data/home/ganleilei/law/ContrastiveLJP/")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="/data/home/ganleilei/law/ContrastiveLJP/models/harnn")
    parser.add_argument('--savedset', default="/data/home/ganleilei/law/ContrastiveLJP/models/harnn/data")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--embedding_path', default='/data/home/ganleilei/law/ContrastiveLJP/cail_thulac.npy')
    parser.add_argument('--word2id_dict', default='/data/home/ganleilei/law/ContrastiveLJP/w2id_thulac.pkl')
    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=250)
    parser.add_argument('--hops', default=3)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--max_decoder_step', default=100)

    parser.add_argument('--HP_iteration', default=30, type=int)
    parser.add_argument('--HP_batch_size', default=128, type=int)
    parser.add_argument('--HP_hidden_dim', default=256, type=int)
    parser.add_argument('--HP_dropout', default=0.2, type=float)
    parser.add_argument('--HP_lstmdropout', default=0.5, type=float)
    parser.add_argument('--HP_lstm_layer', default=1, type=int)
    parser.add_argument('--HP_lr', default=1e-3, type=float)
    parser.add_argument('--HP_lr_decay', default=0.05, type=float)
    parser.add_argument('--HP_freeze_word_emb', action='store_true')

    parser.add_argument('--use_warmup_adam', default='False')
    parser.add_argument('--use_adam', default='True')
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--bert_path', type=str)

    args = parser.parse_args()
    print(args)

    seed_rand(args.seed)
    status = args.status

    if status == 'train':
        if os.path.exists(args.loadmodel) and os.path.exists(args.savedset):
            print('Load model path:', args.loadmodel)
            print('Load save dataset:', args.savedset)
            config: Config = load_data_setting(args.savedset)
            model = LawModel(config)
            model.load_state_dict(torch.load(args.loadmodel))
        else:
            print('New config....')
            config = Config()
            config.HP_iteration = args.HP_iteration
            config.HP_batch_size = args.HP_batch_size
            config.HP_hidden_dim = args.HP_hidden_dim
            config.HP_dropout = args.HP_dropout
            config.HP_lstm_layer = args.HP_lstm_layer
            config.HP_lr = args.HP_lr
            config.MAX_SENTENCE_LENGTH = args.MAX_SENTENCE_LENGTH
            config.HP_lr_decay = args.HP_lr_decay
            config.save_model_dir = args.savemodel
            config.HP_freeze_word_emb = args.HP_freeze_word_emb
            if not os.path.exists(config.save_model_dir):
                os.mkdir(config.save_model_dir)
            
            config.save_dset_dir = args.savedset
            if not os.path.exists(config.save_dset_dir):
                os.mkdir(config.save_dset_dir)

            config.use_warmup_adam = str2bool(args.use_warmup_adam)
            config.use_adam = str2bool(args.use_adam)
            
            config.load_word_pretrain_emb(args.embedding_path)
            config.word2id_dict = pickle.load(open(args.word2id_dict, 'rb'))
            save_data_setting(config, config.save_dset_dir + '.dset')
            model = LawModel(config)

        config.show_data_summary()

        if config.HP_gpu:
            model.cuda()

        print("\nLoading data...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        train_dataset = CustomDataset(train_data, tokenizer, 512)
        valid_dataset = CustomDataset(valid_data, tokenizer, 512)
        test_dataset = CustomDataset(test_data, tokenizer, 512)

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
        config.word2id_dict = pickle.load(open("/data/home/ganleilei/law/ContrastiveLJP/w2id_thulac.pkl", 'rb'))
        tokenizer_path = "/data/home/ganleilei/bert/bert-base-chinese/"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        train_dataset = CustomDataset(train_data, tokenizer, 512)
        valid_dataset = CustomDataset(valid_data, tokenizer, 512)
        test_dataset = CustomDataset(test_data, tokenizer, 512)
        train_dataloader = DataLoader(train_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)

        print("\nFinish loading data!")
        model = load_model(args.loadmodel, config, True)
        score = evaluate(model, test_dataloader, "Test", config)
