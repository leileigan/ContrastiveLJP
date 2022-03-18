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

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import optim
from torch.utils.data.dataloader import DataLoader

from models.model_HARNN_Contra_accu_term_v2 import LawModel, MoCo
from utils.config import Config, seed_rand
from utils.optim import ScheduledOptim
from data.dataset import load_dataset, CustomDataset, collate_qa_fn
np.set_printoptions(threshold=np.inf)
from transformers import AutoTokenizer


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
    print("Accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (accu_macro_f1, accu_macro_precision, accu_macro_recall))
    print("Law task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (law_macro_f1, law_macro_precision, law_macro_recall))
    print("Term task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (term_macro_f1, term_macro_precision, term_macro_recall))

    return (accu_macro_f1 + law_macro_f1 + term_macro_f1) / 3
    # return accu_macro_f1


def evaluate(model, valid_dataloader, name):
    
    model.eval()
    ground_accu_y, ground_law_y, ground_term_y  = [], [], []
    predicts_accu_y, predicts_law_y, predicts_term_y = [], [], []

    for batch_idx, datapoint in enumerate(valid_dataloader):
        fact_list, accu_label_lists, law_label_lists, term_lists = datapoint
        _, _, _, _, accu_preds, law_preds, term_preds = model(fact_list, accu_label_lists,law_label_lists, term_lists)

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
    # confused_matrix_law = confusion_matrix(ground_law_y, predicts_law_y)
    # confused_matrix_term = confusion_matrix(ground_term_y, predicts_term_y)
    print("Accu task accuracy: %.4f, Law task accuracy: %.4f, Term task accuracy: %.4f" % (accu_accuracy, law_accuracy, term_accuracy)) 
    print("Confused matrix accu:", confused_matrix_accu[1])
    # print("Confused matrix law:", confused_matrix_law)
    # print("Confused matirx term:", confused_matrix_term)
    score = get_result(ground_accu_y, predicts_accu_y, ground_law_y, predicts_law_y, ground_term_y, predicts_term_y, name)

    return score


def train(model, dataset, config: Config):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    train_dataloader = DataLoader(train_data_set, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_qa_fn, drop_last=True)
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn, drop_last=True)
    test_dataloader = DataLoader(test_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn, drop_last=True)

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
            cl_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = model(fact_list, accu_label_lists, law_label_lists, term_lists)
            # print("accu loss: %.2f, law loss: %.2f, term loss: %.2f, cl loss: %.2f" % (accu_loss, law_loss, term_loss, cl_loss))
            if idx >= config.warm_epoch:
                loss = (1 - config.alpha) * ((accu_loss + law_loss +term_loss) / batch_size) + config.alpha * cl_loss
            else:
                loss = (accu_loss + law_loss + term_loss) / batch_size
            
            sample_loss += loss.data
            sample_accu_loss += accu_loss.data / batch_size
            sample_law_loss += law_loss.data / batch_size
            sample_term_loss += term_loss.data / batch_size
            sample_cl_loss += cl_loss.data

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
                print("Instance: %s; Time: %.2fs; loss: %.2f; accu loss %.2f; law loss %.2f; term loss %.2f; cl loss %.2f; accu acc %.4f; law acc %.4f; term acc %.4f" % 
                ((batch_idx + 1), temp_cost, sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, sample_cl_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0
                sample_cl_loss = 0

            loss.backward()
            # optimizer.step_and_update_lr()
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_dataloader, "Dev")

        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = config.save_model_dir + f"epoch{str(idx)}.ckpt"
            torch.save(model.state_dict(), model_name)
            # evaluate test data
            _ = evaluate(model, test_dataloader, "Test")
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
    parser.add_argument('--savemodel', default="/data/home/ganleilei/law/ContrastiveLJP/models/harnnContra_v2/")
    parser.add_argument('--savedset', default="/data/home/ganleilei/law/ContrastiveLJP/models/harnnContra_v2/data")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--embedding_path', default='/data/home/ganleilei/law/ContrastiveLJP/cail_thulac.npy')
    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=510)
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
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--warm_epoch', default=0, type=int)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--sample_size', default='all', type=str)
    parser.add_argument('--moco_queue_size', default=65536, type=int)
    parser.add_argument('--moco_momentum', default=0.999, type=float)
    parser.add_argument('--moco_temperature', default=0.07, type=float)

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
            config.moco_temperature = args.moco_temperature
            config.moco_queue_size = args.moco_queue_size
            config.moco_momentum = args.moco_momentum
            config.warm_epoch = args.warm_epoch
            config.alpha = args.alpha
            
            config.load_word_pretrain_emb(args.embedding_path)
            save_data_setting(config, config.save_dset_dir + '.dset')
            model = MoCo(config)

        config.show_data_summary()

        if config.HP_gpu:
            model.cuda()

        print("\nLoading data...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        if args.sample_size != 'all':
            print("sample size:", args.sample_size)
            sample_size = int(args.sample_size)
            sample_train_data  = {}
            for key in train_data.keys():
                sample_train_data[key] = train_data[key][:sample_size]
        else:
            sample_train_data = train_data
        
        train_dataset = CustomDataset(sample_train_data, tokenizer, config.MAX_SENTENCE_LENGTH)
        valid_dataset = CustomDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH)
        test_dataset = CustomDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH)

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
