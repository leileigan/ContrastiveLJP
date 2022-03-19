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
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from torch import optim
from torch.utils.data.dataloader import DataLoader

from models.model_CNN import LawModel
from utils.config import Config
from utils.functions import generate_batch_instance, load_data
from utils.optim import ScheduledOptim
from data.dataset import load_dataset, CustomDataset, collate_qa_fn

from transformers import AutoTokenizer

SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)


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

def get_result(target, preds, claims_type, mode):
    # target = np.argmax(np.array(target), axis=1).tolist()
    macro_f1 = f1_score(target, preds, average="macro")
    macro_precision = precision_score(target, preds, average="macro")
    macro_recall = recall_score(target, preds, average="macro")

    micro_f1 = f1_score(target, preds, average="micro")
    print(("%s," % mode) + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1) + str(
        "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))

    from collections import Counter
    sorted_target = sorted(Counter(target).items())
    sorted_preds = sorted(Counter(preds).items())

    print('ground:')
    for item in sorted_target:
        print(item)

    print('predicts:')
    for item in sorted_preds:
        print(item)

    # print("ground: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_target[0][1], sorted_target[1][1],
    #                                                                sorted_target[2][1]))
    # print("predicts: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_preds[0][1], sorted_preds[1][1],
    #                                                                sorted_preds[2][1]))

    target_names = ['驳回诉请', '部分支持', "支持诉请"]
    # if mode == "test":
    if mode:
        print(classification_report(target, preds, target_names=target_names, digits=4))

    ss_ground_y, ss_predict_y = [], []
    for idx, cur_claim_type in enumerate(claims_type):
        if '诉讼费' == cur_claim_type:
            ss_ground_y.append(target[idx])
            ss_predict_y.append(preds[idx])

    print('诉讼费请求指标：')
    target_names = ['驳回诉请', '部分支持', "支持诉请"]
    print(classification_report(ss_ground_y, ss_predict_y, target_names=target_names, digits=4))

    return micro_f1 + macro_f1


def get_fact_result(target, preds):
    # target = pd.DataFrame(np.argmax(np.array(target), axis=1), columns=["0"])
    macro_f1 = f1_score(target, preds, average="macro")
    macro_precision = precision_score(target, preds, average="macro")
    macro_recall = recall_score(target, preds, average="macro")

    micro_f1 = f1_score(target, preds, average="micro")
    print("fact results: val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1) + str(
        "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))


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


def evaluate(model, dataset, name):

    model.eval()
    batch_size = 16
    train_num = len(dataset)
    total_batch = train_num // batch_size + 1

    start_time = time.time()

    total_loss, total_claim_loss = 0, 0

    claim_predicts  = []
    ground_claim_labels = []
    claims_type = []

    for batch_idx in range(total_batch):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size

        if end > train_num:
            end = train_num

        batch_instances = dataset[start: end]
        if len(batch_instances) <= 0:
            continue

        input_doc_batch, claims_labels_batch, fact_labels_batch, input_sentences_len_batch, input_claims_type, \
        batch_doc_texts, batch_claims_texts, input_claim_ids_batch, input_claims_num, input_claims_len, rate_fact_labels, wenshu_texts = generate_batch_instance(batch_instances, config.HP_gpu)

        claim_loss, batch_claim_predicts = model.neg_log_likelihood_loss(
        input_doc_batch, input_sentences_len_batch, fact_labels_batch, claims_labels_batch, input_claims_type,
            batch_doc_texts, batch_claims_texts, input_claim_ids_batch, input_claims_num, input_claims_len, rate_fact_labels, wenshu_texts)

        loss = claim_loss
        total_loss += loss.data
        total_claim_loss += claim_loss.data

        _, max_claims_num = claims_labels_batch.size()
        input_claims_num = input_claims_num.long().cpu().tolist()

        batch_predicts_y, batch_ground_y = [], []
        for i in range(len(input_claims_num)):
            batch_ground_y.append(claims_labels_batch[i][ :input_claims_num[i]].cpu().tolist())
            batch_predicts_y.append(batch_claim_predicts[i][ :input_claims_num[i]].cpu().tolist())


        ground_claim_labels.extend(sum(batch_ground_y, []))
        claim_predicts.extend(sum(batch_predicts_y, []))
        claims_type.extend(sum(input_claims_type, []))

    score = get_result(ground_claim_labels, claim_predicts, claims_type, name)

    decode_time = time.time() - start_time
    speed = train_num / decode_time
    print(
        "%s finished. Time: %.2fs, speed: %.2fst/s,  total loss: %.4f, total claim loss: %.4f, score: %.4f" %
        (name, decode_time, speed, total_loss, total_claim_loss, score))
    return score

def train(model, dataset, config: Config):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]

    train_dataloader = DataLoader(train_data_set, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_qa_fn)
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)
    test_dataloader = DataLoader(test_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_qa_fn)

    train_num_samples = len(train_data_set)
    # batch_num = (train_num_samples * FLAGS.num_epochs) // FLAGS.batch_size + 1
    batch_num = (train_num_samples * config.HP_iteration) // config.HP_batch_size + 1

    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if config.use_warmup_adam:
        optimizer = ScheduledOptim(optim.Adam(parameters, betas=(0.9, 0.98), eps=1e-9), d_model=256, n_warmup_steps=2000)
    elif config.use_sgd:
        optimizer = optim.SGD(parameters, lr=config.HP_lr, momentum=config.HP_momentum)
    elif config.use_adam:
        optimizer = optim.Adam(parameters, lr=config.HP_lr)
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
        total_loss, total_accu_loss, total_law_loss, total_term_loss = 0, 0, 0, 0

        random.shuffle(train_data_set)

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size
        train_num = len(train_data_set)
        total_batch = train_num // batch_size + 1

        ground_accu_y, predicts_accu_y = [], []
        ground_law_y, predicts_law_y = [], []
        ground_term_y, predicts_term_y = [], []

        for batch_idx, datapoint in enumerate(train_dataloader):
            fact_list, law_label_lists, accu_label_lists, term_lists, _, _ = datapoint
            accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = model.neg_log_likelihood_loss(fact_list, 
                                                                                                              law_label_lists, 
                                                                                                              accu_label_lists, 
                                                                                                              term_lists, 
                                                                                                              config.sent_len, 
                                                                                                              config.doc_len
                                                                                                              )
            loss = accu_loss + term_loss + law_loss
            total_loss += loss.data
            sample_loss += loss.data

            batch_predicts_accu, batch_predicts_law, batch_predicts_term = [], [], []
            for i in range(len(accu_label_lists)):
                batch_predicts_accu.append(accu_preds[i].cpu().tolist())
                batch_predicts_law.append(law_preds[i].cpu().tolist())
                batch_predicts_term.append(term_preds[i].cpu().tolist())

            ground_accu_y.extend(accu_label_lists)
            ground_law_y.extend(law_label_lists)
            ground_term_y.extend(term_lists)

            predicts_accu_y.extend(accu_preds)
            predicts_law_y.extend(law_preds)
            predicts_term_y.extend(term_preds)

            cur_accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
            cur_law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
            cur_term_accuracy = accuracy_score(ground_term_y, predicts_term_y)

            if batch_idx % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("Instance: %s; Time: %.2fs; loss: %.4f; accu loss %.4f; law loss %.4f; term loss %.4f; accu acc %.4f; law acc %.4f; term acc %.4f" % 
                (batch_idx, temp_cost, sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0

            loss.backward()
            # optimizer.step_and_update_lr()
            optimizer.step()
            model.zero_grad()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %.4f" %
              (idx, epoch_cost, train_num / epoch_cost, total_loss))
        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_data_set, "Dev")

        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = config.save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            # evaluate test data
            _ = evaluate(model, test_data_set, "Test")
        else:
            no_imporv_epoch += 1
            if no_imporv_epoch >= 10:
                print("early stop")
                break


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Contrastive Legal Judgement Prediction')
    parser.add_argument('--data_path', default="./data/chaming-train.json.rate.date.bak")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="savemodels/cnn.logic")
    parser.add_argument('--savedset', default="savemodels/cnn.logic")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--embedding_dense_dim', default=300)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=250)
    parser.add_argument('--hops', default=3)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--max_decoder_step', default=100)

    parser.add_argument('--HP_iteration', default=100)
    parser.add_argument('--HP_batch_size', default=128)
    parser.add_argument('--HP_hidden_dim', default=256)
    parser.add_argument('--HP_dropout', default=0.2)
    parser.add_argument('--HP_lstmdropout', default=0.5)
    parser.add_argument('--HP_lstm_layer', default=1)
    parser.add_argument('--HP_lr', default=1e-3, type=float)
    parser.add_argument('--HP_lr_decay', default=0.05, type=float)

    parser.add_argument('--use_warmup_adam', default='False')
    parser.add_argument('--use_adam', default='True')

    args = parser.parse_args()

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
            config.facts = list(map(str, args.facts.split(",")))
            config.total_fact_num = len(config.facts)
            config.using_fact_num = len(config.facts) - 2
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
            config.rule1_lambda = float(args.rule1_lambda)

            config.build_word_alphabet(args.train)
            config.build_word_alphabet(args.dev)
            config.build_word_alphabet(args.test)
            config.fix_alphabet()
            print('word alphabet size:', config.word_alphabet_size)

            config.build_word_pretrain_emb('data/word2vec.dim200.txt')
            save_data_setting(config, config.save_dset_dir + '.dset')
            model = LawModel(config)

        config.show_data_summary()
        if config.HP_gpu:
            model.cuda()

        print("\nLoading data...")
        data_path = "/data/home/ganleilei/law/ContrastiveLJP/"
        tokenizer_path = "/data/home/ganleilei/bert/bert-base-chinese/"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        train_data, valid_data, test_data = load_dataset(data_path)
        train_dataset = CustomDataset(train_data, tokenizer, 512)
        valid_dataset = CustomDataset(valid_data, tokenizer, 512)
        test_dataset = CustomDataset(test_data, tokenizer, 512)

        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_data), len(valid_data), len(test_data)))
        data_dict = {
            "train_data_set": train_data,
            "test_data_set": test_data,
            "valid_data_set": valid_data
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