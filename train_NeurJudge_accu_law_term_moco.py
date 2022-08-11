#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn

import argparse
import copy
import datetime
import os
import pickle
import sys
import time
import random

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, confusion_matrix)
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from models.model_NeurJudge_accu_law_term_moco import MoCo
from utils.optim import ScheduledOptim
from transformers import AutoTokenizer

from utils.utils import Data_Process
import torch.optim as optim
from tqdm import tqdm
import numpy as np

os.chdir('/data/ganleilei/workspace/ContrastiveLJP')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_dir, config, gpu):
    config.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = MoCo(config)
    print(model)
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

def seed_rand(SEED_NUM):
    torch.manual_seed(SEED_NUM)
    random.seed(SEED_NUM)
    np.random.seed(SEED_NUM)
    torch.cuda.manual_seed(SEED_NUM)
    torch.cuda.manual_seed_all(SEED_NUM)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.word_emb_dim = 200
        self.pretrain_word_embedding = None
        self.word2id_dict = None
        self.id2word_dict = None
        self.bert_path = None

        self.accu_label_size = 119
        self.law_label_size = 103
        self.term_label_size = 12
        self.law_relation_threshold = 0.3

        self.sent_len = 100
        self.doc_len = 15
        #  hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 128
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.2
        self.HP_lstmdropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True
        self.HP_gpu = True
        self.HP_lr = 0.015
        self.HP_lr_decay = 0.05
        self.HP_clip = 5.0
        self.HP_momentum = 0
        self.bert_hidden_size = 768
        self.HP_freeze_word_emb = True

        # optimizer
        self.use_adam = True
        self.use_bert = False
        self.use_sgd = False
        self.use_adadelta = False
        self.use_warmup_adam = False
        self.mode = 'train'

        self.save_model_dir = ""
        self.save_dset_dir = ""

        self.filters_size = [1, 3, 4, 5]
        self.num_filters = [50, 50, 50, 50]

        #contrastive learning
        self.moco_temperature = 0.07
        self.moco_queue_size = 65536
        self.moco_momentum = 0.999
        self.alpha = 0.1
        self.warm_epoch = 0
        self.confused_matrix = None
        self.moco_hard_queue_size = 3000

        self.seed = 10


    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Bert Path:           %s" % (self.bert_path))
        print("     Accu label     size: %s" % (self.accu_label_size))
        print("     Law label     size:  %s" % (self.law_label_size))
        print("     Term label     size: %s" % (self.term_label_size))

        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Filter size:        :  %s" % (self.filters_size))
        print("     Number filters      :  %s" % (self.num_filters))

        print("     Temperature         :  %s" % (self.moco_temperature))
        print("     Momentum            :  %s" % (self.moco_momentum))
        print("     Queue size          :  %s" % (self.moco_queue_size))
        print("     Alpha               :  %s" % (self.alpha))
        print("     Hard queue size     :  %s" % (self.moco_hard_queue_size))

        print("DATA SUMMARY END.")
        sys.stdout.flush()


    def load_word_pretrain_emb(self, emb_path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(emb_path))
        self.word_emb_dim = self.pretrain_word_embedding.shape[1]
        print("word embedding size:", self.pretrain_word_embedding.shape)

class NeurJudgeDataset(Dataset):

    def __init__(self, data, tokenizer, max_len, id2word_dict):
        self.tokenizer = tokenizer
        self.max_len = max_len
        filtered_data = {'fact_list':[], 'accu_label_lists':[], 'law_label_lists':[], 'term_lists': [], 'raw_fact_lists': []}
        self.number_intensive_classes = list(range(120))
        # self.number_intensive_classes = [42]
        for index in range(len(data['fact_list'])):
            if data['accu_label_lists'][index] not in self.number_intensive_classes: continue
            filtered_data['fact_list'].append(data['fact_list'][index])
            filtered_data['accu_label_lists'].append(data['accu_label_lists'][index])
            filtered_data['law_label_lists'].append(data['law_label_lists'][index])
            filtered_data['term_lists'].append(data['term_lists'][index])
            filtered_data['raw_fact_lists'].append(data['raw_facts_list'][index])
            
        self.data = filtered_data
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
        # if accu_label_lists in self.number_intensive_classes:
        #     print(raw_fact_list)
        #     print(law_label_lists)
        #     print(term_lists)
        #     print(self.data['raw_fact_lists'][index])
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

def load_dataset(path):
    train_path = os.path.join(path, "train_processed_thulac_Legal_basis_with_fyb_annotate_number_field.pkl")
    valid_path = os.path.join(path, "valid_processed_thulac_Legal_basis_with_fyb_annotate_number_field.pkl")
    test_path = os.path.join(path, "test_processed_thulac_Legal_basis_with_fyb_annotate_number_field.pkl")
    
    train_dataset = pickle.load(open(train_path, mode='rb'))
    valid_dataset = pickle.load(open(valid_path, mode='rb'))
    test_dataset = pickle.load(open(test_path, mode='rb'))

    print("train dataset sample len:", len(train_dataset['law_label_lists']))
    return train_dataset, valid_dataset, test_dataset


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

def get_result(accu_target, accu_preds, law_target, law_preds, term_target, term_preds, mode):
    accu_macro_f1 = f1_score(accu_target, accu_preds, average="macro")
    accu_macro_precision = precision_score(accu_target, accu_preds, average="macro")
    accu_macro_recall = recall_score(accu_target, accu_preds, average="macro")

    law_macro_f1 = f1_score(law_target, law_preds, average="macro")
    law_macro_precision = precision_score(law_target, law_preds, average="macro")
    law_macro_recall = recall_score(law_target, law_preds, average="macro")

    term_macro_f1 = f1_score(term_target, term_preds, average="macro")
    term_macro_precision = precision_score(term_target, term_preds, average="macro")
    term_macro_recall = recall_score(term_target, term_preds, average="macro")

    print("Accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (accu_macro_f1, accu_macro_precision, accu_macro_recall))
    print("Law task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (law_macro_f1, law_macro_precision, law_macro_recall))
    print("Term task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (term_macro_f1, term_macro_precision, term_macro_recall))

    return accu_macro_f1 + law_macro_f1 + term_macro_f1


def evaluate(model, valid_dataloader, process, name, epoch_idx):

    model.eval()
    ground_accu_y, ground_law_y, ground_term_y  = [], [], []
    predicts_accu_y, predicts_law_y, predicts_term_y = [], [], []
    
    legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()
    legals,legals_len,arts,arts_sent_lent = legals.cuda(),legals_len.cuda(),arts.cuda(),arts_sent_lent.cuda()
    for batch_idx, datapoint in enumerate(valid_dataloader):
        documents, _, accu_label_lists, law_label_lists, term_lists = datapoint
        sent_lent = ""
        accu_preds, law_preds, term_preds = model.predict(
            legals,legals_len,arts,arts_sent_lent, \
            charge_tong2id,id2charge_tong,art2id,id2art,documents, \
            sent_lent,process, DEVICE, accu_label_lists, law_label_lists, term_lists)

        ground_accu_y.extend(accu_label_lists.tolist())
        ground_law_y.extend(law_label_lists.tolist())
        ground_term_y.extend(term_lists.tolist())

        predicts_accu_y.extend(accu_preds.tolist())
        predicts_law_y.extend(law_preds.tolist())
        predicts_term_y.extend(term_preds.tolist())
        # if batch_idx == 10: break
    accu_accuracy = accuracy_score(ground_accu_y, predicts_accu_y)
    law_accuracy = accuracy_score(ground_law_y, predicts_law_y)
    term_accuracy = accuracy_score(ground_term_y, predicts_term_y)
    confused_matrix_accu = confusion_matrix(ground_accu_y, predicts_accu_y)
    # confused_matrix_law = confusion_matrix(ground_law_y, predicts_law_y)
    # confused_matrix_term = confusion_matrix(ground_term_y, predicts_term_y)
    print("Accu task accuracy: %.4f, Law task accuracy: %.4f, Term task accuracy: %.4f" % (accu_accuracy, law_accuracy, term_accuracy)) 
    # print("Confused matrix accu of 寻衅滋事罪:", confused_matrix_accu[1])
    # print("Confused matrix accu of 故意伤害罪:", confused_matrix_accu[111])
    score = get_result(ground_accu_y, predicts_accu_y, ground_law_y, predicts_law_y, ground_term_y, predicts_term_y, name)

    abs_score_lists, accu_s_lists = [], []
    # for i in range(119):
    target_classes = list(range(120))
    num_target_classes = [83, 11, 55, 16, 37, 102, 52, 107, 61, 12, 58, 75, 78, 38, 69, 60, 54, 94, 110, 88, 19, 30, 59, 26, 51, 118, 86, 49, 7] # number sensitive classes
    # num_target_classes = [54, 86]
    g_t_lists, p_t_lists = [], []
    num_g_t_lists, num_p_t_lists = [], []
    for g_y, p_y, g_t, p_t in zip(ground_accu_y, predicts_accu_y, ground_term_y, predicts_term_y):
        if g_y in target_classes:
            g_t_lists.append(g_t)
            p_t_lists.append(p_t)
        
        if g_y in num_target_classes:
            num_g_t_lists.append(g_t)
            num_p_t_lists.append(p_t)

    # print(list(zip(g_t_lists, p_t_lists)))
    term_macro_f1 = f1_score(g_t_lists, p_t_lists, average="macro")
    term_macro_precision = precision_score(g_t_lists, p_t_lists, average="macro")
    term_macro_recall = recall_score(g_t_lists, p_t_lists, average="macro")

    g_t_lists = np.array(g_t_lists)
    p_t_lists = np.array(p_t_lists)
    abs_error = sum(abs(g_t_lists - p_t_lists)) / len(g_t_lists)

    print(f"term macro f1: {term_macro_f1}, term_macro_precision: {term_macro_precision}, term_macro_recall: {term_macro_recall}, abs error: {abs_error}")
    
    # evaluate on number sensitive classes
    term_macro_f1 = f1_score(num_g_t_lists, num_p_t_lists, average="macro")
    term_macro_precision = precision_score(num_g_t_lists, num_p_t_lists, average="macro")
    term_macro_recall = recall_score(num_g_t_lists, num_p_t_lists, average="macro")

    num_g_t_lists = np.array(num_g_t_lists)
    num_p_t_lists = np.array(num_p_t_lists)
    num_abs_error = sum(abs(num_g_t_lists - num_p_t_lists)) / len(num_g_t_lists)

    print(f"number sensitive class term macro f1: {term_macro_f1}, term_macro_precision: {term_macro_precision}, term_macro_recall: {term_macro_recall}, abs error: {num_abs_error}")

    return score, abs_error 


def train(model, dataset, config: Config):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
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

    process = Data_Process()
    legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()
    legals,legals_len,arts,arts_sent_lent = legals.cuda(),legals_len.cuda(),arts.cuda(),arts_sent_lent.cuda()
    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        optimizer = lr_decay(optimizer, idx, config.HP_lr_decay, config.HP_lr)
        sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, sample_contra_loss = 0, 0, 0, 0, 0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size

        ground_accu_y, predicts_accu_y = [], []
        ground_law_y, predicts_law_y = [], []
        ground_term_y, predicts_term_y = [], []

        for batch_idx, datapoint in enumerate(train_dataloader):
            documents, _, accu_label_lists, law_label_lists, term_lists = datapoint
            contra_loss, accu_loss, law_loss, term_loss, accu_preds, law_preds, term_preds = \
                model.forward(legals,legals_len,arts,arts_sent_lent, \
                charge_tong2id,id2charge_tong,art2id,id2art,documents, \
                config.MAX_SENTENCE_LENGTH,process, DEVICE, accu_label_lists, law_label_lists, term_lists)

            loss = accu_loss + term_loss + law_loss + config.alpha * contra_loss
            sample_loss += loss.data
            sample_accu_loss += accu_loss.data
            sample_law_loss += law_loss.data
            sample_term_loss += term_loss.data
            sample_contra_loss += contra_loss.data

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
                print("Instance: %s; Time: %.2fs; loss: %.2f; contra loss %.2f;accu loss %.2f; law loss %.2f; term loss %.2f; accu acc %.4f; law acc %.4f; term acc %.4f" % 
                ((batch_idx + 1), temp_cost, sample_loss, sample_contra_loss, sample_accu_loss, sample_law_loss, sample_term_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0
                sample_contra_loss = 0
            # if batch_idx == 10: break
            loss.backward()
            # optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.HP_clip)
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # evaluate dev data
        current_score, abs_score = evaluate(model, valid_dataloader, process,  "Dev", -1)
        print(f"dev current score: {current_score}, abs score: {abs_score}, current score and abs score: {current_score - abs_score}")
        
        model_name = os.path.join(config.save_model_dir, f"{idx}.ckpt")
        torch.save(model.state_dict(), model_name)

        _ = evaluate(model, test_dataloader, process, "Test", -1)


if __name__ == '__main__':
    print(datetime.datetime.now())
    BASE = "/data/ganleilei/law/ContrastiveLJP"
    parser = argparse.ArgumentParser(description='Contrastive Legal Judgement Prediction')
    parser.add_argument('--data_path', default="/data/ganleilei/law/ContrastiveLJP/datasets/fyb_annotate/NeurJudge/")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default=BASE+"/results/NeurJudge/NeurJudge_accu_law_term_moco")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--embedding_path', default=BASE+'/cail_thulac.npy')
    parser.add_argument('--word2id_dict', default=BASE+'/w2id_thulac.pkl')

    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=510)

    parser.add_argument('--HP_iteration', default=16, type=int)
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

    parser.add_argument('--law_relation_threshold', default=0.3)

    args = parser.parse_args()

    status = args.status
    print(args)
    seed_rand(args.seed)    
    status = args.status

    if status == 'train':
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
        config.save_model_dir = os.path.join(args.savemodel, f"{args.seed}")
        config.HP_freeze_word_emb = args.HP_freeze_word_emb
        if not os.path.exists(config.save_model_dir):
            os.mkdir(config.save_model_dir)
        config.use_warmup_adam = str2bool(args.use_warmup_adam)
        config.use_adam = str2bool(args.use_adam)
        config.moco_temperature = args.moco_temperature
        config.moco_queue_size = args.moco_queue_size
        config.moco_momentum = args.moco_momentum
        config.warm_epoch = args.warm_epoch
        config.alpha = args.alpha
        config.word2id_dict = pickle.load(open(args.word2id_dict, 'rb'))
        config.id2word_dict = {item[1]: item[0] for item in config.word2id_dict.items()}
        config.bert_path = args.bert_path
        config.seed = args.seed

        config.load_word_pretrain_emb(args.embedding_path)
        save_data_setting(config, os.path.join(config.save_model_dir,  'data.dset'))
        config.show_data_summary()

        print("\nLoading data...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        train_dataset = NeurJudgeDataset(train_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        valid_dataset = NeurJudgeDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        test_dataset = NeurJudgeDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)

        train_dataloader = DataLoader(train_dataset, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_neur_judge_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)

        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        data_dict = {
            "train_data_set": train_dataloader,
            "test_data_set": test_dataloader,
            "valid_data_set": valid_dataloader
        }

        seed_rand(args.seed)
        model = MoCo(config)
        if config.HP_gpu:
            model.cuda()

        print("\nTraining...")
        train(model, data_dict, config)

    elif status == 'test':
        if os.path.exists(args.loadmodel) is False or os.path.exists(args.savedset) is False:
            print('File path does not exit: %s and %s' % (args.loadmodel, args.savedset))
            exit(1)

        config = load_data_setting(args.savedset)