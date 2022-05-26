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

import torch
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, confusion_matrix)
from torch import optim
from torch.utils.data.dataloader import DataLoader

from models.model_NeurJudge import NeurJudge
from utils.config import Config
from utils.optim import ScheduledOptim
from data.dataset import load_dataset, NeurJudgeDataset, collate_neur_judge_fn 
from transformers import AutoTokenizer
from law_processed.law_processed import get_law_graph
from utils.config import seed_rand
from utils.utils import Data_Process

import torch.optim as optim
from tqdm import tqdm

os.chdir('/data/ganleilei/workspace/ContrastiveLJP')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_dir, config, gpu):
    config.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = NeurJudge(config)
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

    # print(classification_report(accu_target, accu_preds))
    # print(classification_report(law_target, law_preds))
    # print(classification_report(term_target, term_preds))

    print("Accu task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (accu_macro_f1, accu_macro_precision, accu_macro_recall))
    print("Law task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (law_macro_f1, law_macro_precision, law_macro_recall))
    print("Term task: macro_f1:%.4f, macro_precision:%.4f, macro_recall:%.4f" % (term_macro_f1, term_macro_precision, term_macro_recall))

    return (accu_macro_f1 + law_macro_f1 + term_macro_f1) / 3
    # return accu_macro_f1


def evaluate(model, valid_dataloader, process, name, epoch_idx):

    model.eval()
    ground_accu_y, ground_law_y, ground_term_y  = [], [], []
    predicts_accu_y, predicts_law_y, predicts_term_y = [], [], []
    
    process = Data_Process()
    legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art = process.get_graph()
    legals,legals_len,arts,arts_sent_lent = legals.cuda(),legals_len.cuda(),arts.cuda(),arts_sent_lent.cuda()
    for batch_idx, datapoint in enumerate(valid_dataloader):
        documents, _, accu_label_lists, law_label_lists, term_lists = datapoint
        sent_lent = ""
        accu_preds, law_preds, term_preds, accu_loss, law_loss, term_loss, _, _, _ = model(
            legals,legals_len,arts,arts_sent_lent, \
            charge_tong2id,id2charge_tong,art2id,id2art,documents, \
            sent_lent,process,accu_label_lists, law_label_lists, term_lists)

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
    # print("Confused matrix accu of 寻衅滋事罪:", confused_matrix_accu[1])
    # print("Confused matrix accu of 故意伤害罪:", confused_matrix_accu[111])
    score = get_result(ground_accu_y, predicts_accu_y, ground_law_y, predicts_law_y, ground_term_y, predicts_term_y, name)

    return score


def train(model, dataset, config: Config):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]
    print("config batch size:", config.HP_batch_size)
    train_dataloader = DataLoader(train_data_set, batch_size=config.HP_batch_size, shuffle=True, collate_fn=collate_neur_judge_fn)
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)
    test_dataloader = DataLoader(test_data_set, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)

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
        sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss = 0, 0, 0, 0

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size

        ground_accu_y, predicts_accu_y = [], []
        ground_law_y, predicts_law_y = [], []
        ground_term_y, predicts_term_y = [], []

        for batch_idx, datapoint in enumerate(tqdm(train_dataloader)):
            fact_lists, _, accu_label_lists, law_label_lists, term_lists = datapoint
            accu_preds, law_preds, term_preds, accu_loss, law_loss, term_loss, _, _, _ = model.forward( \
                legals, legals_len, arts, arts_sent_lent, charge_tong2id, id2charge_tong, art2id, id2art, fact_lists, \
                config.MAX_SENTENCE_LENGTH, process, accu_label_lists, law_label_lists, term_lists)

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
                print("Instance: %s; Time: %.2fs; loss: %.2f; accu loss %.2f; law loss %.2f; term loss %.2f; accu acc %.4f; law acc %.4f; term acc %.4f" % 
                ((batch_idx + 1), temp_cost, sample_loss, sample_accu_loss, sample_law_loss, sample_term_loss, cur_accu_accuracy, cur_law_accuracy, cur_term_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_accu_loss = 0
                sample_law_loss = 0
                sample_term_loss = 0
            # if batch_idx == 10: break
            loss.backward()
            # optimizer.step_and_update_lr()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.HP_clip)
            optimizer.step()
            model.zero_grad()

        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_dataloader, process,  "Dev", idx)

        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = os.path.join(config.save_model_dir, "best.ckpt")
            torch.save(model.state_dict(), model_name)
            # evaluate test data
        else:
            no_imporv_epoch += 1
            if no_imporv_epoch >= 20:
                print("early stop")
                break
       
        _ = evaluate(model, test_dataloader, process, "Test", idx)


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(description='Contrastive Legal Judgement Prediction')
    parser.add_argument('--data_path', default="data/NeurJudge/")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="/data/ganleilei/law/ContrastiveLJP/results/NeurJudge/")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--embedding_path', default='/data/ganleilei/law/ContrastiveLJP/cail_thulac.npy')
    parser.add_argument('--word2id_dict', default='/data/ganleilei/law/ContrastiveLJP/w2id_thulac.pkl')

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
    parser.add_argument('--use_plus', action='store_true')

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
        config.save_model_dir = args.savemodel
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

        config.load_word_pretrain_emb(args.embedding_path)
        save_data_setting(config, os.path.join(config.save_model_dir,  'data.dset'))
        config.show_data_summary()

        print("\nLoading data...")
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        
        train_dataset = NeurJudgeDataset(train_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        valid_dataset = NeurJudgeDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        test_dataset = NeurJudgeDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_dataset), len(valid_dataset), len(test_dataset)))

        data_dict = {
            "train_data_set": train_dataset,
            "test_data_set": test_dataset,
            "valid_data_set": valid_dataset
        }

        model = NeurJudge(config)
        if config.HP_gpu:
            model.cuda()

        print("\nTraining...")
        train(model, data_dict, config)

    elif status == 'test':
        if os.path.exists(args.loadmodel) is False:
            print('File path does not exit: %s and %s' % args.loadmodel)
            exit(1)

        print("\nLoading data...")
        savedset_path = "/data/ganleilei/law/ContrastiveLJP/results/NeurJudge/data.dset"
        config = load_data_setting(savedset_path)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        train_data, valid_data, test_data = load_dataset(args.data_path)
        train_dataset = NeurJudgeDataset(train_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        valid_dataset = NeurJudgeDataset(valid_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)
        test_dataset = NeurJudgeDataset(test_data, tokenizer, config.MAX_SENTENCE_LENGTH, config.id2word_dict)

        train_dataloader = DataLoader(train_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=config.HP_batch_size, shuffle=False, collate_fn=collate_neur_judge_fn)

        model = load_model(args.loadmodel, config, True)
        score = evaluate(model, test_dataloader, "Test", config, 0)