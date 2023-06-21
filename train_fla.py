#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/05/29
# @Contact: 11921071@zju.edu.cn

import torch
import datetime, argparse
from utils.config import Data
import time, random
from torch import optim
import pickle
from models.model_THU import LawModel
import sys, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
import jieba
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from scipy import stats
import numpy as np
import logging, os, codecs
import ast, copy


def load_model_decode(model_dir, config, dataset, name, gpu):
    config.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = LawModel(config)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    model.load_state_dict(torch.load(model_dir))
    # model = torch.load(model_dir)
    score = evaluate(model, dataset, name)

    return


def save_data_setting(data: Data, save_file):
    new_data = copy.deepcopy(data)
    ## remove input instances
    ## save data settings
    with open(save_file, 'wb') as fp:
        pickle.dump(new_data, fp)
    print("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    print("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def process_data(data):
    # doc_word_ids, doc_sentence_lens,  claims, claims_num, claims_len, claims_type, fact_labels, claims_labels
    doc_word_ids = []
    fact_labels = []
    claims_labels = []
    doc_lens = []
    claims_type = []
    for item in data:

        for claim in item[2]:
            new_dialogue_word_id = item[0]
            if len(new_dialogue_word_id) > 1500:
                new_dialogue_word_id = new_dialogue_word_id[:1500]
            new_dialogue_word_id += claim
            doc_word_ids.append(new_dialogue_word_id)

            doc_lens.append(len(new_dialogue_word_id))
            fact_labels.append(item[6])

        claims_labels.append(item[7])
        claims_type.append(item[5])

    claims_labels = sum(claims_labels, [])
    claims_type = sum(claims_type, [])

    if len(claims_labels) != len(fact_labels):
        print("label not equal")

    print('process doc word ids len:', len(doc_word_ids))
    print('process claim types len:', len(claims_type))
    print('process claim labels len:', len(claims_labels))
    data = list(zip(doc_word_ids, doc_lens, fact_labels, claims_labels, claims_type))
    return data


def get_claim_type(claim):
    if ("借款" in claim or "本金" in claim) and "本息" not in claim and "利息" not in claim and "违约" not in claim \
            and "担保" not in claim and "诉讼费" not in claim:
        claim_type = '本金'
    elif "本金" in claim and "利息" in claim and "违约" not in claim and "担保" not in claim:
        claim_type = '本息'
    elif "利息" in claim and "本金" not in claim and "违约" not in claim and "担保" not in claim:
        claim_type = '利息'
    elif "违约" in claim:
        claim_type = '违约'
    elif "担保" in claim:
        claim_type = '担保'
    elif "诉讼费" in claim:
        claim_type = '诉讼费'
    else:
        claim_type = '其他'
    return claim_type

def load_data(path, config: Data):
    data = []
    for line in codecs.open(path, mode='r', encoding='utf-8'):
        parts = line.strip().split('\t')
        if len(parts) != 3:
            print(parts)
            continue

        words = [config.word_alphabet.get_index(word) for word in parts[0].split(' ')]
        facts = [int(item) for item in parts[2].split(' ')]
        ground_y = int(parts[1])
        data.append((words, len(words), facts, ground_y))

    return data

def get_result(target, preds, mode):
    # target = np.argmax(np.array(target), axis=1).tolist()
    macro_f1 = f1_score(target, preds, average="macro")
    macro_precision = precision_score(target, preds, average="macro")
    macro_recall = recall_score(target, preds, average="macro")

    micro_f1 = f1_score(target, preds, average="micro")
    print(("%s," % mode) + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1) + str(
        "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))

    return micro_f1 + macro_f1


def get_fact_result(target, preds):
    # target = pd.DataFrame(np.argmax(np.array(target), axis=1), columns=["0"])
    macro_f1 = f1_score(target, preds, average="macro")
    macro_precision = precision_score(target, preds, average="macro")
    macro_recall = recall_score(target, preds, average="macro")

    micro_f1 = f1_score(target, preds, average="micro")
    print("fact results: val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1) + str(
        "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))
    logging.info("fact results, val," + str("micro_f1 %.4f," % micro_f1) + str("macro_f1 %.4f," % macro_f1) + str(
        "-macro_precision %.4f," % macro_precision) + str("-macro_recall %.4f" % macro_recall))


def generate_batch_instance(input_data, gpu):
    # input_data = list(zip(doc_word_ids, doc_lens, fact_labels, claims_labels, claims_type))
    # doc_words_id: batch_size, max_sequence_len
    # doc_sentences_lens: batch_size, sentence_num
    # fact_labels: batch_size, fact_num
    # claims_labels: batch_size, claims_num
    word_seq_tensor = [torch.LongTensor(item[0]) for item in input_data]  # batch_size, max_sequence_length
    word_seq_tensor = pad_sequence(word_seq_tensor, batch_first=True, padding_value=0)

    fact_labels_tensor = torch.FloatTensor([item[2] for item in input_data])  # batch_size, fact_num
    claims_labels_tensor = torch.FloatTensor([item[3] for item in input_data])  # batch_size3

    sentence_lens = [item[1] for item in input_data]

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        claims_labels_tensor = claims_labels_tensor.cuda()
        fact_labels_tensor = fact_labels_tensor.cuda()

    return word_seq_tensor, claims_labels_tensor, fact_labels_tensor, sentence_lens


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

    total_loss, total_fact_loss, total_claim_loss = 0, 0, 0

    claim_predicts = []
    fact_predicts = []
    ground_claim_labels = []
    ground_fact_labels = []

    for batch_idx in range(total_batch):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size

        if end > train_num:
            end = train_num

        batch_instances = dataset[start: end]
        if len(batch_instances) <= 0:
            continue

        input_x_batch, claims_labels_batch, fact_labels_batch, input_sentences_len_batch\
            = generate_batch_instance(batch_instances, config.HP_gpu)

        _, _, batch_fact_predicts, batch_claim_predicts = model.neg_log_likelihood_loss(
            input_x_batch, input_sentences_len_batch, fact_labels_batch, claims_labels_batch)

        claim_predicts.extend(batch_claim_predicts.cpu().tolist())
        fact_predicts.extend(batch_fact_predicts.cpu().tolist())
        ground_claim_labels.extend(claims_labels_batch.cpu().tolist())
        ground_fact_labels.extend(fact_labels_batch.cpu().tolist())

    score = get_result(ground_claim_labels, claim_predicts, name)

    if name == 'Test':
        fact_predicts = [[int(y) for y in x] for x in fact_predicts]
        fact_ground_y = []

        for i in range(train_num):
            fact_ground_y.append(dataset[i][2])

        for i in range(config.fact_num):
            ground = []
            predicts = []
            for j in range(train_num):
                ground.append(fact_ground_y[j][i])
                predicts.append(fact_predicts[j][i])

            print('facts：%d' % i)
            get_fact_result(ground, predicts)

        fact_ground_all = sum(fact_ground_y, [])
        fact_predicts_all = sum(fact_predicts, [])
        print('Total:')
        get_fact_result(fact_ground_all, fact_predicts_all)

    decode_time = time.time() - start_time
    speed = train_num / decode_time
    print(
        "%s finished. Time: %.2fs, speed: %.2fst/s,  total loss: %.4f, total claim loss: %.4f, total fact loss %.4f, score: %.4f" %
        (name, decode_time, speed, total_loss, total_claim_loss, total_fact_loss, score))
    return score


def train(dataset, config: Data):
    train_data_set = dataset["train_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]

    train_num_samples = len(train_data_set)
    # batch_num = (train_num_samples * FLAGS.num_epochs) // FLAGS.batch_size + 1
    batch_num = (train_num_samples * config.HP_iteration) // config.HP_batch_size + 1

    model = LawModel(config)
    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if config.use_sgd:
        optimizer = optim.SGD(parameters, lr=config.HP_lr, momentum=config.HP_momentum)
    elif config.use_adam:
        optimizer = optim.Adam(parameters, lr=config.HP_lr)
    elif config.use_bert:
        optimizer = optim.Adam(parameters, lr=5e-6)  # fine tuning
    else:
        raise ValueError("Unknown optimizer")
    print('optimizer: ', optimizer)

    best_dev = -1

    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        optimizer = lr_decay(optimizer, idx, config.HP_lr_decay, config.HP_lr)

        total_loss = 0
        total_claim_loss = 0
        total_fact_loss = 0

        sample_loss = 0
        sample_claim_loss = 0
        sample_fact_loss = 0

        random.shuffle(train_data_set)

        model.train()
        model.zero_grad()

        batch_size = config.HP_batch_size
        train_num = len(train_data_set)
        total_batch = train_num // batch_size + 1

        ground_y = []
        predicts_y = []

        for batch_idx in range(total_batch):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            if end > train_num:
                end = train_num

            batch_instances = train_data_set[start: end]
            if len(batch_instances) <= 0:
                continue
            input_x_batch, claims_labels_batch, fact_labels_batch, input_sentences_len_batch \
                = generate_batch_instance(batch_instances, config.HP_gpu)

            claim_loss, fact_loss, batch_fact_predicts, batch_claim_predicts = model.neg_log_likelihood_loss(
                input_x_batch, input_sentences_len_batch, fact_labels_batch, claims_labels_batch)

            ground_y.extend(claims_labels_batch.cpu().tolist())
            predicts_y.extend(batch_claim_predicts.cpu().tolist())
            cur_accuracy = accuracy_score(ground_y, predicts_y)

            loss = claim_loss + fact_loss
            total_loss += loss.data
            total_claim_loss += claim_loss.data
            total_fact_loss += fact_loss.data

            sample_loss += loss.data
            sample_claim_loss += claim_loss.data
            sample_fact_loss += fact_loss.data

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; claim loss %.4f; fact loss %.4f; accuracy: %.4f" % (
                    end, temp_cost, sample_loss, sample_claim_loss, sample_fact_loss, cur_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_claim_loss = 0
                sample_fact_loss = 0

            loss.backward()
            optimizer.step()
            model.zero_grad()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print(
            "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %.4f, total claim loss: %.4f, total fact loss %.4f" %
            (idx, epoch_cost, train_num / epoch_cost, total_loss, total_claim_loss, total_fact_loss))
        sys.stdout.flush()

        # evaluate dev data
        current_score = evaluate(model, valid_data_set, "Dev")

        if current_score > best_dev:
            print("Exceed previous best acc score:", best_dev)
            best_dev = current_score
            # save model
            model_name = config.save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            # evaluate test data
            _ = evaluate(model, test_data_set, "Test")


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser(
        description='Augmenting Deep Learning with Expert Prior Knowledge for Reasonable Charge Prediction')
    parser.add_argument('--train', default="data/attribute_data/data/train")
    parser.add_argument('--dev', default="data/attribute_data/data/valid")
    parser.add_argument('--test', default="data/attribute_data/data/test")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="savemodels/thu")
    parser.add_argument('--savedset', default="savemodels/thu")
    parser.add_argument('--loadmodel', default="")

    parser.add_argument('--word_emb_dim', default=100)
    parser.add_argument('--embedding_dense_dim', default=300)
    parser.add_argument('--fact_edim', default=100)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=250)
    parser.add_argument('--fact_num', default=12)
    parser.add_argument('--facts',
                        default="是否夫妻共同债务,是否物权担保,是否存在还款行为,是否约定利率,是否约定借款期限,是否约定保证期间,是否保证人不承担担保责任,是否保证人担保,是否约定违约条款,是否约定还款期限,是否超过诉讼时效,是否借款成立")
    parser.add_argument('--hops', default=3)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--max_decoder_step', default=100)

    parser.add_argument('--HP_iteration', default=100)
    parser.add_argument('--HP_batch_size', default=64)
    parser.add_argument('--HP_hidden_dim', default=256)
    parser.add_argument('--HP_dropout', default=0.2)
    parser.add_argument('--HP_lstmdropout', default=0.5)
    parser.add_argument('--HP_lstm_layer', default=1)
    parser.add_argument('--HP_lr', default=1e-3, type=float)
    parser.add_argument('--HP_lr_decay', default=0.05, type=float)

    args = parser.parse_args()
    status = args.status

    if status == 'train':
        config = Data()
        config.fact_num = 10
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

        config.build_thu_word_alphabet(args.train)
        config.build_thu_word_alphabet(args.dev)
        config.build_thu_word_alphabet(args.test)
        config.fix_alphabet()
        print('word alphabet size:', config.word_alphabet_size)

        config.build_word_pretrain_emb('data/attribute_data/data/words.vec.bak')

        print("\nLoading data...")
        train_data = load_data(args.train, config)
        valid_data = load_data(args.dev, config)
        test_data = load_data(args.test, config)

        print("train_data %d, valid_data %d, test_data %d." % (
            len(train_data), len(valid_data), len(test_data)))

        data_dict = {
            "train_data_set": train_data,
            "test_data_set": test_data,
            "valid_data_set": valid_data
        }
        print("\nSampling data...")
        pass

        config.show_data_summary()
        save_data_setting(config, config.save_model_dir + '.dset')
        print("\nTraining...")
        train(data_dict, config)

    elif status =='test':
        if os.path.exists(args.loadmodel) is False or os.path.exists(args.savedset) is False:
            print('File path does not exit: %s and %s' % (args.loadmodel, args.savedset))
            exit(1)

        config: Data = load_data_setting(args.savedset)
        print("\nLoading data...")
        test_data = load_data(args.test, config)
        decode_results = load_model_decode(args.loadmodel, config, test_data, 'Test', True)
