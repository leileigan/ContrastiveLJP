#-*- coding:utf-8 _*-
# @Author: Leilei Gan
# @Time: 2020/05/13
# @Contact: 11921071@zju.edu.cn

import torch
# from pytorch_pretrained_bert import * # 12 * 1 * seq_len * 768
import datetime, argparse
from utils.data import Data
import time, random
from torch import optim
import pickle
from models.model_CNN import LawModel
import sys, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
import jieba
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from scipy import stats
import numpy as np
import logging
from utils.optimization import BertAdam

SEED_NUM = 2020
torch.manual_seed(SEED_NUM)
random.seed(SEED_NUM)
np.random.seed(SEED_NUM)

def process_data(data):
# doc_word_ids, sentences_num, doc_sentence_lens,  claims, claims_num, claims_len, fact_labels, claims_labels
    doc_word_ids = []
    fact_labels = []
    claims_labels = []
    doc_lens = []
    for item in data:

        for claim in item[3]:
            new_dialogue_word_id = item[0]
            if len(new_dialogue_word_id) > 1500:
                new_dialogue_word_id = new_dialogue_word_id[:1500]
            new_dialogue_word_id += claim
            doc_word_ids.append(new_dialogue_word_id)

            doc_lens.append(len(new_dialogue_word_id))
            fact_labels.append(item[6])

        claims_labels.append(item[7])
    claims_labels=sum(claims_labels,[])
    if len(claims_labels)!=len(fact_labels):
        print("label not equal")
    print('process doc word ids len:', len(doc_word_ids))
    print('process claim labels len:', len(claims_labels))
    print('doc word ids:', doc_word_ids[0])
    print('doc lens:', doc_lens[0])
    print('fact labels:', fact_labels[0])
    print('claims labels:', claims_labels[0])
    data = list(zip(doc_word_ids, doc_lens, fact_labels, claims_labels))
    return data


def load_data(path, config: Data):
    with open(path, 'r', encoding="utf-8") as load_f:
        jsonContent = json.load(load_f)
    # jsonContent = data["all_data"]
    print(len(jsonContent))

    #         vocab = Vocabulary()
    #         vocab.load(FLAGS.vocab_model_file, keep_words=FLAGS.vocab_size)

    def _do_vectorize(jsonContent):
        keys = list(jsonContent.keys())

        claims = []
        claims_num = []
        claims_len = []

        doc_words_ids = []
        sentence_nums = []
        doc_sentences_lens = []
        for key in keys:

            #-------------------claim-------------------------------
            claims_split = jsonContent[key]["claims_split"]
            claims_sequences = []
            for claim in claims_split:
                claim_cut = jieba.cut(claim)
                claims_id = [config.word_alphabet.get_index(word) for word in list(claim_cut)]
                claims_sequences.append(claims_id)

            claims_len.append([len(x) for x in claims_sequences])
            claims.append(claims_sequences)
            claims_num.append(len(claims_split))

            #----------------------chaming----------------------------
            chaming = jsonContent[key]["chaming"]
            sentence_nums.append(len(chaming.split('\n')))
            chaming_cut = jieba.cut(chaming)
            chaming_cut_list = list(chaming_cut)
            chaming_ids = [config.word_alphabet.get_index(word) for word in chaming_cut_list]
            doc_sentences_lens.append(len(chaming_cut_list))
            doc_words_ids.append(chaming_ids)

        print("doc word len:", len(doc_words_ids))
        print('doc sentences lengths statics:')
        print(stats.describe(np.array(doc_sentences_lens)))
        total_claim_lens = sum(claims_len, [])
        print('claim statics:')
        print(stats.describe(np.array(total_claim_lens)))
        return doc_words_ids, sentence_nums, doc_sentences_lens, claims, claims_num, claims_len

    def _do_label_vectorize(jsonContent):
        keys = list(jsonContent.keys())

        fact_labels = []
        claims_labels = []

        for key in keys:
            fact_label_sample = jsonContent[key]["fact_labels"]
            fact_label_decode = []
            claim_label_decode = []
            for fact in config.facts:
                label = 0
                if fact_label_sample[fact] != 0:
                    label = 1
                fact_label_decode.append(label)
            for label in jsonContent[key]["claims_labels"]:
                if label == "驳回":
                    claim_label_decode.append(0)
                elif label == "部分支持":
                    claim_label_decode.append(1)
                elif label == "支持":
                    claim_label_decode.append(2)
                else:
                    print(label)
                    print("claim label error")

            fact_labels.append(fact_label_decode)
            claims_labels.append(claim_label_decode)

        return fact_labels, claims_labels # (batch_size, 10), (batch_size, claim_num)

    # encode x
    doc_word_ids, sentences_num, doc_sentence_lens, claims, claims_num, claims_len = _do_vectorize(jsonContent)
    fact_labels, claims_labels = _do_label_vectorize(jsonContent)
    data = list(zip(doc_word_ids, sentences_num, doc_sentence_lens,  claims, claims_num, claims_len, fact_labels, claims_labels))
    return data


def get_result(target, preds, mode):
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

    ''' 
    print("ground: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_target[0][1], sorted_target[1][1],
                                                                   sorted_target[2][1]))
    print("predicts: (0, {:d}), (1, {:d}), (2, {:d}) ".format(sorted_preds[0][1], sorted_preds[1][1],
                                                                  sorted_preds[2][1]))
    '''

    target_names = ['驳回诉请', '部分支持', "支持诉请"]
    # if mode == "test":
    if mode:
        print(classification_report(target, preds, target_names=target_names, digits=4))
    logging.info(classification_report(target, preds, target_names=target_names, digits=4))
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
    # input_data = list(zip(doc_word_ids, doc_lens, fact_labels, claims_labels))
    # doc_words_id: batch_size, max_sequence_len
    # doc_sentences_lens: batch_size, sentence_num
    # fact_labels: batch_size, fact_num
    # claims_labels: batch_size, claims_num
    word_seq_tensor = [torch.LongTensor(item[0]) for item in input_data] # batch_size, max_sequence_length
    word_seq_tensor = pad_sequence(word_seq_tensor, batch_first=True, padding_value=0)

    fact_labels_tensor = torch.FloatTensor([item[2] for item in input_data]) # batch_size, fact_num
    claims_labels_tensor = torch.FloatTensor([item[3] for item in input_data]) # batch_size

    sentence_lens = [item[1] for item in input_data]

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        claims_labels_tensor = claims_labels_tensor.cuda()
        fact_labels_tensor = fact_labels_tensor.cuda()

    '''
    print('word seq tensor size:', word_seq_tensor.size())
    print('claims labels tensor size:', claims_labels_tensor.size())
    print('fact labels tensor size:', fact_labels_tensor.size())
    '''

    return word_seq_tensor, claims_labels_tensor, fact_labels_tensor, sentence_lens


def data_initialization(data, train_file, dev_file, test_file):
    print('begin building word alphabet set')
    data.build_word_alphabet(train_file)
    data.build_word_alphabet(dev_file)
    data.build_word_alphabet(test_file)
    data.fix_alphabet()
    print('word alphabet size:', data.word_alphabet_size)
    return data

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

def evaluate(model, dataset, name):

    model.eval()
    batch_size = 16
    train_num = len(dataset)
    total_batch = train_num // batch_size + 1

    start_time = time.time()

    total_loss, total_fact_loss, total_claim_loss = 0, 0, 0

    claim_predicts = []
    ground_claim_labels = []

    for batch_idx in range(total_batch):
        start = batch_idx * batch_size
        end = (batch_idx + 1) * batch_size

        if end > train_num:
            end = train_num

        batch_instances = dataset[start: end]
        if len(batch_instances) <= 0:
            continue

        input_x_batch, claims_labels_batch, fact_labels_batch, input_sentences_len_batch \
            = generate_batch_instance(batch_instances, config.HP_gpu)
        claim_loss, batch_claim_predicts = model.neg_log_likelihood_loss(input_x_batch, input_sentences_len_batch,
                                                                         fact_labels_batch, claims_labels_batch)
        total_claim_loss += claim_loss.data

        claim_predicts.extend(batch_claim_predicts.cpu().tolist())
        ground_claim_labels.extend(claims_labels_batch.cpu().tolist())

    score = get_result(ground_claim_labels, claim_predicts, name)
    accuracy = accuracy_score(ground_claim_labels, claim_predicts)

    decode_time = time.time() - start_time
    speed = train_num / decode_time
    print(
        "%s finished. Time: %.2fs, speed: %.2fst/s, total claim loss: %.4f, accuracy: %.4f, score: %.4f" %
        (name, decode_time, speed, total_claim_loss, accuracy, score))
    return score

def train(dataset, config: Data):
    train_data_set = dataset["train_data_set"]
    # train_data_set = dataset["valid_data_set"]
    valid_data_set = dataset["valid_data_set"]
    test_data_set = dataset["test_data_set"]

    train_num_samples = len(train_data_set)
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
        optimizer = optim.Adam(parameters, lr=config.HP_lr)  # fine tuning
    else:
        raise ValueError("Unknown optimizer")

    # optimizer = BertAdam(parameters, config.HP_lr, warmup=0.05 , t_total=batch_num)
    print('optimizer: ', optimizer)

    best_dev = -1

    for idx in range(config.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, config.HP_iteration))
        optimizer = lr_decay(optimizer, idx, config.HP_lr_decay, config.HP_lr)

        sample_loss = 0
        sample_claim_loss = 0
        sample_fact_loss = 0

        total_loss = 0
        total_claim_loss = 0
        total_fact_loss = 0

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

            ground_y.extend(claims_labels_batch.cpu().tolist())

            claim_loss,  batch_claim_predicts = model.neg_log_likelihood_loss(
                    input_x_batch, input_sentences_len_batch, fact_labels_batch, claims_labels_batch)

            predicts_y.extend(batch_claim_predicts.cpu().tolist())

            loss = claim_loss
            total_loss += loss.data
            total_claim_loss += claim_loss.data

            sample_loss += loss.data
            sample_claim_loss += claim_loss.data

            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                cur_accuracy = accuracy_score(ground_y, predicts_y)
                print("     Instance: %s; Time: %.2fs; loss: %.4f; claim loss %.4f; accuracy %.4f" % (
                    end, temp_cost, sample_loss, sample_claim_loss, cur_accuracy))
                sys.stdout.flush()
                sample_loss = 0
                sample_claim_loss = 0

            loss.backward()
            optimizer.step()
            model.zero_grad()


        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        epoch_accuracy = accuracy_score(ground_y, predicts_y)
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %.4f, total claim loss: %.4f, total accuracy: %.4f" %
              (idx, epoch_cost, train_num / epoch_cost, total_loss, total_claim_loss, epoch_accuracy))
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
    parser = argparse.ArgumentParser(description='Augmenting Deep Learning with Expert Prior Knowledge for Reasonable Charge Prediction')
    parser.add_argument('--train', default="./data/chaming-train.json")
    parser.add_argument('--dev', default="./data/chaming-dev.json")
    parser.add_argument('--test', default="./data/chaming-test.json")
    parser.add_argument('--status', default="train")
    parser.add_argument('--savemodel', default="")
    parser.add_argument('--savedset', default="")

    parser.add_argument('--word_emb_dim', default=200)
    parser.add_argument('--embedding_dense_dim', default=300)
    parser.add_argument('--fact_edim', default=300)
    parser.add_argument('--MAX_SENTENCE_LENGTH', default=250)
    parser.add_argument('--fact_num', default=12)
    parser.add_argument('--facts',
                        default="是否夫妻共同债务,是否物权担保,是否存在还款行为,是否约定利率,是否约定借款期限,是否约定保证期间,是否保证人不承担担保责任,是否保证人担保,是否约定违约条款,是否约定还款期限,是否超过诉讼时效,是否借款成立")
    parser.add_argument('--hops', default=3)
    parser.add_argument('--heads', default=4)
    parser.add_argument('--max_decoder_step', default=100)

    parser.add_argument('--HP_iteration', default=50)
    parser.add_argument('--HP_batch_size', default=16)
    parser.add_argument('--HP_hidden_dim', default=256)
    parser.add_argument('--HP_dropout', default=0.2)
    parser.add_argument('--HP_lstmdropout', default=0.2)
    parser.add_argument('--HP_lstm_layer', default=1)
    parser.add_argument('--HP_lr', default=1e-3)

    parser.add_argument('--filters_size', default='1,2,3,4')
    parser.add_argument('--num_filters', default='64, 64, 64, 64')

    args = parser.parse_args()

    config = Data()
    config.facts = list(map(str, args.facts.split(",")))
    config.HP_iteration = args.HP_iteration
    config.HP_batch_size = args.HP_batch_size
    config.HP_hidden_dim = args.HP_hidden_dim
    config.HP_dropout = args.HP_dropout
    config.HP_lstmdropout = args.HP_lstmdropout
    config.HP_lstm_layer = args.HP_lstm_layer
    config.HP_lr = args.HP_lr
    config.MAX_SENTENCE_LENGTH = args.MAX_SENTENCE_LENGTH
    config.filters_size = [int(item) for item in args.filters_size.strip().split(',')]
    config.num_filters = [int(item) for item in args.num_filters.strip().split(',')]

    data_initialization(config, args.train, args.dev, args.test)

    config.build_word_pretrain_emb('data/word2vec.dim200.txt')

    print("\nLoading data...")
    train_data = load_data(args.train, config)
    valid_data = load_data(args.dev, config)
    test_data = load_data(args.test, config)

    print("train_data %d, valid_data %d, test_data %d." % (
        len(train_data), len(valid_data), len(test_data)))

    data_dict = {
        "train_data_set": process_data(train_data),
        "test_data_set": process_data(test_data),
        "valid_data_set": process_data(valid_data)
    }
    print("\nSampling data...")
    pass

    config.show_data_summary()
    print("\nTraining...")
    train(data_dict, config)
