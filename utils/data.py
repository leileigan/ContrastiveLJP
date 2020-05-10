# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-14 17:34:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-29 15:26:51
import sys
import numpy as np
from utils.alphabet import Alphabet
from utils.functions import *
# import cPickle as pickle
import pickle
import gensim
import jieba, json

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:
    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.number_normalized = False
        self.norm_word_emb = True
        self.word_alphabet = Alphabet(name='word')
        self.biword_alphabet = Alphabet(name='biword')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)

        self.train_instances = []
        self.dev_instances = []
        self.test_instances = []
        self.raw_instances = []

        self.use_bigram = False
        self.word_emb_dim = 300
        self.biword_emb_dim = 300

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.label_size = 0
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.label_alphabet_size = 0

        #  hyperparameters
        self.HP_iteration = 100
        self.HP_batch_size = 10
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

        # optimizer
        self.use_adam = True
        self.use_bert = False
        self.use_sgd = False
        self.use_adadelta = False
        self.mode = 'train'

        self.save_model_dir = ""
        self.save_dset_dir = ""

        self.facts = ""
        self.fact_num = 12
        self.hops = 3
        self.heads = 4
        self.fact_edim = 300


    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word embedding size: %s" % (self.word_emb_dim))
        print("     Biword embedding size: %s" % (self.biword_emb_dim))
        print("     Norm     word   emb: %s" % (self.norm_word_emb))
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
        print("     Law         fact num:  %s" % (self.fact_num))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def build_alphabet_cases(self, path):
        for file in os.listdir(path):
            lines = codecs.open(os.path.join(path, file), 'r', 'utf-8').readlines()
            for idx, line in enumerate(lines):
                if line.strip() == '======chaming=======':
                    chaming = lines[idx + 1].strip()
                    chaming_cut = jieba.cut(chaming)
                    chaming_cut_list = list(chaming_cut)
                    for word in chaming_cut_list:
                        self.word_alphabet.add(word)
                elif line == '======suqiu-split=======':
                    claims = json.loads(lines[idx + 1].strip())
                    for claim in claims:
                        claim_cut = jieba.cut(claim)
                        claim_cut_list = list(claim_cut)
                        for word in claim_cut_list:
                            self.word_alphabet.add(word)

        self.word_alphabet_size = self.word_alphabet.size()


    def build_alphabet(self, input_file):

        with codecs.open(input_file, mode='r', encoding='utf-8') as input_data:
            json_data = json.load(input_data)

        for _, v in json_data.items():
            chaming = v['chaming']
            chaming_cut = jieba.cut(chaming)
            chaming_cut_list = list(chaming_cut)
            for word in chaming_cut_list:
                self.word_alphabet.add(word)

            claims_split = v["claims_split"]
            for claim in claims_split:
                claim_cut = jieba.cut(claim)
                claim_cut_list = list(claim_cut)
                for word in claim_cut_list:
                    self.word_alphabet.add(word)

        self.word_alphabet_size = self.word_alphabet.size()
        self.fix_alphabet()

    def fix_alphabet(self):
        self.word_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet, self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet, self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_word_vec_100(self):
        self.pretrain_word_embedding, self.pretrain_biword_embedding = self.get_embedding()
        self.word_emb_dim, self.biword_emb_dim = 100, 100

    # get pre-trained embeddings
    def get_embedding(self, size=100):
        fname = 'data/wordvec_' + str(size)
        print("build pretrain word embedding from: ", fname)
        word_init_embedding = np.zeros(shape=[self.word_alphabet.size(), size])
        bi_word_init_embedding = np.zeros(shape=[self.biword_alphabet.size(), size])
        pre_trained = gensim.models.KeyedVectors.load(fname, mmap='r')
        # pre_trained_vocab = set([unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
        pre_trained_vocab = set([w for w in pre_trained.vocab.keys()])
        c = 0
        for word, index in self.word_alphabet.iteritems():
            if word in pre_trained_vocab:
                word_init_embedding[index] = pre_trained[word]
            else:
                word_init_embedding[index] = np.random.uniform(-0.5, 0.5, size)
                c += 1

        for word, index in self.biword_alphabet.iteritems():
            bi_word_init_embedding[index] = (word_init_embedding[self.word_alphabet.get_index(word[0])]
                                             + word_init_embedding[self.word_alphabet.get_index(word[1])]) / 2
        # word_init_embedding[word2id[PAD]] = np.zeros(shape=size)
        # bi_word_init_embedding[]
        print('oov character rate %f' % (float(c) / self.word_alphabet.size()))
        return word_init_embedding, bi_word_init_embedding


    def write_decoded_results(self, output_file, predict_results, name):
        fout = open(output_file, 'w')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + "\t" + predict_results[idx][idy][0] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))
