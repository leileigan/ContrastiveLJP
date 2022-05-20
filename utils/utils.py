import math 
import json
import torch
import numpy as np


class Data_Process():
    def __init__(self):
        self.word2id = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/word2id.json', "r"))
        self.charge2id = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/charge2id.json'))
        self.article2id = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/article2id.json'))
        self.time2id = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/time2id.json'))
        self.symbol = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]
        self.last_symbol = ["?", "。", "？"]
        self.charge2detail = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/charge_details.json','r'))
        self.sent_max_len = 200
        self.law = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/law.json'))
    
    def transform(self, word):
        if not (word in self.word2id.keys()):
            return self.word2id["UNK"]
        else:
            return self.word2id[word]
    
    def parse(self, sent):
        result = []
        sent = sent.strip().split()
        for word in sent:
            if len(word) == 0:
                continue
            if word in self.symbol:
                continue
            result.append(word)
        return result

    def parseH(self, sent):
        result = []
        temp = []     
        sent = sent.strip().split()
        for word in sent:
            if word in self.symbol and word not in self.last_symbol:
                continue
            temp.append(word)
            last = False
            for symbols in self.last_symbol:
                if word == symbols:
                    last = True
            if last:
                #不要标点
                result.append(temp[:-1])
                temp = []
        if len(temp) != 0:
            result.append(temp)
        
        return result

    def seq2Htensor(self, docs, max_sent=16, max_sent_len=128):
        
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)

        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        # for lstm encoder
        #sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        # for textcnn encoder
        sent_tensor = torch.LongTensor(len(docs), max_sent, max_sent_len).zero_()
        
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
                    
        return sent_tensor

    def seq2tensor(self, sents, max_len=350):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                #print(word)
                sent_tensor[s_id][w_id] = self.transform(word) 
        return sent_tensor,sent_len
    
    def seq2hlstm(self, docs, max_sent=16, max_sent_len=64):
        sent_num_max = max([len(s) for s in docs])
        sent_num_max = min(sent_num_max, max_sent)

        sent_len_max = max([len(w) for s in docs for w in s])
        sent_len_max = min(sent_len_max, max_sent_len)
        sent_tensor = torch.LongTensor(len(docs), sent_num_max, sent_len_max).zero_()
        sent_len = torch.LongTensor(len(docs), sent_num_max).zero_()
        doc_len = torch.LongTensor(len(docs)).zero_()
        for d_id, doc in enumerate(docs):
            doc_len[d_id] = len(doc)
            for s_id, sent in enumerate(doc):
                if s_id >= sent_num_max: break
                sent_len[d_id][s_id] = len(sent)
                for w_id, word in enumerate(sent):
                    if w_id >= sent_len_max: break
                    sent_tensor[d_id][s_id][w_id] = self.transform(word)
                    
        return sent_tensor,doc_len,sent_len
    

    def get_graph(self):
        charge_tong = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/charge_tong.json'))
        art_tong = json.load(open('/data/ganleilei/law/ContrastiveLJP/NeurJudge_config_data/art_tong.json'))
        charge_tong2id = {}
        id2charge_tong = {}
        legals = []
        for index,c in enumerate(charge_tong):
            charge_tong2id[c] = str(index)
            id2charge_tong[str(index)] = c
        
        legals = []  
        for i in charge_tong:
            legals.append(self.parse(self.charge2detail[i]['定义']))
           
        legals,legals_len = self.seq2tensor(legals,max_len=100)

        art2id = {}
        id2art = {}
        for index,c in enumerate(art_tong):
            art2id[c] = str(index)
            id2art[str(index)] = c
        arts = []
        for i in art_tong:
            arts.append(self.parse(self.law[str(i)]))
        arts,arts_sent_lent = self.seq2tensor(arts,max_len=150)
        
        return legals,legals_len,arts,arts_sent_lent,charge_tong2id,id2charge_tong,art2id,id2art
        

    def process_data(self,data):
        fact_all = []
        charge_label = []
        article_label = []
        time_label = []
        for index,line in enumerate(data):
            line = json.loads(line)
            fact = line['fact']
            charge = line['charge']
            article = line['article']
            if line['meta']['term_of_imprisonment']['death_penalty'] == True or line['meta']['term_of_imprisonment']['life_imprisonment'] == True:
                time_labels = 0
            else:
                time_labels = self.time2id[str(line['meta']['term_of_imprisonment']['imprisonment'])]
  
            charge_label.append(self.charge2id[charge[0]])
            article_label.append(self.article2id[str(article[0])])

            
            time_label.append(int(time_labels))

            fact_all.append(self.parse(fact))

        article_label = torch.tensor(article_label,dtype=torch.long)
        charge_label = torch.tensor(charge_label,dtype=torch.long)
        time_label = torch.tensor(time_label,dtype=torch.long)

        documents,sent_lent = self.seq2tensor(fact_all,max_len=350)
        return charge_label,article_label,time_label,documents,sent_lent


    def process_law(self,label_names,type = 'charge'):
        if type == 'charge':
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.charge2detail[i]['定义']))
            labels , labels_len = self.seq2tensor(labels,max_len=100)
            return labels , labels_len
        else:
            labels = []  
            for i in label_names:
                labels.append(self.parse(self.law[str(i)]))
            labels , labels_len = self.seq2tensor(labels,max_len=150)
            return labels , labels_len

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.HP_lr
    if args.cos:  # cosine lr schedule
        eta_min = lr * (args.HP_lr_decay ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.HP_iteration)) / 2
    else:  # stepwise lr schedule
        for milestone in args.HP_lr_decay:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

class Data_Process_1():
    def __init__(self):
        self.word2id = json.load(open('/home/libaokui/nlp/LJP/NeurJudge/neurjudge/data/word2id.json', "r"))
        
        self.charge2id = json.load(open('/home/libaokui/nlp/LJP/NeurJudge/neurjudge/data/charge2id.json'))
        self.article2id = json.load(open('/home/libaokui/nlp/LJP/NeurJudge/neurjudge/data/article2id.json'))
        self.time2id = json.load(open('/home/libaokui/nlp/LJP/NeurJudge/neurjudge/data/time2id.json'))
        self.sent2sents = Sent2sents()

    def process_data(self,data):
        fact_all = []
        charge_label = []
        article_label = []
        time_label = []
        for index,line in enumerate(data):
            line = json.loads(line)
            fact = line['fact']
            charge = line['charge']
            article = line['article']
            if line['meta']['term_of_imprisonment']['death_penalty'] == True or line['meta']['term_of_imprisonment']['life_imprisonment'] == True:
                time_labels = 0
            else:
                time_labels = self.time2id[str(line['meta']['term_of_imprisonment']['imprisonment'])]
  
            charge_label.append(self.charge2id[charge[0]])
            article_label.append(self.article2id[str(article[0])])

            
            time_label.append(int(time_labels))

            fact_all.append(self.sent2sents.sentence2index_matrix(fact, self.word2id))

        fact_all = np.array(fact_all)
        article_label = torch.tensor(article_label,dtype=torch.long)
        charge_label = torch.tensor(charge_label,dtype=torch.long)
        time_label = torch.tensor(time_label,dtype=torch.long)
        fact_all = torch.from_numpy(fact_all).long()

        return charge_label.cuda(), article_label.cuda(), time_label.cuda(), fact_all.cuda()





class Sent2sents():
    def __init__(self):
        self.cut = self.get_cutter()
        self.doc_len = 15
        self.sent_len = 100
        self.rm = [",", ".", "?", "\"", "”", "。", "？", "","，",",","、","”"]

    def hanzi_to_num(self, hanzi_1):
        # for num<10000
        hanzi = hanzi_1.strip().replace('零', '')
        if hanzi == '':
            return str(int(0))
        d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
        m = {'十': 1e1, '百': 1e2, '千': 1e3, }
        w = {'万': 1e4, '亿': 1e8}
        res = 0
        tmp = 0
        thou = 0
        for i in hanzi:
            if i not in d.keys() and i not in m.keys() and i not in w.keys():
                return hanzi

        if (hanzi[0]) == '十': hanzi = '一' + hanzi
        for i in range(len(hanzi)):
            if hanzi[i] in d:
                tmp += d[hanzi[i]]
            elif hanzi[i] in m:
                tmp *= m[hanzi[i]]
                res += tmp
                tmp = 0
            else:
                thou += (res + tmp) * w[hanzi[i]]
                tmp = 0
                res = 0
        return int(thou + res + tmp)

    def get_cutter(self, dict_path="/home/libaokui/nlp/LADAN/data_and_config/law_processed/Thuocl_seg.txt"):
        import thulac
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ')]


    def seg_sentence(self, sentence):
        sentence_seged = self.cut(sentence)
        outstr = []
        for word in sentence_seged:
            if word != '\t':
                word = str(self.hanzi_to_num(word))
                outstr.append(word)
        return outstr

    def punc_delete(self, fact_list):
        fact_filtered = []
        for word in fact_list:
            fact_filtered.append(word)
            if word in self.rm:
                fact_filtered.remove(word)
        return fact_filtered

    def lookup_index_for_sentences(self, sentences, word2id):
        res = []
        if len(sentences) == 0:
            tmp = [word2id['BLANK']] * self.sent_len
            res.append(np.array(tmp))
        else:
            for sent in sentences:
                sent = self.punc_delete(sent)
                tmp = [word2id['BLANK']] * self.sent_len
                for i in range(len(sent)):
                    if i >= self.sent_len:
                        break
                    try:
                        tmp[i] = word2id[sent[i]]
                    except KeyError:
                        tmp[i] = word2id['UNK']

                res.append(np.array(tmp))
        if len(res) < self.doc_len:
            res = np.concatenate([np.array(res), word2id['BLANK'] * np.ones([self.doc_len - len(res), self.sent_len], dtype=np.int)], 0)
        else:
            res = np.array(res[:self.doc_len])

        return res

    def sentence2index_matrix(self, sentence, word2id):
        sentence = sentence.replace(' ', '')
        sent_words, sent_n_words = [], []
        for i in sentence.split('。'):
            if i != '':
                sent_words.append((self.seg_sentence(i)))
        index_matrix = self.lookup_index_for_sentences(sent_words, word2id)
        return index_matrix