from copy import deepcopy
import thulac
import jieba
import json
import pickle as pk
import numpy as np
from string import punctuation
import re

add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc = punctuation + add_punc
delete_punc = list(all_punc)
print(delete_punc)
all_punc = []
for word in delete_punc:
    if word != ".":
        all_punc.append(word)
print(all_punc)
doc_len = 15
sent_len = 100


def punc_delete(fact_list):
    fact_filtered = []
    for word in fact_list:
        fact_filtered.append(word)
        if word in all_punc:
            fact_filtered.remove(word)
    return fact_filtered


def hanzi_to_num(hanzi_1):
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


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def get_cutter(dict_path="/data/ganleilei/workspace/ContrastiveLJP/law_processed/Thuocl_seg.txt", mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist('/data/ganleilei/workspace/ContrastiveLJP/law_processed/stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]

weight_pattern = re.compile(pattern="(\d+\.?\d+?)(?:克|千克)")
money_pattern = re.compile(pattern="(\d+\.?\d*)(?:万元|元)")

def seg_sentence(sentence, cut):
    # cut=get_cutter()
    # sentence_seged = thu.cut(sentence.strip(), text=True).split(' ')
    # print("sentence:", sentence)
    sentence_seged = cut(sentence)
    # print("seged sentence:", sentence_seged)
    float_nums = weight_pattern.findall(sentence)
    int_nums = money_pattern.findall(sentence)
    extracted_nums = float_nums + int_nums
    # print("extracted nums:", extracted_nums)

    copy_sentence_seged = deepcopy(sentence_seged)
    for num in extracted_nums:
        start_index = sentence.find(num)
        end_index = start_index + len(num)
        for i, word in enumerate(sentence_seged):
            # print("word:", word)
            tmp_index = sentence.find(word)
            if tmp_index >= start_index and tmp_index+len(word) <= end_index:
                copy_sentence_seged[i] = list(word)

    final_sentence_seged = []
    for item in copy_sentence_seged:
        if type(item) == str:
            final_sentence_seged.append(item)    
        else:
            final_sentence_seged.extend(item)
    # print("seged sentence:", final_sentence_seged)

    outstr = []
    for word in final_sentence_seged:
        if word != '\t':
            word = str(hanzi_to_num(word))
            outstr.append(word)

            # outstr += " "
    return outstr


def lookup_index_for_sentences(sentences, word2id, doc_len, sent_len):
    id2word = {v:k for k, v in word2id.items()}
    item_num = 0
    res = []
    if len(sentences) == 0:
        tmp = [word2id['BLANK']] * sent_len
        res.append(np.array(tmp))
    else:
        for sent in sentences:
            # print("sent:", sent)
            sent = punc_delete(sent)
            # print("punc delete sent:", sent)
            tmp = [word2id['BLANK']] * sent_len
            for i in range(len(sent)):
                if i >= sent_len:
                    break
                try:
                    tmp[i] = word2id[str(sent[i])]
                    item_num += 1
                except KeyError:
                    tmp[i] = word2id['UNK']

            # print([id2word[id] for id in tmp])
            res.append(np.array(tmp))
    if len(res) < doc_len:
        res = np.concatenate([np.array(res), word2id['BLANK'] * np.ones([doc_len - len(res), sent_len], dtype=np.int)], 0)
    else:
        res = np.array(res[:doc_len])

    return res, item_num


def sentence2index_matrix(sentence, word2id, doc_len, sent_len, cut):
    sentence = sentence.replace(' ', '')
    sent_words, sent_n_words = [], []
    for i in sentence.split('。'):
        if i != '':
            sent_words.append((seg_sentence(i, cut)))
    index_matrix, item_num = lookup_index_for_sentences(sent_words, word2id, doc_len, sent_len)
    return index_matrix, item_num, sent_words


with open('/data/ganleilei/law/ContrastiveLJP/w2id_thulac.pkl', 'rb') as f:
    word2id_dict = pk.load(f)
    f.close()

print("word2id dict len:", len(word2id_dict))
print(word2id_dict['.'])
file_list = ['train', 'valid', 'test']
# file_list = ['test']
cut = get_cutter(stop_words_filtered= False)

for i in range(len(file_list)):
    fact_lists = []
    law_label_lists = []
    accu_label_lists = []
    term_lists = []
    money_amount_lists = []
    drug_weight_lists = []
    ##add for pre-trained models
    raw_facts = []

    num = 0

    with open('/data/ganleilei/law/ContrastiveLJP/big/{}_cs_with_fyb_annotate_number.json'.format(file_list[i]), 'r', encoding= 'utf-8') as f:
        idx = 0
        for line in f.readlines():
            idx += 1
            print(f"idx: {idx}")
            line = json.loads(line)
            fact = line['fact_cut']
            line['drug_weight'] = 0
            sentence, word_num, sent_words = sentence2index_matrix(fact, word2id_dict, doc_len, sent_len, cut)

            if word_num <= 10:
                print(fact)
                print(sent_words)
                print(idx)
                continue

            fact_lists.append(sentence)
            law_label_lists.append(line['law'])
            accu_label_lists.append(line['accu'])
            term_lists.append(line['term'])
            money_amount_lists.append(line['money_amount'])
            drug_weight_lists.append(line['drug_weight'])

            ##adding for pre-trained models
            raw_facts.append(fact.replace(" ", ""))

            num += 1
        f.close()
    data_dict = {'fact_list': fact_lists, 'law_label_lists': law_label_lists,
                 'accu_label_lists': accu_label_lists, 'term_lists': term_lists, 'raw_facts_list': raw_facts,
                 'money_amount_lists': money_amount_lists, 'drug_weight_lists': drug_weight_lists}

    pk.dump(data_dict, open('/data/ganleilei/law/ContrastiveLJP/big/{}_processed_thulac_Legal_basis_with_fyb_annotate_number_field.pkl'.format(file_list[i]), 'wb'))
    print(num)
    print('{}_dataset is processed over'.format(file_list[i])+'\n')