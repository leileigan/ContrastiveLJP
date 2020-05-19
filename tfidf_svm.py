#-*- coding:utf-8 _*-  
# @Author: Leilei Gan
# @Time: 2020/05/12
# @Contact: 11921071@zju.edu.cn


# %%

import tensorflow as tf

tf.app.flags.DEFINE_string('f', '', 'kernel')
import jieba
import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from collections import Counter
from sklearn.metrics import classification_report

# %%

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("all_data_file", "./data/claims_judgment_all.json",
                       "Data source for training.")
tf.flags.DEFINE_string("word_embedding_file", "../model/embedding_matrix.npy",
                       "Pre train embedding file.")
tf.flags.DEFINE_string("tokenizer_model_file", "data/tokenizer.pickle",
                       "tokenizer model file.")

# Data sample bound
# tf.flags.DEFINE_integer("lower_bound", 3000, "lower bound frequency for over-sampling (default: 3,000)")
# tf.flags.DEFINE_integer("upper_bound", 1000000, "upper bound frequency for sub-sampling (default: 100,000)")
# tf.flags.DEFINE_integer("over_sample_times", 0, "over_sample_times (default: 1)")

# Embedding params
tf.flags.DEFINE_integer("edim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("embedding_dense_size", 300, "Dimensionality of word embedding dense layer (default: 300)")
# tf.flags.DEFINE_string("label_embedding", "", "Init label embedding (default: 300)")
tf.flags.DEFINE_boolean("use_role_embedding", True, "Use role embedding or not  (default:True)")
tf.flags.DEFINE_integer("role_num", 4, "How many roles  (default: 3)")
tf.flags.DEFINE_integer("role_edim", 300, "Dimensionality of role embedding  (default: 100)")
tf.flags.DEFINE_integer("fact_edim", 300, "Dimensionality of fact embedding  (default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_integer("max_sequence_length", 140, "Max sentence sequence length (default: 140)")
# tf.flags.DEFINE_integer("num_classes", 41, "Number of classes (default: 41)")
tf.flags.DEFINE_integer("fact_num", 10, "Number of classes (default: 10)")
tf.flags.DEFINE_string("facts",
                       "是否存在还款行为,是否约定利率,是否约定借款期限,是否约定保证期间,是否保证人不承担担保责任,是否保证人担保,是否约定违约条款,是否约定还款期限,是否超过诉讼时效,是否借款成立",
                       "facts name")
# memmory network used_
tf.flags.DEFINE_integer("hops", 2, "Memory network hops")

# transformer used
# tf.flags.DEFINE_integer("transformer_layers", 2, "Transformer layers (default: 4)")
# tf.flags.DEFINE_integer("sen_transformer_layers", 0, "Sentence Level Transformer layers (default: 1)")
tf.flags.DEFINE_integer("heads", 4, "multi-head attention (default: 4)")
# tf.flags.DEFINE_integer("intermediate_size", 1000, "Intermediate size (default: 1000)")

# rnn used
tf.flags.DEFINE_integer("rnn_hidden_size", 128, "rnn hidden size (default: 300)")
tf.flags.DEFINE_integer("rnn_layer_num", 1, "rnn layer num (default: 1)")
# tf.flags.DEFINE_integer("rnn_attention_size", 400, "rnn attention dense layer size (default: 300)")
# tf.flags.DEFINE_integer("rnn_output_mlp_size", 500, "rnn output mlp size (default: 500)")
# tf.flags.DEFINE_integer("num_k", 15, "drnn window size (default: 15)")
# tf.flags.DEFINE_integer("ram_gru_size", 300, "recurrent attention gru cell size (default: 300)")
# tf.flags.DEFINE_integer("ram_times", 4, "recurrent attention times (default: 4)")
# tf.flags.DEFINE_integer("ram_hidden_size", 300,
#                         "recurrent attention final episode attention hidden size (default: 300)")

# cnn used
# tf.flags.DEFINE_string("filter_sizes", "1,2,3,4", "Comma-separated filter sizes (default: '1,2,3,4,5')")
# tf.flags.DEFINE_integer("fc1_dense_size", 512, "fc size before output layer (default: 2048)")
# tf.flags.DEFINE_string("num_filters", "64,64,64,64", "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("init_std", 0.01, "Init std value for variables (default: 0.01)")
# tf.flags.DEFINE_float("input_noise_std", 0.05, "Input for noise  (default: 0.01)")

# tf.flags.DEFINE_float("max_grad_norm", 10, "clip gradients to this norm (default: 10)")
# tf.flags.DEFINE_string("activation_function", "relu",
#                        "activation function used (default: relu) ")  # relu swish elu crelu tanh gelu

# Training parameters

tf.flags.DEFINE_integer("max_decoder_steps", 100, "max_decoder_steps")
tf.flags.DEFINE_boolean("fine_tuning", False, "fine_tuning from pretrained lm files")
tf.flags.DEFINE_boolean("continue_training", True, "continue training from restore, or start from scratch")
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 128)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 50)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "initial learning rate (default: 1e-3)")
tf.flags.DEFINE_float("decay_rate", 0.9, "learning rate decay rate (default: 0.7)")
tf.flags.DEFINE_integer("decay_step", 500, "learning rate decay step (default: 20000)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on valid set after this many steps (default: 1000)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("warm_up_steps_percent", 0.05, "Warm up steps percent (default: 5%)")

tf.flags.DEFINE_string("checkpoint_path",
                       "/home/admin/workspace/shared/team/intelligent-judge/maluyao/project/classification_model/judgment_prediction/trail_gen_runs/2019-10-31_00-08-11/checkpoints/model",
                       "Checkpoint file path without extension, as list in file 'checkpoints'")
tf.flags.DEFINE_string("pre_train_lm_checkpoint_path", "lm_runs_v2/2019-03-08_16-01-48/checkpoints/model-186000",
                       "Checkpoint file path from pre trained language model.'")

tf.flags.DEFINE_string("pre_trained_word_embeddings", "", "pre_trained_word_embeddings")
tf.flags.DEFINE_string("label_names", "", "name for each class")
tf.flags.DEFINE_string("cuda_device", "0", "GPU used")
tf.flags.DEFINE_string("model", "selfatt", "seq2seq/selfatt")

FLAGS = tf.flags.FLAGS
# FLAGS.filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
# FLAGS.num_filters = list(map(int, FLAGS.num_filters.split(",")))
FLAGS.facts = list(map(str, FLAGS.facts.split(",")))


# %%

def load_data():
    with open(FLAGS.all_data_file, 'r', encoding="utf-8") as load_f:
        data = json.load(load_f)
    jsonContent = data
    print(len(jsonContent))

    def _do_vectorize(jsonContent):
        keys = list(jsonContent.keys())
        tokenizer = Tokenizer()
        tokenizer = joblib.load("data/tokenizer.pickle")

        sentence_nums = []
        dialogue_word_ids = []
        sentences_lens = []
        roles = []
        claims = []
        claims_num = []
        claims_len = []
        for key in keys:
            dialogue = jsonContent[key]["tokenized_dialogue"]
            sentence_nums.append(len(dialogue))
            sentences_lens.append([len(x) for x in dialogue])
            roles_text = jsonContent[key]["roles"]
            roles_num = []
            for role in roles_text:
                if role == "审":
                    roles_num.append(1)
                elif role == "原":
                    roles_num.append(2)
                elif role == "被":
                    roles_num.append(3)
                elif role == "其他":
                    roles_num.append(4)
                else:
                    print("error")
            roles.append(roles_num)

            #             dialogue = tf.keras.preprocessing.sequence.pad_sequences(
            #                 dialogue, maxlen=FLAGS.max_sequence_length, padding="post", truncating="post", value=0
            #             )
            claims_split = jsonContent[key]["claims_split"]
            claims_cut = []
            for claim in claims_split:
                claim_cut = jieba.cut(claim)
                claims_cut.append(" ".join(claim_cut))
            claims_sequences = tokenizer.texts_to_sequences(claims_cut)
            claims_len.append([len(x) for x in claims_sequences])
            #             claims_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            #                 claims_sequences, maxlen=FLAGS.max_sequence_length, padding="post", truncating="post", value=0
            #             )
            claims.append(claims_sequences)
            claims_num.append(len(claims_split))
            dialogue_word_ids.append(dialogue)
        print(len(dialogue_word_ids))
        return dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len

    def _do_label_vectorize(jsonContent):
        keys = list(jsonContent.keys())

        fact_labels = []
        claims_labels = []

        for key in keys:
            fact_label_sample = jsonContent[key]["fact_labels"]
            fact_label_decode = []
            claim_label_decode = []
            for fact in FLAGS.facts:
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
        return fact_labels, claims_labels

    # encode x
    dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len = _do_vectorize(jsonContent)

    fact_labels, claims_labels = _do_label_vectorize(jsonContent)

    data = list(zip(dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len,
                    fact_labels, claims_labels))

    return data


# %%

print("\nParameters:")
'''
print(FLAGS.flag_values_dict())

print("\nLoading data...")

embedding_matrix = np.load("../model/embedding_matrix.npy")

print(embedding_matrix.shape)
FLAGS.pre_trained_word_embeddings = embedding_matrix
# FLAGS.label_names = label_names
'''
all_data = load_data()
train_data, other_data = train_test_split(all_data, test_size=0.2, random_state=2019)
_, train_hand_out = train_test_split(train_data, test_size=0.1, random_state=2019)
test_data, valid_data = train_test_split(other_data, test_size=0.5, random_state=2019)

print("train_data %d, valid_data %d, test_data %d." % (
    len(train_data), len(valid_data), len(test_data)))

data_dict = {
    "train_data_set": train_data,
    "test_data_set": test_data,
    "train_hand_out_set": train_hand_out,
    "valid_data_set": valid_data
}
print("\nSampling data...")
pass


# %%

def process_data(data):
    #     data = list(zip(dialogue_word_ids, roles, sentence_nums, sentences_lens, claims, claims_num, claims_len,
    #                     fact_labels, claims_labels))
    dialogue_word_ids = []
    fact_labels = []
    claims_labels = []
    for item in data:
        for claim in item[4]:
            dialogue_word_ids.append(item[0] + [claim])
        claims_labels.append(item[8])
    claims_labels = sum(claims_labels, [])
    print(len(dialogue_word_ids))
    print(len(claims_labels))

    return dialogue_word_ids, claims_labels


# %%

dialogue_word_ids = []
for item in train_data:
    dialogue_word_ids.append(item[0] + item[4])

print(len(dialogue_word_ids))

# %%

X_train, y_train = process_data(train_data)
X_test, y_test = process_data(test_data)

# %%

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

# %%

import numpy as np

dialogue_word_ids = np.asarray(dialogue_word_ids)
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# %%

new_dialogue_word_ids = [sum(a, []) for a in dialogue_word_ids]
new_dialogue_word_ids2 = [[str(char) for char in a] for a in new_dialogue_word_ids]
new_dialogue_word_ids3 = [" ".join(a) for a in new_dialogue_word_ids2]

new_X_train = [sum(a, []) for a in X_train]
new_X_train2 = [[str(char) for char in a] for a in new_X_train]
new_X_train3 = [" ".join(a) for a in new_X_train2]
new_X_test = [sum(a, []) for a in X_test]
new_X_test2 = [[str(char) for char in a] for a in new_X_test]
new_X_test3 = [" ".join(a) for a in new_X_test2]

# %%

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=0, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vec.fit(new_dialogue_word_ids3)
trn_term_doc = vec.transform(new_X_train3)
test_term_doc = vec.transform(new_X_test3)

# %%

lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc, y_train)
preds = lin_clf.predict(test_term_doc)

# %%

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

_macro_f1 = f1_score(y_test, preds, average="macro")
_micro_f1 = f1_score(y_test, preds, average="micro")

_val_recall = recall_score(y_test, preds, average="macro")
_val_precision = precision_score(y_test, preds, average="macro")
print("— macro_f1: % f — val_precision: % f — val_recall % f" % (_macro_f1, _val_precision, _val_recall))
print("— micro_f1: % f" % (_micro_f1))

# %%

from collections import Counter

sorted_target = sorted(Counter(y_test).items())
sorted_preds = sorted(Counter(preds).items())
print(sorted_target)
print(sorted_preds)
target_names = ['驳回诉请', '部分支持', "支持诉请"]
print(classification_report(y_test, preds, target_names=target_names, digits=4))
