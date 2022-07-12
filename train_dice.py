
import argparse
from unittest import defaultTestLoader
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import optim
from models.model_DICE import DICE

import numpy as np
import time
from tqdm import tqdm
import os
import pickle as pk

def getargs():
    parser = argparse.ArgumentParser(description="this model used for number embed")
    
    parser.add_argument('--embedding_path', default='/data/ganleilei/law/ContrastiveLJP/cail_thulac.npy')
    parser.add_argument('--word_embd_dim', default=200)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.05, type=float)

    parser.add_argument('--savemodel', default="/data/ganleilei/law/ContrastiveLJP/results/Dice/")    
    parser.add_argument('--seed', default=2020, type=int)
    parser.add_argument('--num_dice', default=10000, type=int)
    args = parser.parse_args()

    return args

class Config():
    def __init__(self) -> None:
        self.savemodel = None
        self.seed = 2020
        self.num_dice = 10000

        self.embedding_path = '/data/ganleilei/law/ContrastiveLJP/cail_thulac.npy'
        self.word_embd_dim = 200
        self.hidden_dim = 512
        self.epoch = 100
        self.batch_size = 128
        self.lr = 1e-3
        self.lr_decay = 0.05
        self.clip = 5.0

        self.pretrain_word_embedding = None
        self.word2id_dict = {str(i):i for i in range(10)}
        self.id2word_dict = {i:str(i) for i in range(10)}
        

    def show(self):
        print('Config summary start:')
        print('             savemodel: %s' % (self.savemodel))
        print('             seed: %d' % (self.seed))
        print('             seed: %d' % (self.num_dice))

        print('             embedding_path: %s' % (self.embedding_path))
        print('             word2id_dict: %s' % (self.word2id_dict))
        print('             word_embd_dim: %d' % (self.word_embd_dim))
        print('             hidden_dim: %d' % (self.hidden_dim))
        print('             epoch: %d' % (self.epoch))
        print('             batch_size: %d' % (self.batch_size))
        print('             lr: %f' % (self.lr))
        print('             lr_decay: %f' % (self.lr_decay))
        print('             clip: %f' % (self.clip))
        print('             word2id_dict:', self.word2id_dict)
        print('             id2word_dict: %f', self.id2word_dict)

        print('config summary end')
    
    def load_word_pretain_embd(self, path):
        self.pretrain_word_embedding = np.cast[np.float32](np.load(path))
        self.word_embd_dim = self.pretrain_word_embedding.shape[1]

def args2config(args, config):
    config.savemodel = args.savemodel
    config.seed = args.seed
    config.num_dice = args.num_dice
    config.show()

class DiceDataset(Dataset):
    def __init__(self, config) -> None:
        super(DiceDataset, self).__init__()
        self.config = config

    def __len__(self):
        return config.batch_size * 1000
    
    def __getitem__(self, index):
        num1 = np.random.randint(0, config.num_dice) * 500
        num2 = np.random.randint(0, config.num_dice) * 500 # * 1000
        num_label = 2*np.abs(num1 - num2) / (num1+num2)
        num1 = [self.config.word2id_dict[w] for w in list(str(num1))]
        num2 = [self.config.word2id_dict[w] for w in list(str(num2))]

        return num1, num2, num_label

word2id_dict = None

def collate(batch):
    num1_list, num2_list, label_list = [], [], []

    for item in batch:
        num1_list.append(item[0]) #[bsz, num_seq_len]
        num2_list.append(item[1]) #[bsz, num_seq_len]
        label_list.append(item[2])

    max_num1_len = max([len(item) for item in num1_list])
    padded_num1_lists = []
    for item in num1_list:
        padded_num1_lists.append([word2id_dict['0']] *(max_num1_len-len(item)) + item)

    max_num2_len = max([len(item) for item in num2_list])
    padded_num2_lists = []
    for item in num2_list:
        padded_num2_lists.append([word2id_dict['0']] *(max_num2_len-len(item)) + item)

    padded_num1_lists = torch.LongTensor(padded_num1_lists).cuda()
    padded_num2_lists = torch.LongTensor(padded_num2_lists).cuda()
    batch_num_label_lists = torch.FloatTensor(label_list).cuda()
    return padded_num1_lists, padded_num2_lists, batch_num_label_lists

def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr * ((1 - decay_rate) ** epoch)
    print(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def evaluate(model, valid_dataloader, mode):
    model.eval()
    ground_y, predicts_y = [], []
    loss = 0
    for batch_idx, datapoint in enumerate(valid_dataloader):
        num1, num2, label = datapoint
        dice_loss, preds = model(num1, num2, label)
        loss += dice_loss.data

    # ans = loss / config.epoch
    print("%s results: %.2f" % (mode, loss))
    return loss.item()

def train(model, data_dict, config):
    train_data_set = data_dict["train_data_set"]
    valid_data_set = data_dict["valid_data_set"]
    test_data_set = data_dict["test_data_set"]

    train_dataloader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    valid_dataloader = DataLoader(valid_data_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_dataloader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    print("Training model...")
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.Adam(parameters, lr=config.lr)
    print('optimizer: ', optimizer)
    best_dev = 100
    no_imporv_epoch = 0

    for idx in range(config.epoch):
        epoch_start = time.time()
        tmp_start = epoch_start
        print("epoch: %d/%d" % (idx, config.epoch))
        optimizer = lr_decay(optimizer, idx, config.lr_decay, config.lr)
        sample_dice_loss = 0

        model.train()
        model.zero_grad()

        batch_size = config.batch_size
        ground_y, predicts_y = [], []
        cnt = 0
        for batch_idx, datapoint in enumerate(tqdm(train_dataloader)):
            # cnt += 1
            num1, num2, label = datapoint
            dice_loss, preds = model(num1, num2, label)
            
            sample_dice_loss += dice_loss.data

            ground_y.extend(label.tolist())
            predicts_y.extend(preds.tolist())

            if (batch_idx + 1 ) % 100 == 0:
                tmp_time = time.time()
                tmp_cost = tmp_time - tmp_start
                tmp_start = tmp_time
                print("Instance: %s; Time: %.2fs; Loss: %.2f;" % ((batch_idx + 1), tmp_cost, sample_dice_loss))
                sample_dice_loss = 0

            dice_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)
            optimizer.step()
            model.zero_grad()
        
        current_score = evaluate(model, valid_dataloader, "valid")
        if current_score < best_dev:
            print("Exceed previous best acc score:", best_dev)
            no_imporv_epoch = 0
            best_dev = current_score
            # save model
            model_name = os.path.join(config.savemodel, "best.ckpt")
            torch.save(model.state_dict(), model_name)
        else:
            no_imporv_epoch += 1
            if no_imporv_epoch >= 20:
                print("early stop")
                break
       
        _ = evaluate(model, test_dataloader, "test")


def save_config(config, path):
    pk.dump(config, open(path, "wb"))
    print("save config path: %s" % (path))

if __name__ == '__main__':

    args = getargs()
    config = Config()
    args2config(args, config)
    save_config(config, os.path.join(config.savemodel, "data.dset"))
    word2id_dict = config.word2id_dict

    train_dataset = DiceDataset(config)
    valid_dataset = DiceDataset(config)
    test_dataset = DiceDataset(config)
    print("train_data %d; tvalid_data %d; test_data %d" % (len(train_dataset), len(valid_dataset), len(test_dataset)))

    data_dict = {
        "train_data_set": train_dataset,
        "valid_data_set": valid_dataset,
        "test_data_set": test_dataset
    }

    model = DICE(config)
    model.cuda()

    print("\nTraining...")
    train(model, data_dict, config)
