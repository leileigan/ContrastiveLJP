#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python bert_lm_fine_tuning.py \
--bert_model /home/ganleilei/workspace/bert \
--bert_model_tokenizer /home/ganleilei/workspace/bert/bert-base-chinese-vocab.txt \
--do_train \
--train_file ../data/ctb6.0/origin/ctb_bert_fine.txt \
--output_dir /home/ganleilei/Workspace/bert/bert_fine_ctb \
--num_train_epochs 5 \
--learning_rate 5e-6 \
--train_batch_size 16 \
--max_seq_length 200
