export PYTHONPATH=${PWD}
for t in 10 1024 2020 512 9090
do
  CUDA_VISIBLE_DEVICES=0 python train_ladan_mtl.py --HP_lr 1e-3 --seed ${t} \
  --HP_batch_size 128 \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --data_path /data/ganleilei/law/ContrastiveLJP/ \
  --savemodel /data/ganleilei/law/ContrastiveLJP/models/ladan_mtl/ \
  --HP_iteration 150 \
  --word2id_dict /data/ganleilei/law/ContrastiveLJP/w2id_thulac.pkl \
  --embedding_path /data/ganleilei/law/ContrastiveLJP/cail_thulac.npy > logs/ladan_mtl/ladan_mtl_lr1e-3_seed${t}_bs128.log 2>&1
done