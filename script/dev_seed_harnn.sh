export PYTHONPATH=${PWD}
for t in 10 1024 2020 512 9090
do
  CUDA_VISIBLE_DEVICES=0 python train_harnn.py --HP_lr 5e-4 --seed ${t} \
  --HP_batch_size 32 \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --data_path /data/ganleilei/law/ContrastiveLJP/ \
  --savemodel /data/ganleilei/law/ContrastiveLJP/models/harnn/ \
  --savedset /data/ganleilei/law/ContrastiveLJP/models/harnn/data \
  --embedding_path /data/ganleilei/law/ContrastiveLJP/cail_thulac.npy > logs/harnn/harnn_lr5e-4_seed${t}_bs32.log 2>&1
done