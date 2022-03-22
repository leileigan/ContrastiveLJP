export PYTHONPATH=${PWD}
for t in 10 1024 2020 512 9090
do
  CUDA_VISIBLE_DEVICES=0 python train_harnn_contra_v1.py --alpha 0.1 --HP_lr 5e-4 --seed ${t} \
  --HP_batch_size 32 \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --data_path /data/ganleilei/law/ContrastiveLJP/ \
  --savemodel /data/ganleilei/law/ContrastiveLJP/models/harnnContra_v1/ \
  --savedset /data/ganleilei/law/ContrastiveLJP/models/harnnContra_v1/data \
  --temperature 0.07 \
  --embedding_path /data/ganleilei/law/ContrastiveLJP/cail_thulac.npy > logs/harnn_contra_v1/harnn_contra_accu_v1_alpha0.1_lr5e-4_seed${t}_bs32.log 2>&1
done