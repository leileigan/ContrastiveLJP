export PYTHONPATH=${PWD}
for t in 10 1024 2020 512 9090
do
  CUDA_VISIBLE_DEVICES=1 python train_harnn_contra_v3.py --alpha 0.1 --HP_lr 5e-4 --seed ${t} \
  --HP_batch_size 32 \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --data_path /data/ganleilei/law/ContrastiveLJP/ \
  --savemodel /data/ganleilei/law/ContrastiveLJP/models/harnnContra_v3/ \
  --savedset /data/ganleilei/law/ContrastiveLJP/models/harnnContra_v3/data \
  --moco_queue_size 65536 \
  --HP_iteration 50 \
  --confused_matrix /data/ganleilei/law/ContrastiveLJP/confused_matrix.npy \
  --embedding_path /data/ganleilei/law/ContrastiveLJP/cail_thulac.npy > logs/harnn_contra_accu_v3/harnn_contra_accu_v3_alpha0.1_lr5e-4_seed${t}_queue65536_bs32.log 2>&1
done