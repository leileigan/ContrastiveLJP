export PYTHONPATH=${PWD}
for t in 10 1024 2020 512 9090
do
  CUDA_VISIBLE_DEVICES=4 python train_harnn_contra_v2.py --alpha 0.1 --HP_lr 5e-4 --seed ${t} \
  --HP_batch_size 32 \
  --bert_path /data/ganleilei/bert/bert-base-chinese > logs/harnn_contra_accu/harnn_contra_accu_alpha0.1_lr5e-4_seed${t}_bs32.log 2>&1
done