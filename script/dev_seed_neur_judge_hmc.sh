export PYTHONPATH=${PWD}
for t in 10 512 1024 2020 9090
do
  CUDA_VISIBLE_DEVICES=1 python train_NeurJudge_moco_hmc.py \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --HP_hidden_dim 150 \
  --seed $t > logs/neu_judge_moco_hmc/lr1e-3_seed${t}_bs128_hdim150.log 2>&1
done