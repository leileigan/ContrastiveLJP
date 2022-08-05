export PYTHONPATH=${PWD}
for t in 10 512 1024 2020 9090
do
  CUDA_VISIBLE_DEVICES=0 taskset -c 10,11,12,13,14,15,16,17,18,19,20 python train_NeurJudge.py \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --HP_hidden_dim 150 \
  --seed $t > logs/neu_judge/lr1e-3_seed${t}_bs128_hdim150_charge_law_term.log 2>&1
done