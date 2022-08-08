export PYTHONPATH=${PWD}
for t in 10 512 1024 2020 9090
do
  CUDA_VISIBLE_DEVICES=2 python train_NeurJudge_moco_hmc.py \
  --bert_path /data/ganleilei/bert/bert-base-chinese \
  --HP_hidden_dim 150 \
  --alpha 10 \
  --layer_penalty 1 0 0 \
  --seed $t > logs/neu_judge_moco_hmc/lr1e-3_seed${t}_alpha10.0_bs128_hdim150_p1,0,0_charge_law_term.log 2>&1
done