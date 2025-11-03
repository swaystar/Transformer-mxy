#!/usr/bin/env bash
set -euo pipefail

python -m src.train \
  --dataset tiny_shakespeare \
  --seq_len 256 \
  --vocab byte \
  --d_model 256 \
  --n_heads 4 \
  --n_layers 4 \
  --ffn_hidden 1024 \
  --dropout 0.1 \
  --lr 3e-4 \
  --optimizer adamw \
  --weight_decay 0.01 \
  --scheduler cosine \
  --warmup_steps 500 \
  --batch_size 64 \
  --max_steps 5000 \
  --eval_interval 200 \
  --seed 3407 \
  --results_dir results/exp_tinysha_seed3407



