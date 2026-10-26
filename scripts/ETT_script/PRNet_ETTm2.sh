#!/usr/bin/env bash
# Run from the repository root after: conda activate PRNet
# Data file: ./datasets/ETTm2.csv
# Prediction lengths: 96, 192, 336, 720.
# e_layers controls the shared encoder. PRNet has one shared decoder.
set -e

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./datasets/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_96_${pred_len} \
    --model PRNet \
    --data ETTm2 \
    --features M \
    --target OT \
    --freq t \
    --seq_len 96 \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 64 \
    --n_heads 4 \
    --e_layers 1 \
    --d_ff 128 \
    --dropout 0.1 \
    --embed timeF \
    --patch_len 16 \
    --patch_stride 8 \
    --future_patch_len 16 \
    --ma_short 5 \
    --ma_long 25 \
    --top_k 4 \
    --num_queries 5 \
    --router_hidden 64 \
    --min_period_overlap 8 \
    --revin_affine 1 \
    --initial_period_bias 1.0 \
    --train_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --patience 3 \
    --loss mse \
    --lradj type1 \
    --num_workers 0 \
    --checkpoints ./checkpoints/ \
    --gpu 0 \
    --des Exp \
    --itr 1
done
