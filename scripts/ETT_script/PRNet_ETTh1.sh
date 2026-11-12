#!/usr/bin/env bash
# Run from the repository root after: conda activate PRNet
# Data file: ./datasets/ETTh1.csv
# Prediction lengths: 96, 192, 336, 720.
# Each component has an independent encoder and a single cross-attention decoder.
set -e

export CUDA_VISIBLE_DEVICES=0

for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./datasets/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_${pred_len} \
    --model PRNet \
    --data ETTh1 \
    --features M \
    --target OT \
    --freq h \
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
    --revin_affine 1 \
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
