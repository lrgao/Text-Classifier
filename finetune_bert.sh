#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python cli_classifier.py \
        --do_train \
        --output_dir out/aspect_classifier \
        --train_file data/example/train_remove_ttv.json \
        --predict_file data/example/valid.json \
        --test_file data/example/test.json \
        --label_file data/aspect_label_map.json \
        --model_config data/aspect_config.json \
        --model_path bert-base-chinese \
        --tokenizer_path bert-base-chinese \
        --attribute aspect \
        --train_batch_size 512 \
        --predict_batch_size 128 \
        --max_input_length 112 \
        --append_another_bos \
        --learning_rate 1e-4 \
        --num_train_epochs 100 \
        --warmup_steps 1600 \
        --eval_period 1000

# CUDA_VISIBLE_DEVICES=1,2,3,0 python cli_classifier.py \
#         --do_train \
#         --output_dir out/user_type_classifier \
#         --train_file data/example/train_remove_ttv.json \
#         --predict_file data/example/valid.json \
#         --test_file data/example/test.json \
#         --label_file data/uc_label_map.json \
#         --model_config data/uc_config.json \
#         --model_path bert-base-chinese \
#         --tokenizer_path bert-base-chinese \
#         --attribute user_category \
#         --train_batch_size 512 \
#         --predict_batch_size 128 \
#         --max_input_length 112 \
#         --append_another_bos \
#         --learning_rate 1e-4 \
#         --num_train_epochs 100 \
#         --warmup_steps 1600 \
#         --eval_period 1000