#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python cli_classifier.py \
        --do_predict \
        --output_dir out/aspect_classifier \
        --predict_file data/example/test.json \
        --label_file data/aspect_label_map.json \
        --tokenizer_path bert-base-chinese \
        --model_config data/aspect_config.json \
        --attribute aspect \
        --predict_batch_size 128 \
        --max_input_length 112 \
        --append_another_bos \
        --prefix test_

# CUDA_VISIBLE_DEVICES=0 python cli_classifier.py \
#         --do_predict \
#         --output_dir out/user_type_classifier \
#         --predict_file data/example/test.json \
#         --label_file data/uc_label_map.json \
#         --tokenizer_path bert-base-chinese \
#         --model_config data/uc_config.json \
#         --attribute user_category \
#         --predict_batch_size 128 \
#         --max_input_length 112 \
#         --append_another_bos \
#         --prefix test_
