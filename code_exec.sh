#!/bin/bash

MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
MODEL_PATH=/mnt/sakura-bs/koshieguchi/converted_checkpoint/Meta-Llama-3-8B-Instruct/LLTM-all-numeric-depth_lr_2e-5-minlr_4e-6_GB_64_3epoch/iter_0006849/

python -m lcb_runner.runner.main \
  --model $MODEL_NAME \
  --scenario codeexecution \
  --cot_code_execution \
  --evaluate \
  --local_model_path $MODEL_PATH \
  --tensor_parallel_size 1 \
  --max_tokens 8192 \
