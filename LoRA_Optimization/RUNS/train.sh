export CUDA_VISIBLE_DEVICES=0

python train.py \
  --train_epochs 3 \
  --train_batch 4 \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --lora_rank 8 \
  --lora_alpha 16 \
  --use_lora 1 \
  --lora_q 1 \
  --lora_k 1 \
  --lora_v 1 \
  

