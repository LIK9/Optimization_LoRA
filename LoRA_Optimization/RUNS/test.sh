export CUDA_VISIBLE_DEVICES=0

python test.py \
  --trained_model_dir ./outputs/q \
  --inference_batch 32 \
  --num_beam 1 \
  --do_sample 1 \
  --temperature 1.0 \
  --top_p 1.0 \
  --use_lora 1 \
  --trained 1 \

