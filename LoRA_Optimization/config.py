import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--seed', type=int, default=26)
    parser.add_argument('--train_data', type=str, default="./original_dataset/train_self_original.json")
    parser.add_argument('--valid_data', type=str, default="./original_dataset/valid_self_original.json")
    parser.add_argument('--test_data', type=str, default="./original_dataset/test_self_original.json")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--train_batch', type=int, default=8)
    parser.add_argument('--prompt_test', type=str, default=None)
    parser.add_argument('--log_step', type=int, default=1)
    parser.add_argument('--save_step', type=int, default=100)



    # train hyperparameter
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--grad_acc_step', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default="paged_adamw_32bit")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_step', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")


    # LoRA
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--use_lora', type=int, default=0)
    parser.add_argument('--lora_q', type=int, default=0)
    parser.add_argument('--lora_k', type=int, default=0)
    parser.add_argument('--lora_v', type=int, default=0)


    # inference
    parser.add_argument('--trained_model_dir', type=str, default="None")
    parser.add_argument('--inference_batch', type=int, default=32)
    parser.add_argument('--num_beam', type=int, default=1)
    parser.add_argument('--do_sample', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=0)
    parser.add_argument('--trained', type=int, default=1)

    










    return parser.parse_args()