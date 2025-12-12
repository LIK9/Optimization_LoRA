import config
from datasets import load_dataset
from tqdm import tqdm
from peft import get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed
import torch
import os
from functools import partial
import prompt
import json
from math import ceil
from evaluate import load
from utils.test_utils import *
from itertools import combinations
import matplotlib.pyplot as plt

def get_model(args, trained_model_dir):
    if args.trained:
        tokenizer = AutoTokenizer.from_pretrained(trained_model_dir, use_fast=True, padding_side = 'left', fix_mistral_regex=True)
        if args.use_lora:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                return_dict=True,
                low_cpu_mem_usage=True,
                device_map={"": torch.cuda.current_device()},
            )
            model = PeftModel.from_pretrained(base_model, trained_model_dir)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                trained_model_dir,
                return_dict=True,
                low_cpu_mem_usage=True,
                device_map={"": torch.cuda.current_device()},
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
                return_dict=True,
                low_cpu_mem_usage=True,
                device_map={"": torch.cuda.current_device()},
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side = 'left', fix_mistral_regex=True)
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def inference(args):
    epoch = 2

    if args.trained:
        trained_model_dir = f'{args.trained_model_dir}/epoch{epoch}'
        output_file_path = f"{trained_model_dir}/response.json"
    else:
        trained_model_dir = './outputs/pretrained'
        output_file_path = f"./outputs/pretrained/response.json"


    model, tokenizer = get_model(args, trained_model_dir)


    model.config.use_cache=True
    model.eval()

    dataset = load_dataset("json", data_files={"test": args.test_data})            
    dataset = dataset.map(partial(prompt.format_prompt_test_single, tokenizer=tokenizer), load_from_cache_file=False)

    batch_size = args.inference_batch

    prompts = [ex["text"] for ex in dataset["test"]]
    print(prompts[0])

    num_batches = ceil(len(prompts) / batch_size)


    progress_bar = tqdm(range(num_batches))
    predictions = []

    for batch_idx in progress_bar:


        batch_prompts = prompts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs["input_ids"].to(model.device)



        attention_mask = inputs["attention_mask"].to(model.device)


        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=args.do_sample,
                num_beams=args.num_beam,
                pad_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                top_p=args.top_p
            )
        for i in range(len(batch_prompts)):
            prompt_lens = len(input_ids[i])

            gen_ids = outputs[i][prompt_lens:] # outputs에는 instruction prompt도 포함되어서 나오기 때문
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predictions.append(pred)


        # break

    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_list = json.load(f)

    save_list = []
    for idx in range(len(predictions)):
        if predictions[idx] != None:
            test_list[idx]["generated_response"] = predictions[idx]
            save_list.append(test_list[idx])

    print(f"save list len: {len(save_list)}")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)

    compute_metric(trained_model_dir)


def compute_metric(trained_model_dir):
    # model_file = "pretrained"

    # response_file_path = f"./outputs/{model_file}/epoch2/response.json"
    # output_file_path = f"./outputs/{model_file}/epoch2/result.json"

    response_file_path = f"{trained_model_dir}/response.json"
    output_file_path = f"{trained_model_dir}/result.json"


    rouge = load("rouge")
    bleu = load("bleu")
    bertscore = load("bertscore")
    meteor = load("meteor")

    predictions = []
    references = []    
    agent_persona_list = []

    with open(response_file_path, "r", encoding="utf-8") as f:
        original_panda_list = json.load(f)


    for original_panda_dict in tqdm(original_panda_list, total=len(original_panda_list)):

        agent_persona = original_panda_dict["persona_info"]
        predictions.append(original_panda_dict["generated_response"])
        references.append(original_panda_dict["dialogue"][-1])
        agent_persona_list.append(agent_persona)


    p_f1_agent = get_p_f1(predictions, agent_persona_list)
    c_score_agent = get_c_score(predictions, agent_persona_list)



    bleu1_score = compute_bleu1(predictions, references)
    bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])["rougeL"]
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]

    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en", device = 'cuda:0', batch_size = 256)
    bertscore_f1 = np.mean(bertscore_result["f1"])

    test_result_dict = {
        'BLEU1': round(bleu1_score, 3),
        'BLEU': round(bleu_score, 3),
        'ROUGE-L': round(rouge_score, 3),
        'METEOR': round(meteor_score, 3),
        'BERTScore-F1': round(bertscore_f1, 3),
        'c-score': round(c_score_agent, 3),
        'p_f1': round(p_f1_agent, 3),
    }

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(test_result_dict, f, ensure_ascii=False, indent=4)
    
    # break


def gpu_usage(setting):
    file_name = f"./outputs/{setting}/train_process.json"
    with open(file_name, 'r', encoding='utf-8') as f:
        data_file = json.load(f)

    gpu_usage = data_file["gpu"]
    print(sum(gpu_usage) / len(gpu_usage))


def calculate_shap(values):
    players = ['q', 'k', 'v']
    n = len(players)
    fact = [math.factorial(i) for i in range(n + 1)]

    phi = {i: 0.0 for i in players}
    N = set(players)

    for i in players:
        for r in range(0, n): 
            for S in combinations(N - {i}, r):
                S = set(S)
                weight = fact[len(S)] * fact[n - len(S) - 1] / fact[n]
                S_set = frozenset(S)
                S_i_set = frozenset(S | {i})
                phi[i] += weight * (values[S_i_set] - values[S_set])

    shap_values = phi
    return shap_values



if __name__ == '__main__':
    args = config.get_args()
    set_seed(args.seed)
    
    # gpu_usage('q,k,v')
    # compute_metric()

    inference(args)
    

