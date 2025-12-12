import re
import string
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed,  BertTokenizer, BertForSequenceClassification
import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
import json


def get_p_f1(predictions, persona):
    mean_p_f1 = 0.0
    p_f1_list = []
    for idx, pred in enumerate(predictions):
        persona_list = persona[idx]
        p_fl = compute_p_f1(pred, persona_list)
        mean_p_f1 += (p_fl / len(predictions))
        p_f1_list.append(p_fl)
    return mean_p_f1


def normalize_text(text):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(prediction, attribute):

    pred_tokens = prediction.split()
    truth_tokens = attribute.split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)

def compute_p_f1(pred, persona_list):
    norm_pred = normalize_text(pred)

    if persona_list is not None:
        f1 = 0
        for p in persona_list:
            att = normalize_text(p.strip())
            _f1 = compute_f1(norm_pred, att)
            f1 += _f1
        f1 = f1/len(persona_list)
    else:
        'persona list empty error'
        f1=None

    return f1


def get_c_score(predictions, persona):
    nli_model_dir = "./nli/model"
    tokenizer = BertTokenizer.from_pretrained(nli_model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(nli_model_dir, num_labels=3).to(device)

    mean_nli_value = 0.0

    for idx, pred in tqdm(enumerate(predictions), total=len(predictions)):
        persona_list = persona[idx]
        nli_value = get_nli_value(response=pred, persona_list=persona_list, model=model, tokenizer=tokenizer)
        mean_nli_value += nli_value / len(predictions)

    return mean_nli_value

def get_nli_value(response, persona_list, model, tokenizer):
    label_map = {"entail": 0, "neutral": 1, "negative": 2}
    nli_value = 0
    premises = [response] * len(persona_list)
    hypotheses = persona_list
    encodings = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids = encodings['input_ids'], attention_mask = encodings['attention_mask'])
        preds = outputs.logits.argmax(dim=1).cpu().tolist() 
        
        for idx, pred in enumerate(preds):
            if pred == 0:
                nli_value += 1
            elif pred == 1:
                nli_value += 0
            else:
                nli_value += -1

    return nli_value

def compute_bleu1(predictions, references):
    smoothie = SmoothingFunction().method1
    tokenized_hypotheses = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references] 
    return corpus_bleu(tokenized_references, tokenized_hypotheses, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothie)
