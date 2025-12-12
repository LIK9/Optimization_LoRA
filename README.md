# Optimization_LoRA
LoRA Analysis and Optimal Module Identification for Persona Dialogue Generation Task

## Overview

This repository contains the official implementation for analyzing the effectiveness of **LoRA (Low-Rank Adaptation)** in the **Persona-Based Dialogue Generation (PDG)** task.  
We aim to identify *which attention modules benefit most from LoRA adaptation* by systematically evaluating their contributions to persona consistency and response fluency.

While LoRA is widely used as a parameter-efficient fine-tuning method, it is often applied uniformly across all attention projections. This work challenges that assumption by providing a principled, module-level analysis.

---

## Motivation

Persona-based dialogue generation requires models to satisfy two competing objectives:

1. Generate **fluent and contextually appropriate responses**
2. Maintain **consistency with a predefined persona**

Although LoRA significantly reduces training cost, its low-rank constraint can limit expressive capacity. Moreover, applying LoRA to all modules increases inference latency in multi-tenant deployment scenarios.  
To address this, we ask:

> **Which LoRA modules actually matter for persona-grounded dialogue generation?**

---

## Method

We analyze LoRA-applied attention modules using **Shapley value analysis**, a cooperative game-theoretic approach that quantifies the marginal contribution of each component.

- **Players**: Attention projection modules (**Query, Key, Value**)
- **Value functions**:
  - Persona consistency (**C-Score**)
  - Response fluency (**ROUGE-L**)

By evaluating all combinations of these modules, we estimate how much each contributes to overall performance and identify optimal LoRA configurations.

---

## Experimental Setup

### Dataset
- **PersonaChat** (Zhang et al., 2018)
  - Train: 8,939 dialogues
  - Validation: 1,000 dialogues
  - Test: 968 dialogues

### Model
- **Qwen2.5-Instruct (1.5B)**

### LoRA Configuration
- Rank: 8  
- Scaling factor (α): 16  
- Dropout: 0.1  

All experiments are conducted under identical training settings for fair comparison.

---

## Key Results

- **LoRA applied to Query & Value projections** achieves the best balance between:
  - Persona consistency
  - Response fluency
  - Training efficiency
- Shapley value ranking consistently follows:

```text
Value > Query > Key


---

## How to Run

All runnable scripts are located in the `RUNS/` directory.  
Each script corresponds to a specific stage of training or evaluation and can be executed directly from the command line.

### Training

The following script trains the model using LoRA.  
You can control which attention projections (Query / Key / Value) receive LoRA by toggling the corresponding flags.

**`RUNS/train.sh`**
```bash
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
  --lora_v 1
  --lora_v 1













