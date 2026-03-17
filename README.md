# Neural Chain-of-Thought Search (NCoTS)

[![paper](https://img.shields.io/badge/cs.CL-2601.11340-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2601.11340)
[![model](https://img.shields.io/badge/🤗-hugging_face-yellow)](https://huggingface.co/linggm/head_1.5b)
[![license](https://img.shields.io/github/license/MilkThink-Lab/Neural-CoT-Search.svg)](LICENSE)

This repository is the official codebase of our paper "Neural Chain-of-Thought Search: Searching the Optimal Reasoning Path to Enhance Large Language Models" [[paper]](https://arxiv.org/abs/2601.11340). NCoTS reformulates CoT reasoning as a dynamic search for the optimal thinking strategy, achieving a Pareto improvement — boosting accuracy by over 3.5% while reducing generation length by over 22%.

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

#### Training

```bash
python main_train.py \
    --slm_model_path <path_to_1.5B_model> \
    --llm_model_path <path_to_7B_model> \
    --dataset_name <path_to_dataset> \
    --num_epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-4
```

#### Evaluation

```bash
python evaluation.py \
    --checkpoint <path_to_head1_checkpoint> \
    --dataset <dataset_name> \
    --max_new_tokens 4096 \
    --output_dir ./evaluation_results \
    --verbose
```


## 📁 Project Structure

```
├── config.py            # Configuration dataclasses
├── model.py             # DualHeadModel, SwitchDetector, SpecialVocabMapper
├── trainer.py           # Distillation training loop
├── inference.py         # Dual-head inference engine
├── evaluation.py        # Benchmark evaluation (math, multiple-choice)
├── data_utils.py        # Dataset loading and preprocessing
├── main_train.py        # Training entry point
└── main_inference.py    # Inference entry point
```

