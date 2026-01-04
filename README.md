# 🧠 LLM Training Pipeline: From Pre-training to SFT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-red?logo=pytorch)
![Status](https://img.shields.io/badge/Status-Educational_Project-green)

This repository contains an educational project exploring the complete lifecycle of Large Language Models (LLMs). The goal was to master the implementation of two critical stages in LLM development: **Pre-training from scratch** and **Supervised Fine-Tuning (SFT)** using modern NLP libraries.

> **⚠️ Note on Project Structure**
> This project is designed as a modular playground to demonstrate implementation skills across different model architectures. For educational purposes, the two stages are executed independently:
> * **Part 1 (Pre-training):** Trains a custom **Llama** architecture from scratch on a domain-specific corpus.
> * **Part 2 (SFT):** Fine-tunes a pre-trained **Qwen2.5** model on instructions.
>
> While a production pipeline would typically pipeline these steps on a single model, this project treats them as separate experiments to showcase versatility with different tools (`Trainer` vs `SFTTrainer`) and initialization methods.


## 🛠️ Project Stages

### 1. Pre-training from Scratch (Domain Adaptation)
In this stage, I built a complete pipeline to train a language model on a raw text corpus of **Russian novels**. The objective was to understand how models learn syntax and grammar from a "blank slate."

* **Data Pipeline:** Implemented data cleaning, deduplication, and non-Cyrillic filtering to prepare the `RussianNovels` dataset.
* **Tokenizer:** Trained a custom **BPE (Byte-Pair Encoding)** tokenizer specifically optimized for the dataset vocabulary.
* **Architecture:** Initialized a `LlamaForCausalLM` from scratch (`LlamaConfig`) with ~177M parameters.
* **Training:** Used the standard Hugging Face `Trainer` for Causal Language Modeling (CLM).
* **Monitoring:** Implemented a custom `GenerationCallback` to visually evaluate text generation progress at the end of each epoch.

### 2. Supervised Fine-Tuning (Instruction Tuning)
In this stage, I focused on the alignment phase, turning a base model into a helpful assistant that can follow instructions.

* **Base Model:** Utilized `Qwen/Qwen2.5-0.5B` as a strong baseline.
* **Dataset:** Processed the `d0rj/alpaca-cleaned-ru` dataset, converting raw Alpaca-style inputs into a conversational dialogue format.
* **Library:** Leveraged the **TRL (Transformer Reinforcement Learning)** library.
* **Optimization:** Used `SFTTrainer` with `bfloat16` precision and gradient accumulation to optimize GPU memory usage.

## 💻 Tech Stack

* **Libraries:** `transformers`, `trl`, `datasets`, `tokenizers`, `torch`
* **Models:** Llama (Custom Config), Qwen2.5
* **Hardware Optimization:** CUDA support, Mixed Precision Training (`fp16`/`bf16`)

## 🚀 Key Learnings & Features implemented

- [x] **Custom Data Chunking:** Grouping tokenized text into context windows (e.g., 512 or 1024 tokens) for efficient CLM training.
- [x] **Tokenizer Training:** Understanding special tokens (`<bos>`, `<eos>`, `<pad>`) and vocabulary sizing.
- [x] **Custom Callbacks:** Writing Python classes to hook into the training loop for real-time inference logging.
- [x] **Instruction Formatting:** Mapping datasets to chat templates for SFT.

## 📊 Results & Limitations

* **Pipeline Validation:** Both Pre-training and SFT pipelines executed successfully without errors.
* **Resource Constraints:** Due to hardware limitations (Google Colab free tier), the SFT phase was limited to **1 epoch** instead of the planned 3.
* **Observations:** Even with limited training steps, the Qwen2.5 model showed immediate improvement in following Russian instructions, confirming that the data formatting and training loop were implemented correctly.

