# End-to-End LLM Training Pipeline: From Pre-training to SFT
Full LLM Development Cycle for the Russian language

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)
![Status](https://img.shields.io/badge/Status-Educational_Project-green)

The project combines two major tasks in a single pipeline:
1.  **Domain Adaptation:** Pre-training a language model on a corpus of Russian Classical Literature.
2.  **Instruction Tuning (SFT):** Fine-tuning the model to follow instructions using the Alpaca dataset.

## 📂 Project Structure

* `LLM_Pretrain_SFT.ipynb`: The main notebook containing the entire code pipeline (Data Collection -> Tokenization -> Pre-training -> SFT).

## 🚀 Key Features Implemented

### 1. Data Engineering
* Recursive collection of raw text files.
* Cleaning pipeline: Deduplication, Regex-based normalization, and non-Cyrillic filtering.
* **Custom BPE Tokenizer:** Trained a 25k-vocabulary tokenizer optimized for Russian morphology.

### 2. Model Training
* **Stage 1 (Pre-training):** Training a Transformer model from scratch (or adapting a base model) on the literature corpus using Causal Language Modeling objective.
* **Stage 2 (SFT):** Fine-tuning `Qwen2.5-0.5B` using **TRL (Transformer Reinforcement Learning)** and the Alpaca dataset.
* **Optimization:** Used **BF16 precision** for memory efficiency.

### 3. Monitoring
* Implemented a custom `GenerationCallback` to visually monitor the model's text generation quality at the end of each epoch, instead of relying solely on Loss metrics.


## 📊 Results & Analysis

Since this is an educational project using a lightweight model (`Qwen2.5-0.5B`) and limited compute budget, the goal was to validate the training pipeline rather than achieve State-of-the-Art performance.

### 1. Training Dynamics
* **Pre-training:** The model demonstrated steady loss convergence, successfully learning the **syntactic and morphological structure** of the Russian language (correct Cyrillic usage, punctuation), although semantic coherence remains limited due to the dataset size.
* **SFT (Instruction Tuning):** A clear behavioral shift was observed by Epoch 3. The model transitioned from "text continuation" (LM objective) to "dialogue format" (Instruction following), although it is still prone to hallucinations — a known limitation of <1B parameter models.

### 2. Qualitative Comparison

| Evaluation Prompt | Base Model (Pre-trained, Epoch 3) | SFT Model Behavior (Epoch 3) |
| :--- | :--- | :--- |
| **User:** "Что бы ни случилось, я всегда буду" | *Generates syntactically correct but irrelevant text:* "Что бы ни случилось, я всегда будупегали, что мне все-таки они, я все-таки не могу.
" | *Attempts to structure an answer:* "Нужно читать книги и говорить..." (Basic instruction following) |
| **User:** "Чтобы жить честно" | *Hallucinations / irrelevant text:* "Чтобы жить честновпим-с." | *Generates rhymed lines (even if semantically nonsensical).* |

### 3. Limitations & Future Work
To achieve production-level quality, the following improvements would be required:
* **Scale:** Scaling up to a 7B+ parameter model (e.g., Mistral or Llama 3).
* **Data:** Increasing the pre-training corpus from a few novels to a generic web corpus (e.g., CommonCrawl).
* **Compute:** Extending training from 3 epochs to multiple epochs with a larger batch size.
