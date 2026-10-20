<div align="center">

# Fine-Tuning Large Language Models

**A complete, reproducible reference for fine-tuning transformer models — from first principles to a deployed application.**

Nine end-to-end projects covering full fine-tuning, LoRA, QLoRA, Unsloth-accelerated training, quantization internals, and low-resource (Bangla) language modelling. Every notebook trains on a single 16 GB NVIDIA T4, reports measured before/after metrics, and ships its weights to the Hugging Face Hub.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.x%20%7C%205.x-FFD21E)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20QLoRA-FFD21E)](https://huggingface.co/docs/peft)
[![Unsloth](https://img.shields.io/badge/Unsloth-2x%20faster-00C853)](https://github.com/unslothai/unsloth)
[![Hugging Face](https://img.shields.io/badge/Hub-emonHasan01-FFD21E)](https://huggingface.co/emonHasan01)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Results at a Glance](#results-at-a-glance)
- [Repository Structure](#repository-structure)
- [Project Catalogue](#project-catalogue)
- [Techniques Covered](#techniques-covered)
- [Choosing a Method](#choosing-a-method)
- [Getting Started](#getting-started)
- [Running the Demo Application](#running-the-demo-application)
- [Hardware Requirements](#hardware-requirements)
- [Published Models](#published-models)
- [Deployment Formats](#deployment-formats)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Overview

Fine-tuning turns a general-purpose pretrained model into a specialist. This repository is a practical curriculum for doing that well: each directory is a self-contained project that takes a published base model, a public dataset, and a constrained GPU budget, then measures exactly what the training bought.

The projects move deliberately from the bottom up:

1. **Fundamentals** — what quantization actually does to a weight tensor, implemented from scratch.
2. **Parameter-efficient fine-tuning** — LoRA and QLoRA adapters on encoder-decoder and decoder-only models.
3. **Full fine-tuning** — every parameter updated, on both a classifier and a 1B generative model.
4. **Accelerated training** — the same work through Unsloth's fused kernels on a single commodity GPU.
5. **Low-resource languages** — adapting English-centric models to Bangla.
6. **Deployment** — a trained adapter served behind a Streamlit interface.

Every notebook is written to be *run*, not just read: dependency pinning, GPU checks, version-robust `TrainingArguments` handling for both Transformers v4 and v5, baseline evaluation before training, and a generated model card at the end.

---

## Results at a Glance

All figures below are measured on held-out splits and taken directly from the notebook outputs in this repository.

| Project | Task | Metric | Base | Fine-tuned | Delta |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-3B Text-to-SQL** | NL → SQL generation | Exact match | 7.7% | **74.3%** | **+66.6** |
| | | Token F1 | 83.1 | **97.5** | +14.4 |
| **Gemma-3-1B Paraphrase** | Paraphrase generation | iBLEU | 1.65 | **17.99** | **+16.34** |
| | | Copy rate | 41.7% | **0.7%** | −41.0 |
| **Qwen2.5-1.5B DialogSum** | Dialogue summarization | ROUGE-1 | 0.2822 | **0.4185** | +0.1363 |
| | | ROUGE-Lsum | 0.2424 | **0.3624** | +0.1200 |
| **Marian EN→ES** | Machine translation | sacreBLEU | 17.09 | **19.99** | +2.90 |
| **Bangla Summarizer** | Bangla abstractive summarization | chrF++ | 22.78 | **24.91** | +2.13 |
| | | ROUGE-L | 8.23 | **11.43** | +3.20 |
| **RoBERTa LIAR** | Fake-news classification | Macro-F1 | 0.6025 <sup>1</sup> | **0.6351** | +0.0326 |
| | | Accuracy | 0.6415 <sup>2</sup> | **0.6446** | +0.0031 |
| **Bangla Gemma** | Bangla language modelling | Held-out perplexity | 24.30 | **12.70** | **−48%** |
| **Llama-3.2-3B Reflect** | Chain-of-thought reasoning | GSM8K accuracy | 75.0% | 65.0% | **−10.0** <sup>3</sup> |
| **Bangla GPT (Qwen3.5-0.8B)** | Bangla general ability | MMLU-bn + PIQA-bn mean | 0.392 | 0.385 | **−0.007** <sup>3</sup> |

---

## Repository Structure

```
Fine-Tuning/
├── Quantization Fundamentals/            # Int8 and NF4 quantization implemented from scratch
│   └── Quantization Basics.ipynb
│
├── PEFT with LoRA Fine Tuning/           # LoRA adapters on a decoder-only model
│   └── qwen_lora_dialogsum.ipynb
│
├── PEFT with QLoRA Fine Tuning/          # Quantized base + LoRA on a seq2seq model
│   └── opus_mt_en_es_lora.ipynb
│
├── Full Supervised Fine Tuning/          # All-parameter fine-tuning of an encoder classifier
│   └── liar_roberta_colab.ipynb
│
├── LoRA Fine Tuning with Unsloth AI/     # Unsloth-accelerated LoRA on 3B models
│   ├── qwen25_3b_text2sql.ipynb
│   └── unsloth-llama32-reflect-kaggle.ipynb
│
├── Full Fine Tuning with Unsloth AI/     # Full fine-tune of a 1B model on one 16 GB GPU
│   └── gemma3_1b_paraphrase_full_finetune_T4(1).ipynb
│
├── Bangla Language Models/               # Low-resource language adaptation
│   ├── bangla_gemma_pro.ipynb
│   ├── bangla_gpt-pro.ipynb
│   └── bangla_summarizer_pro.ipynb
│
├── End to End Project with Fine Tuning/  # Train → save → serve
│   ├── experiment.ipynb                  #   T5-small + LoRA on CNN/DailyMail
│   ├── app.py                            #   Streamlit inference interface
│   ├── app.png                           #   Application screenshot
│   └── t5-small-summarizer-lora/         #   Trained adapter weights
│
├── LICENSE
└── README.md
```

---

## Project Catalogue

### Quantization Fundamentals

> [`Quantization Fundamentals/Quantization Basics.ipynb`](Quantization%20Fundamentals/Quantization%20Basics.ipynb)

Quantization implemented by hand before any library is imported. Covers symmetric int8 quantization — deriving the scale and zero point, mapping float weights to integers, clipping to the valid range, and dequantizing back — then reconstructs the **NF4** (4-bit NormalFloat) scheme used by `bitsandbytes`, mapping a simulated Gaussian weight distribution onto 16 representative values. Includes plots of original versus reconstructed weights so the quantization error is visible rather than asserted.

**Read this first.** It is the prerequisite for understanding why QLoRA works.

---

### PEFT with LoRA — Dialogue Summarization

> [`PEFT with LoRA Fine Tuning/qwen_lora_dialogsum.ipynb`](PEFT%20with%20LoRA%20Fine%20Tuning/qwen_lora_dialogsum.ipynb)

| | |
| --- | --- |
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Dataset** | `knkarthick/dialogsum` |
| **Method** | LoRA — 18.46M trainable parameters (**1.20%** of the base) |
| **Result** | ROUGE-1 **0.2822 → 0.4185** · ROUGE-Lsum **0.2424 → 0.3624** |
| **Hub** | [`emonHasan01/QwenSum-LoRA`](https://huggingface.co/emonHasan01/QwenSum-LoRA) |

A careful LoRA reference implementation. Prompt tokens are masked from the loss so gradients are computed on roughly 13% of each sequence — the completion only. The zero-shot baseline is evaluated two ways (full output, and first sentence only) to rule out length artefacts inflating the measured gain, and a length-controlled attribution check runs before any credit is claimed for the adapter. Ends with adapter export, a generated model card, and a Hub push.

**Companion model — the same task at 3B.** [`qwen2.5-3b-dialogsum-qlora`](https://huggingface.co/emonHasan01/qwen2.5-3b-dialogsum-qlora) applies QLoRA to `Qwen/Qwen2.5-3B-Instruct` on the same DialogSum split, reaching **ROUGE-1 44.69** and **ROUGE-L 36.62** against the 1.5B LoRA run's 41.85 / 36.24. Doubling the base size buys roughly three ROUGE-1 points while a 4-bit base keeps the memory cost at or below the 1.5B run. The two figures come from separate evaluation harnesses, so treat the gap as indicative rather than exact. Training for this variant was done outside this repository; only the published adapter is available.

---

### PEFT with QLoRA — English → Spanish Translation

> [`PEFT with QLoRA Fine Tuning/opus_mt_en_es_lora.ipynb`](PEFT%20with%20QLoRA%20Fine%20Tuning/opus_mt_en_es_lora.ipynb)

| | |
| --- | --- |
| **Base model** | `Helsinki-NLP/opus-mt-en-es` (Marian) |
| **Dataset** | `Helsinki-NLP/opus_books` (en–es) |
| **Method** | Quantized base + LoRA adapters on a seq2seq architecture |
| **Result** | **17.09 → 19.99 sacreBLEU** on 1,000 held-out pairs |
| **Hub** | [`emonHasan01/omt-en-es-books`](https://huggingface.co/emonHasan01/omt-en-es-books) · [ONNX](https://huggingface.co/emonHasan01/omt-en-es-books-onnx) · [Space](https://huggingface.co/spaces/emonHasan01/english-into-spanish) |

Adapting an already-strong specialist model to a new domain (literary prose). Demonstrates the seq2seq label-handling pitfall explicitly — a verification cell decodes a label back and asserts it is valid Spanish rather than mangled token ids — tracks BLEU per epoch against the frozen baseline (19.23 → 19.89 → 19.99), and includes side-by-side translations plus a from-scratch check that the uploaded artefact loads and runs.

---

### Full Supervised Fine-Tuning — Fake News Detection

> [`Full Supervised Fine Tuning/liar_roberta_colab.ipynb`](Full%20Supervised%20Fine%20Tuning/liar_roberta_colab.ipynb)

| | |
| --- | --- |
| **Base model** | `roberta-base` (all parameters updated) |
| **Dataset** | LIAR (political fact-checking statements) |
| **Result** | **64.46% accuracy · 0.6351 macro-F1** on the test split |
| **Baselines** | TF-IDF + LogReg 0.6025 macro-F1 · Majority class 0.3908 |
| **Hub** | [`emonHasan01/liar-roberta`](https://huggingface.co/emonHasan01/liar-roberta) · [Space](https://huggingface.co/spaces/emonHasan01/liar-roberta) |

The most rigorously evaluated project here, and a deliberate lesson in modest gains. A full fine-tune beats a 1.8-second TF-IDF baseline by **+0.0326 macro-F1** — real, reproducible, and far smaller than the headline numbers tutorials usually promise. Includes multi-seed training (mean ± std), a confusion matrix, per-class accuracy, a decision-threshold sweep with precision/recall/flagged-rate trade-offs, calibration and error analysis, and an ONNX export for in-browser inference.

---

### LoRA with Unsloth — Text-to-SQL

> [`LoRA Fine Tuning with Unsloth AI/qwen25_3b_text2sql.ipynb`](LoRA%20Fine%20Tuning%20with%20Unsloth%20AI/qwen25_3b_text2sql.ipynb)

| | |
| --- | --- |
| **Base model** | `unsloth/Qwen2.5-3B-Instruct` |
| **Dataset** | `b-mc2/sql-create-context` |
| **Method** | 16-bit LoRA via Unsloth |
| **Result** | Exact match **7.7% → 74.3%** · Token F1 **83.1 → 97.5** · Valid SQL **98.7% → 100%** |
| **Hub** | [`emonHasan01/qwen2.5-3b-text2sql`](https://huggingface.co/emonHasan01/qwen2.5-3b-text2sql) |

The clearest demonstration of what fine-tuning is *for*. The base model already writes syntactically valid SQL 98.7% of the time but matches the reference query only 7.7% of the time — it knows the language, not the conventions. LoRA closes that gap almost entirely. An intermediate checkpoint is evaluated as well (step 150 already reaches 69.0% exact match) to show where the curve begins to flatten.

---

### LoRA with Unsloth — Reasoning Style Transfer

> [`LoRA Fine Tuning with Unsloth AI/unsloth-llama32-reflect-kaggle.ipynb`](LoRA%20Fine%20Tuning%20with%20Unsloth%20AI/unsloth-llama32-reflect-kaggle.ipynb)

| | |
| --- | --- |
| **Base model** | `unsloth/Llama-3.2-3B-Instruct` |
| **Dataset** | `ServiceNow-AI/R1-Distill-SFT` (distilled reasoning traces) |
| **Evaluation** | `openai/gsm8k` — 100 problems, greedy decoding, identical prompts |
| **Result** | **75.0% → 65.0% accuracy (−10.0)** |
| **Hub** | [`emonHasan01/llama-3.2-3b-reflect`](https://huggingface.co/emonHasan01/llama-3.2-3b-reflect) · [GGUF](https://huggingface.co/emonHasan01/llama-3.2-3b-reflect-GGUF) |

**A documented failure, kept on purpose.** Training on distilled reasoning traces successfully transferred the *style* of extended reasoning — the model reasons at visibly greater length — while measurably degrading the *substance*. GSM8K accuracy falls by ten points. The notebook includes per-problem comparison charts and a style analysis isolating what the adapter actually changed. It also covers the full deployment path: merged 16-bit weights, adapter-only export, and a GGUF build for `llama.cpp`.

---

### Full Fine-Tuning with Unsloth — Paraphrase Generation

> [`Full Fine Tuning with Unsloth AI/gemma3_1b_paraphrase_full_finetune_T4(1).ipynb`](Full%20Fine%20Tuning%20with%20Unsloth%20AI/gemma3_1b_paraphrase_full_finetune_T4%281%29.ipynb)

| | |
| --- | --- |
| **Base model** | `unsloth/gemma-3-1b-it` (**every parameter updated**) |
| **Datasets** | `google-research-datasets/paws` + `humarin/chatgpt-paraphrases` |
| **Result** | iBLEU **1.65 → 17.99** · Copy rate **41.7% → 0.7%** · Similarity held at 0.91 |
| **Hub** | [`emonHasan01/gemma-3-1b-paraphrase`](https://huggingface.co/emonHasan01/gemma-3-1b-paraphrase) |

Full fine-tuning of a 1B model inside a single 16 GB GPU's memory budget — no adapters, no quantized base. The evaluation is the interesting part: paraphrasing has a trivial degenerate solution (copy the input), and the base model takes it 41.7% of the time. The notebook scores semantic similarity *and* lexical diversity jointly via iBLEU, and plots the trade-off curve that makes the task legible. Fine-tuning raises diversity from 24.6 to 54.9 while holding similarity constant, eliminating copying almost entirely. Handles Gemma 3's template quirk (no system role) correctly.

---

### Bangla Language Models

> [`Bangla Language Models/`](Bangla%20Language%20Models/)

Adapting English-centric models to a low-resource language, with the tokenizer penalty measured rather than assumed — Bangla costs **4.13 tokens per word** on a Qwen tokenizer, which materially changes the compute budget for a fixed context length.

| Notebook | Base model | Dataset | Outcome |
| --- | --- | --- | --- |
| `bangla_summarizer_pro.ipynb` | `unsloth/gemma-3-1b-it` | `csebuetnlp/xlsum` (bn) | chrF++ **22.78 → 24.91** · ROUGE-L **8.23 → 11.43** · output length 46.2 → 34.4 words |
| `bangla_gpt-pro.ipynb` | `unsloth/Qwen3.5-0.8B` | `md-nishat-008/Bangla-Instruct` | Benchmark mean **0.392 → 0.385** — see note below |
| `bangla_gemma_pro.ipynb` | `unsloth/gemma-3-1b-it` | `md-nishat-008/Bangla-Instruct` | Perplexity **24.30 → 12.70** (−48%) · benchmark mean 0.458 → 0.457 |

The summarizer is also benchmarked against a **lead-3 extractive baseline** (chrF++ 26.02) which the fine-tuned model does not beat — an honest result for a 1B model on Bangla news, and a reminder that extractive baselines are hard to beat on news summarization. The ROUGE pipeline is validated for Bangla script before use (identical strings must score 1.00, which naive tokenizers fail).

`bangla-gemma-pro` is the clearest illustration of why a single metric is not enough. Trained on the 32,037 Bangla-Instruct pairs that fit in 512 tokens, it **halves held-out perplexity (24.30 → 12.70)** — the model is unambiguously better at producing Bangla — while the three multiple-choice benchmarks that loaded move by less than the noise floor (mean 0.458 → 0.457). Fluency improved; measurable knowledge did not. At 300 items per benchmark and scores near chance, those accuracy numbers cannot resolve the difference, and the card says so.

**Hub — models:** [`bangla-gemma-pro`](https://huggingface.co/emonHasan01/bangla-gemma-pro) · [`bangla-gemma-pro-ONNX`](https://huggingface.co/emonHasan01/bangla-gemma-pro-ONNX) · [`bangla-gpt-pro`](https://huggingface.co/emonHasan01/bangla-gpt-pro) · [`bangla-gpt-pro-GGUF`](https://huggingface.co/emonHasan01/bangla-gpt-pro-GGUF) · [`bangla-summarizer-pro`](https://huggingface.co/emonHasan01/bangla-summarizer-pro) · [`bangla-summarizer-pro-GGUF`](https://huggingface.co/emonHasan01/bangla-summarizer-pro-GGUF)

**Hub — demos:** [`bangla-gemma-pro-demo`](https://huggingface.co/spaces/emonHasan01/bangla-gemma-pro-demo) · [`bangla-summarizer-pro-demo`](https://huggingface.co/spaces/emonHasan01/bangla-summarizer-pro-demo) · [`bangla-gpt-pro-results`](https://huggingface.co/spaces/emonHasan01/bangla-gpt-pro-results)

---

### End-to-End Project — Summarization Service

> [`End to End Project with Fine Tuning/`](End%20to%20End%20Project%20with%20Fine%20Tuning/)

<div align="center">
<img src="End%20to%20End%20Project%20with%20Fine%20Tuning/app.png" alt="Streamlit summarization application" width="720">
</div>

| | |
| --- | --- |
| **Base model** | `t5-small` |
| **Dataset** | `cnn_dailymail` |
| **Adapter** | LoRA — `r=8`, `lora_alpha=32`, `lora_dropout=0.1`, targets `q` and `v`, task `SEQ_2_SEQ_LM` |
| **Interface** | Streamlit (`app.py`) with beam-search decoding (`num_beams=4`) |

The complete lifecycle in one directory: dataset loading and preprocessing, LoRA configuration, training through the Trainer API, checkpointing, adapter export, and a working user-facing application. The trained adapter is committed to the repository, so the app runs without retraining.

**Scope note.** This project demonstrates the *pipeline*, not a quality result. It trains for one epoch on a 1% subset (2,871 examples, 294,912 trainable parameters — 0.49% of the base) and reaches a training loss of 1.25, but no baseline was evaluated and no ROUGE was computed, so there is no measured claim that the adapter improves on plain `t5-small`. Treat it as a working template to adapt, and run the evaluation yourself before trusting the output.

---

## Techniques Covered

| Technique | Trainable params | GPU memory | Best for |
| --- | --- | --- | --- |
| **Full fine-tuning** | 100% | Highest | Small models (≤1B), classifiers, maximum adaptation |
| **LoRA** | ~0.1–2% | Moderate | The default choice — most of the gain at a fraction of the cost |
| **QLoRA** | ~0.1–2% | Lowest | Large models on consumer GPUs; 4-bit frozen base + adapters |
| **Unsloth** | Either | Reduced | Roughly 2× faster training and lower memory via fused kernels |
| **Quantization** | — | — | Inference-time compression; the foundation of QLoRA |

---

## Choosing a Method

```
Does the base model already perform the task acceptably?
├── Yes → Do not fine-tune. Improve the prompt first. It is free.
└── No
    ├── Is this a classification task with a model ≤ 350M params?
    │   └── Full fine-tuning (RoBERTa LIAR) — cheap at this scale, and it wins.
    │
    ├── Will the model fit in your GPU in 16-bit with room for gradients?
    │   ├── Yes → LoRA (DialogSum, Text-to-SQL) — the default. Start at r=8 or r=16.
    │   └── No  → QLoRA (Marian EN→ES) — 4-bit frozen base, adapters in 16-bit.
    │
    └── Need maximum adaptation on a ≤1B model with one 16 GB GPU?
        └── Full fine-tuning via Unsloth (Gemma-3-1B paraphrase).
```

**Before you train anything:** evaluate the base model on your held-out split and record the number. Without it you cannot tell improvement from noise — and, as the reasoning and Bangla runs below show, you cannot tell improvement from regression either.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- A CUDA-capable GPU with 16 GB of VRAM (NVIDIA T4, P100, RTX 3090/4090, A10 or better)
- A [Hugging Face account](https://huggingface.co/join) and access token, if you intend to push models

### Installation

```bash
git clone https://github.com/Md-Emon-Hasan/Fine-Tuning.git
cd Fine-Tuning

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Core stack:

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes evaluate
pip install jupyter streamlit sacrebleu rouge-score scikit-learn matplotlib
```

For the Unsloth-accelerated projects (5, 6, 7, 8):

```bash
pip install unsloth
```

> **Note.** Each notebook installs its own pinned dependencies in its first cell, so it runs end to end in any clean CUDA environment without the setup above. The notebooks introspect whether `TrainingArguments` comes from Transformers v4 or v5 and adapt their arguments accordingly, rather than pinning you to one major version.

### Running a Notebook

```bash
jupyter notebook
```

Open any project directory and run the cells top to bottom. Recommended order for a first pass:

1. `Quantization Fundamentals/` — build the mental model
2. `PEFT with LoRA Fine Tuning/` — the canonical LoRA workflow
3. `PEFT with QLoRA Fine Tuning/` — add quantization to it
4. `Full Supervised Fine Tuning/` — rigorous evaluation practice
5. `LoRA Fine Tuning with Unsloth AI/` — scale and speed
6. `End to End Project with Fine Tuning/` — ship it

To push your own trained models, authenticate first:

```bash
huggingface-cli login
```

---

## Running the Demo Application

The summarization service runs against the adapter committed in this repository.

```bash
cd "End to End Project with Fine Tuning"
pip install streamlit transformers torch peft
streamlit run app.py
```

Then open <http://localhost:8501>.

> **Before the first run:** [`app.py`](End%20to%20End%20Project%20with%20Fine%20Tuning/app.py) currently points `MODEL_PATH` at an absolute path from the original development machine. Change it to the adapter directory in this repository:
>
> ```python
> MODEL_PATH = "t5-small-summarizer-lora"
> ```

---

## Hardware Requirements

Every notebook in this repository, with the base model it trains and the hardware it was verified on.

| Notebook | Base model | Method | Minimum VRAM | Trained on |
| --- | --- | --- | --- | --- |
| `Quantization Basics.ipynb` | — | Quantization from scratch | None (CPU) | Any |
| `experiment.ipynb` | `t5-small` | LoRA | 8 GB | NVIDIA Tesla T4 |
| `opus_mt_en_es_lora.ipynb` | `Helsinki-NLP/opus-mt-en-es` | QLoRA | 8 GB | NVIDIA Tesla T4 |
| `qwen_lora_dialogsum.ipynb` | `Qwen/Qwen2.5-1.5B-Instruct` | LoRA | 16 GB | NVIDIA Tesla T4 |
| `liar_roberta_colab.ipynb` | `roberta-base` | Full fine-tuning | 16 GB | NVIDIA Tesla T4 |
| `qwen25_3b_text2sql.ipynb` | `unsloth/Qwen2.5-3B-Instruct` | LoRA (Unsloth) | 16 GB | NVIDIA Tesla T4 |
| `unsloth-llama32-reflect-kaggle.ipynb` | `unsloth/Llama-3.2-3B-Instruct` | LoRA (Unsloth) | 16 GB | NVIDIA Tesla T4 |
| `gemma3_1b_paraphrase_full_finetune_T4(1).ipynb` | `unsloth/gemma-3-1b-it` | Full fine-tuning (Unsloth) | 16 GB | NVIDIA Tesla T4 |
| `bangla_gemma_pro.ipynb` | `unsloth/gemma-3-1b-it` | Full fine-tuning (Unsloth) | 16 GB | 2 × NVIDIA Tesla T4 |
| `bangla_gpt-pro.ipynb` | `unsloth/Qwen3.5-0.8B` | Full fine-tuning (Unsloth) | 16 GB | NVIDIA Tesla T4 |
| `bangla_summarizer_pro.ipynb` | `unsloth/gemma-3-1b-it` | Full fine-tuning (Unsloth) | 16 GB | NVIDIA Tesla T4 |

A single 16 GB card runs everything here. The two 8 GB entries fit comfortably on a consumer GPU such as an RTX 3060 or 4060.

---

## Published Models

All trained artefacts are available on the Hugging Face Hub under [**@emonHasan01**](https://huggingface.co/emonHasan01).

| Model | Task | Formats |
| --- | --- | --- |
### Models

| Model | Task | Additional formats |
| --- | --- | --- |
| [`qwen2.5-3b-text2sql`](https://huggingface.co/emonHasan01/qwen2.5-3b-text2sql) | Natural language → SQL | Adapter, merged 16-bit |
| [`gemma-3-1b-paraphrase`](https://huggingface.co/emonHasan01/gemma-3-1b-paraphrase) | Paraphrase generation | Full weights |
| [`QwenSum-LoRA`](https://huggingface.co/emonHasan01/QwenSum-LoRA) | Dialogue summarization (1.5B) | LoRA adapter |
| [`qwen2.5-3b-dialogsum-qlora`](https://huggingface.co/emonHasan01/qwen2.5-3b-dialogsum-qlora) <sup>†</sup> | Dialogue summarization (3B) | QLoRA adapter, 4-bit |
| [`omt-en-es-books`](https://huggingface.co/emonHasan01/omt-en-es-books) | EN→ES literary translation | LoRA adapter |
| [`omt-en-es-books-onnx`](https://huggingface.co/emonHasan01/omt-en-es-books-onnx) | EN→ES literary translation | ONNX |
| [`liar-roberta`](https://huggingface.co/emonHasan01/liar-roberta) | Fake-news classification | Full weights, ONNX |
| [`llama-3.2-3b-reflect`](https://huggingface.co/emonHasan01/llama-3.2-3b-reflect) | Extended reasoning style | Adapter, merged 16-bit |
| [`llama-3.2-3b-reflect-GGUF`](https://huggingface.co/emonHasan01/llama-3.2-3b-reflect-GGUF) | Extended reasoning style | GGUF |
| [`bangla-gemma-pro`](https://huggingface.co/emonHasan01/bangla-gemma-pro) | Bangla instruction following (Gemma) | Full weights |
| [`bangla-gemma-pro-ONNX`](https://huggingface.co/emonHasan01/bangla-gemma-pro-ONNX) | Bangla instruction following (Gemma) | ONNX |
| [`bangla-gpt-pro`](https://huggingface.co/emonHasan01/bangla-gpt-pro) | Bangla instruction following (Qwen) | Full weights |
| [`bangla-gpt-pro-GGUF`](https://huggingface.co/emonHasan01/bangla-gpt-pro-GGUF) | Bangla instruction following (Qwen) | GGUF |
| [`bangla-summarizer-pro`](https://huggingface.co/emonHasan01/bangla-summarizer-pro) | Bangla summarization | Full weights |
| [`bangla-summarizer-pro-GGUF`](https://huggingface.co/emonHasan01/bangla-summarizer-pro-GGUF) | Bangla summarization | GGUF |

<sup>†</sup> A companion QLoRA run on `Qwen/Qwen2.5-3B-Instruct` for the same DialogSum task as the LoRA run above, trained outside this repository. Reports ROUGE-1 44.69 and ROUGE-L 36.62 on the DialogSum test split (self-reported).

### Interactive Demos

| Space | What it does |
| --- | --- |
| [`liar-roberta`](https://huggingface.co/spaces/emonHasan01/liar-roberta) | Paste a political claim, get a truthfulness prediction |
| [`english-into-spanish`](https://huggingface.co/spaces/emonHasan01/english-into-spanish) | Live EN→ES translation with the tuned Marian adapter |
| [`bangla-gemma-pro-demo`](https://huggingface.co/spaces/emonHasan01/bangla-gemma-pro-demo) | Bangla instruction following, interactive |
| [`bangla-summarizer-pro-demo`](https://huggingface.co/spaces/emonHasan01/bangla-summarizer-pro-demo) | Bangla news summarization, interactive |
| [`bangla-gpt-pro-results`](https://huggingface.co/spaces/emonHasan01/bangla-gpt-pro-results) | Published benchmark results page for the Qwen Bangla run |

---

## Deployment Formats

The projects cover five distinct paths from a trained model to a served one:

| Format | Use case |
| --- | --- |
| **LoRA adapter** | Tens of MB; swap task behaviour at runtime on a shared base |
| **Merged 16-bit weights** | Standard `from_pretrained` loading, no PEFT dependency |
| **GGUF** | CPU and edge inference via `llama.cpp` / Ollama |
| **ONNX** | Cross-runtime and in-browser inference |
| **Hosted Space** | Interactive Gradio demos, linked above |

---

## Contributing

Contributions are welcome. Issues and pull requests are both useful — particularly reproductions on different hardware, corrections to reported metrics, and additional baselines.

1. Fork the repository and create a feature branch.
2. Keep notebooks self-contained: pinned installs, a GPU check, a baseline evaluation, and a results summary.
3. Clear large outputs before committing; keep charts and metric tables.
4. Open a pull request describing what was measured and on what hardware.

---

## License

Released under the [MIT License](LICENSE). Note that the base models and datasets used here carry their own licences — Gemma, Llama, and Qwen in particular have terms worth reviewing before commercial use.

---

## Author

**Md Emon Hasan** — Machine Learning Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-emonlabs--ai-4C1D95?logo=googlechrome&logoColor=white)](https://emonlabs-ai.hitechparks.com/)
[![Email](https://img.shields.io/badge/Email-emon.mlengineer%40gmail.com-EA4335?logo=gmail&logoColor=white)](mailto:emon.mlengineer@gmail.com)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-%2B8801834363533-25D366?logo=whatsapp&logoColor=white)](https://wa.me/8801834363533)
[![GitHub](https://img.shields.io/badge/GitHub-Md--Emon--Hasan-181717?logo=github&logoColor=white)](https://github.com/Md-Emon-Hasan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Md%20Emon%20Hasan-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-emon-hasan-695483237/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-emonHasan01-FFD21E)](https://huggingface.co/emonHasan01)
[![Facebook](https://img.shields.io/badge/Facebook-Md%20Emon%20Hasan-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/mdemon.hasan2001/)

Open to collaboration on applied LLM work — fine-tuning, evaluation, and deployment.

---

## Acknowledgments

- **[Hugging Face](https://huggingface.co/)** — Transformers, PEFT, Datasets, TRL, Accelerate, and the Hub
- **[Unsloth](https://github.com/unslothai/unsloth)** — fused kernels that make 3B fine-tuning viable on a single 16 GB GPU
- **[Dettmers et al.](https://arxiv.org/abs/2305.14314)** — QLoRA and `bitsandbytes`
- **[Hu et al.](https://arxiv.org/abs/2106.09685)** — LoRA
- **[CSE BUET NLP](https://huggingface.co/csebuetnlp)** — XL-Sum, which makes Bangla summarization research possible
- Dataset authors: `b-mc2`, `knkarthick`, `humarin`, `md-nishat-008`, Helsinki-NLP, ServiceNow AI, and the LIAR and GSM8K teams

---

## Citation

```bibtex
@misc{hasan2025finetuning,
  author       = {Md Emon Hasan},
  title        = {Fine-Tuning: End-to-End Reference Implementations for
                  Parameter-Efficient and Full Fine-Tuning of Language Models},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Md-Emon-Hasan/Fine-Tuning}}
}
```

---

<div align="center">

**If this repository helped you, consider leaving a star.**

</div>
