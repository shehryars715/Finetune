# Pythia-410M step5000 Alpaca full fine-tune

This workspace contains a reproducible Kaggle notebook for full-parameter supervised fine-tuning of the early `EleutherAI/pythia-410m` `step5000` checkpoint on 10,000 examples from `yahma/alpaca-cleaned`.

[![Download on Hugging Face](https://img.shields.io/badge/Hugging_Face-Download_model-FFD21E?style=for-the-badge)](https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca)

## Hugging Face model

[View or download the fine-tuned model on Hugging Face](https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca)

## Files

- [`pythia_410m_step5000_alpaca_full_finetune.ipynb`](./pythia_410m_step5000_alpaca_full_finetune.ipynb) — Kaggle-ready training and evaluation notebook.
- [`huggingface_model_card/README.md`](./huggingface_model_card/README.md) — model card to upload at the root of the Hugging Face model repository.
- [`comparison_results.csv`](./comparison_results.csv) — complete deterministic generations before and after fine-tuning.

## Experiment summary

- Base checkpoint: `EleutherAI/pythia-410m`, revision `step5000`
- Dataset: `yahma/alpaca-cleaned`
- Training examples: 10,000
- Held-out validation examples: 256
- Training: one epoch, full-parameter fine-tuning
- Objective: completion-only causal language-modeling loss
- Maximum sequence length: 384 tokens
- Adapter or quantization method: none

The notebook compares fixed validation loss and deterministic responses before and after training. This is an educational experiment using a partially pretrained checkpoint, not a production chatbot.

## Result snapshot

Across the six supplied comparison prompts, the untouched checkpoint repeatedly echoed prompt-template markers. The fine-tuned checkpoint produced answer-shaped text for all six prompts and passed one of four transparent surface-rule checks. It learned recognizable instruction-response behavior, but translation, arithmetic, code correctness, and factual explanation remained weak. See the full [comparison CSV](./comparison_results.csv) and the more readable analysis in the [model card](./huggingface_model_card/README.md).
