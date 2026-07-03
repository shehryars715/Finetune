# Pythia-410M step50000 full Alpaca fine-tune

This folder contains a Kaggle-ready notebook for full-parameter supervised fine-tuning of EleutherAI/pythia-410m at revision step50000.

The notebook loads yahma/alpaca-cleaned, removes rows with empty outputs, reserves 256 shuffled examples for validation, and trains on every remaining usable example. It does not apply the former 10,000-example cap.

## Run on Kaggle

1. Upload pythia_410m_step50000_alpaca_full_finetune.ipynb to Kaggle.
2. Enable a GPU accelerator and Internet access.
3. Run all cells from top to bottom.
4. Download /kaggle/working/pythia-410m-step50000-alpaca-final.zip.

The ZIP contains the complete fine-tuned model and tokenizer, not a LoRA adapter.
