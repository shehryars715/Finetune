# Pythia-410M step50000 full Alpaca fine-tune

This folder contains a Kaggle-ready notebook for full-parameter supervised fine-tuning of EleutherAI/pythia-410m at revision step50000. After training, it automatically pushes the complete model to:

https://huggingface.co/shehryars715/pythia-410m-step50000-alpaca

The notebook loads yahma/alpaca-cleaned, removes rows with empty outputs, reserves 256 shuffled examples for validation, and trains on every remaining usable example. It does not apply the former 10,000-example cap.

## Run on Kaggle

1. Upload pythia_410m_step50000_alpaca_full_finetune.ipynb to Kaggle.
2. Create a Hugging Face access token with write permission.
3. In the Kaggle notebook, open Add-ons -> Secrets and add the token as HF_TOKEN.
4. Enable the HF_TOKEN secret for the notebook.
5. Enable a GPU accelerator and Internet access.
6. Run all cells from top to bottom.
7. The final cell creates the Hugging Face repository if needed and uploads the model automatically.
8. A ZIP backup is also written to /kaggle/working/pythia-410m-step50000-alpaca-final.zip.

The Hugging Face repository and ZIP contain the complete fine-tuned model and tokenizer, not a LoRA adapter. Never paste the token directly into the notebook.

## Model card

The complete Hugging Face model card and its training-curve asset are in
[huggingface_model_card](./huggingface_model_card/README.md). Upload that
README and training_curves.png to the root of the model repository.
