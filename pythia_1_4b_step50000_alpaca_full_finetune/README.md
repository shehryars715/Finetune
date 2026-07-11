# Pythia-1.4B step50000 full Alpaca fine-tune

This folder contains a Kaggle-ready notebook for full-parameter supervised fine-tuning of `EleutherAI/pythia-1.4b` at revision `step50000` on the full usable `yahma/alpaca-cleaned` dataset.

This is the direct larger-model version of the earlier `pythia-410m step50000 + Alpaca` experiment. It keeps the dataset and checkpoint stage comparable, while increasing the model size from 410M to 1.4B parameters.

## Files

- `pythia_1_4b_step50000_alpaca_full_finetune.ipynb` — Kaggle-ready full fine-tuning notebook.

## Training setup

- Base model: `EleutherAI/pythia-1.4b`
- Revision: `step50000`
- Dataset: `yahma/alpaca-cleaned`
- Training data: all usable Alpaca rows after holding out validation examples
- Fine-tuning method: full-parameter SFT, not LoRA/QLoRA
- HF push target: `shehryars715/pythia-1.4b-step50000-alpaca`

## Important Kaggle note

This is much heavier than the 410M run. The notebook uses conservative settings:

- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=16`
- `optim="adamw_torch"`
- `MAX_LENGTH=384`

If you hit CUDA out-of-memory, reduce `MAX_LENGTH` to `256`. The notebook uses `adamw_torch` instead of bitsandbytes because some Kaggle GPU/runtime combinations throw `no kernel image is available` with bitsandbytes optimizers. If it still fails, the practical fallback is the QLoRA version.
