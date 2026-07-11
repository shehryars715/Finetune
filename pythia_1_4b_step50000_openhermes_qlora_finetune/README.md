# Pythia-1.4B step50000 OpenHermes 100k QLoRA fine-tune

This folder contains the practical Kaggle version of the bigger-model experiment.

## What it trains

- Base model: `EleutherAI/pythia-1.4b`
- Revision: `step50000`
- Dataset: `teknium/OpenHermes-2.5`
- Training examples: `100,000`
- Validation examples: `1,000`
- Method: QLoRA / LoRA adapter training
- Default HF repo: `shehryars715/pythia-1.4b-step50000-openhermes-100k-qlora`

## Why QLoRA

The full fine-tune version of 1.4B does not fit on a normal Kaggle 15GB GPU. QLoRA keeps the base model in 4-bit precision and trains adapter weights, which is the realistic path for this hardware.

## Kaggle advice

Use a T4 or A100 if possible. If Kaggle gives a P100/K80 and you see `CUDA error: no kernel image is available for execution on the device`, switch GPU type or restart the session until you get T4/A100.

The notebook defaults to adapter-only upload:

```python
MERGE_AND_PUSH_FULL_MODEL = False
```

Keep that setting for Kaggle. You can merge the adapter into the base model later on a stronger machine.
