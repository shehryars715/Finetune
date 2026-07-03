---
base_model: EleutherAI/pythia-410m
base_model_relation: finetune
datasets:
- yahma/alpaca-cleaned
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- pythia
- alpaca
- instruction-tuning
- full-finetune
---

# Pythia-410M step5000 Alpaca full fine-tune

<p align="center">
  <a href="https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca/tree/main"><img src="https://img.shields.io/badge/Hugging_Face-Download_model-FFD21E?style=for-the-badge" alt="Download model"></a>
  <img src="https://img.shields.io/badge/Parameters-410M-6C63FF?style=for-the-badge" alt="410M parameters">
  <img src="https://img.shields.io/badge/Training-Full_fine--tune-2EA44F?style=for-the-badge" alt="Full fine-tune">
</p>

> **An intentionally small, inspectable before/after experiment:** can one epoch of Alpaca SFT turn a very early Pythia checkpoint from prompt-template repetition into a recognizable instruction follower?

This is a full-parameter supervised fine-tune of the early `step5000` revision of [`EleutherAI/pythia-410m`](https://huggingface.co/EleutherAI/pythia-410m). It was trained for one epoch on 10,000 examples from [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned), using a separate fixed set of 256 examples for validation.

This repository contains a complete model checkpoint. It is not a LoRA or PEFT adapter and does not require the original checkpoint to be loaded separately.

**[Download the complete checkpoint](https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca/tree/main)** · **[See all raw before/after generations](https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca/blob/main/comparison_results.csv)**

## Why this checkpoint is interesting

- **A genuine early-checkpoint transformation:** the base `step5000` model mostly loops over prompt markers; after SFT it emits answer-shaped responses.
- **A full model, not an adapter:** useful for direct loading and for studying how every parameter changes during instruction tuning.
- **Small enough to explore:** 410M parameters is approachable for educational inference and analysis.
- **Reproducible:** the training split, validation split, prompt format, hyperparameters, deterministic test prompts, and decoding settings are documented.
- **Transparent about failure:** the included comparisons show improved instruction behavior alongside major factual weaknesses.

## At a glance

| Property | Value |
|---|---|
| Starting point | Pythia-410M at pretraining revision `step5000` |
| Fine-tuning data | 10,000 cleaned Alpaca examples |
| Validation data | 256 fixed held-out examples |
| Update method | Full-parameter SFT, completion-only loss |
| Training duration | 1 epoch |
| Context length | 384 tokens |
| Best use | Education, checkpoint analysis, SFT demonstrations |
| Production ready | **No** |

## Model details

- **Architecture:** GPT-NeoX causal language model
- **Base model:** `EleutherAI/pythia-410m`
- **Base revision:** `step5000`
- **Language:** primarily English
- **Task:** instruction-conditioned text generation
- **Fine-tuning method:** full-parameter supervised fine-tuning
- **Quantization:** none
- **Adapters:** none

The starting revision is only partially pretrained. Alpaca fine-tuning teaches instruction/response structure but does not replace the language pretraining missing from this early checkpoint.

## Training data

The source dataset was filtered to remove examples with empty outputs, shuffled with seed 42, and split as follows:

- First 256 shuffled examples: held-out validation set
- Next 10,000 examples: training set

Examples were converted to the original Alpaca prompt format. The target response was kept as a separate completion, with EOS appended. Loss was calculated only on completion and EOS tokens; prompt tokens supplied context but were masked from the loss.

## Training procedure

| Setting | Value |
|---|---:|
| Epochs | 1 |
| Training examples | 10,000 |
| Validation examples | 256 |
| Maximum sequence length | 384 |
| Per-device batch size | 2 |
| Gradient accumulation | 8 |
| Learning rate | 5e-5 |
| Scheduler | Cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Packing | Disabled |
| Gradient checkpointing | Enabled during training |
| Seed | 42 |

BF16 was selected only on GPUs with compute capability 8.0 or newer; otherwise FP16 mixed precision was used. All model parameters were enabled for training.

## Prompt format

Without an additional input:

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

With an additional input:

```text
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
```

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "shehryars715/pythia-410m-step5000-alpaca"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
).to(device)
model.eval()

instruction = "Explain photosynthesis in exactly two sentences."
prompt = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    f"### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=128,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

new_tokens = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
```

## Evaluation

The training notebook evaluates the untouched checkpoint and the fine-tuned model on the same 256 held-out examples. It also compares deterministic generations on six fixed prompts and includes limited rule-based checks plus a blank human-review table.

### What changed in the six-prompt comparison

| Observation | Before SFT | After SFT |
|---|---:|---:|
| Avoided the prompt-template repetition seen in the base response | 0/6 | 6/6 |
| Passed the notebook's four transparent surface-rule checks | 0/4 | 1/4 |

The useful signal is **behavioral, not capability-level**: the model learned to produce response-shaped text, but most tested answers remain incorrect. The rule checks cover only translation keywords, speed, prime-number logic, and three-item formatting.

### Side-by-side results

| Prompt | `step5000` before SFT | After full fine-tuning |
|---|---|---|
| Photosynthesis in exactly two sentences | Repeats “Write a response…” and `### Response:` | Produces two complete sentences about photosynthesis |
| Translate weather sentence into French | Repeats instruction/response markers | Produces a direct sentence response |
| Average speed: 180 km in 3 hours | Repeats the prompt and markers | Produces a concise numerical response |
| Python prime-checking function | Repeats the prompt and markers | Produces a Python function and explanation |
| Exactly three interview tips | Repeats instruction/response markers | Produces three numbered tips |
| Why leaves appear green | Repeats instruction/response markers | Produces answer-like paragraphs |

<details>
<summary><strong>Example 1: visible instruction-following improvement</strong></summary>

**Prompt**

> Explain photosynthesis in exactly two sentences.

**Before SFT**

> Write a response that accurately completes the request.  
> `### Response:`  
> *(the same template continues repeatedly)*

**After SFT**

> Photosynthesis is a process by which plants and animals convert sunlight, carbon dioxide, and other organic compounds into glucose, oxygen, and other chemical compounds. It is a complex process that involves the conversion of carbon dioxide into glucose, oxygen, and other chemical compounds.

The response now follows the requested two-sentence shape, although its claim about animals is inaccurate.

</details>

<details>
<summary><strong>Example 2: fluent form does not guarantee correctness</strong></summary>

**Prompt**

> Calculate the average speed for 180 km in 3 hours.

**Before SFT**

> Calculate the average speed for 180 km in 3 hours.  
> `### Instruction:`  
> *(the prompt and markers continue repeatedly)*

**After SFT**

> The average speed for 180 km in 3 hours is approximately 0.8 km/h.

The expected answer is 60 km/h. This is a compact example of why the checkpoint should be studied, not trusted.

</details>

<details>
<summary><strong>Example 3: formatting learned better than content</strong></summary>

**Prompt**

> Give exactly three concise job interview tips.

**After SFT**

> 1. **Know Your Job:** A good job interview is a great way to learn about your skills, interests, and abilities.  
> 2. **Be Prepared:** Prepare your resume and cover letter to ensure that you are prepared for the interview.  
> 3. **Be Prepared for the Interview:** Prepare your resume and cover letter to ensure that you are prepared for the interview.

The response has three distinguishable items, but it is redundant and not especially concise.

</details>

The complete, unedited generations are provided in [`comparison_results.csv`](https://huggingface.co/shehryars715/pythia-410m-step5000-alpaca/blob/main/comparison_results.csv). Measured baseline and final validation losses were not present in that CSV, so no loss values are invented here.

## Intended uses

- Studying the effect of instruction tuning on an early language-model checkpoint
- Demonstrating full-parameter SFT and completion-only loss
- Small-scale educational experiments with Pythia checkpoints

## Limitations and risks

- The `step5000` checkpoint is only partially pretrained.
- The model is expected to be weak in factual knowledge, reasoning, safety, formatting reliability, and multi-turn conversation.
- Outputs may be incorrect, biased, incoherent, repetitive, or unsafe.
- The model has not been evaluated sufficiently for consequential decisions or production deployment.
- The simple rule-based checks in the training notebook are not substitutes for broad human evaluation.

Users should validate outputs for their domain and add appropriate safeguards before any interactive use.

## License and attribution

The Pythia base model is published under Apache-2.0. At the time this card was written, the Hugging Face metadata for `yahma/alpaca-cleaned` displayed CC BY 4.0, while text inside its dataset card stated CC BY-NC 4.0. Review the current upstream model and dataset terms before choosing a license for or redistributing this fine-tuned checkpoint.

## Citation

```bibtex
@article{biderman2023pythia,
  title={Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling},
  author={Biderman, Stella and Schoelkopf, Hailey and Anthony, Quentin and others},
  journal={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```

```bibtex
@misc{alpaca,
  author={Taori, Rohan and Gulrajani, Ishaan and Zhang, Tianyi and others},
  title={Stanford Alpaca: An Instruction-following LLaMA Model},
  year={2023},
  howpublished={\url{https://github.com/tatsu-lab/stanford_alpaca}}
}
```
