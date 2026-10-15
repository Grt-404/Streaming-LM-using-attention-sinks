<div align="center">

# StreamingLLM from Scratch in TensorFlow

**Attention sinks for infinite-length streaming inference on a pretrained Llama-family model**

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/grt404/streaminglm-using-attentionsinks-implementation)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Model-TinyLlama--1.1B-blue)
![Dataset](https://img.shields.io/badge/Data-PG--19-green)

</div>

A from-scratch TensorFlow re-implementation of [StreamingLLM (Xiao et al., ICLR 2024)](https://arxiv.org/abs/2309.17453) on top of pretrained **TinyLlama-1.1B**. It includes RMSNorm, RoPE, grouped-query attention, a SwiGLU MLP and a rolling KV cache with position re-assignment. No fine-tuning is used.

## TL;DR

Keeping just **4 initial "sink" tokens** plus a rolling window of recent tokens stops the model from collapsing once the cache is full.

| Method (cache = 1024) | Perplexity, 6K-token PG-19 stream |
|---|---|
| Dense attention | 663.96 |
| Window attention (0 + 1024) | 164.43 |
| **StreamingLLM (4 + 1020)** | **8.39** |

---

## Repository

```
.
├── streaminglm-using-attentionsinks-implementation.ipynb   # main experiment
├── AblationStudies.ipynb                                   # ablations
├── Reproducing_StreamingLLM__...Llama_Family_Model.pdf     # IEEE-format report
└── README.md
```

| File | What it does |
|---|---|
| **Main notebook** | Loads TinyLlama weights, implements the model and cache, runs correctness tests, compares dense / window / StreamingLLM, and plots attention sinks. |
| **Ablation notebook** | Varies the number of sink tokens and the sink content. Standalone: it redefines the model code. |
| **Report (PDF)** | Method, results, failure cases and discussion. |

---

## Quick start

### Run on Kaggle (recommended)

1. Open the [main notebook on Kaggle](https://www.kaggle.com/code/grt404/streaminglm-using-attentionsinks-implementation).
2. Turn on a **GPU** accelerator and **Internet** in the notebook settings.
3. Click **Save & Run All**.

### Run locally

```bash
pip install tensorflow transformers safetensors huggingface_hub datasets matplotlib numpy pandas
jupyter notebook streaminglm-using-attentionsinks-implementation.ipynb
```

Run all cells top to bottom. The first cell is Kaggle boilerplate (`kagglehub`) and can be deleted outside Kaggle.

**Requirements:** Python 3.10+, a GPU (experiments used a Tesla T4 in fp16; the code falls back to fp32 on CPU but is very slow), about 3 GB of disk, and internet access.

### What gets downloaded automatically

| Resource | Identifier |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` (22 layers, 32 query heads, 4 KV heads, head dim 64, pre-trained on 2048 tokens) |
| Data | `emozilla/pg19-test`, first book only, truncated to 6,000 tokens |

No Hugging Face login is needed for TinyLlama.

### Outputs

The main notebook saves `attention_sinks.png`, `ppl_comparison.png`, `ppl_bars.png` and `attn_cache_slots.png`. The ablation notebook caches each run as `A_<s>.npy` and `nll_nl.npy`, so an interrupted run resumes where it stopped.

---

## Seeds and reproducibility

- **No sampling is used.** Evaluation is teacher-forced next-token loss on a fixed token stream with fixed weights, so the perplexity results do not depend on a seed. Tiny differences can still come from fp16 accumulation order on the GPU.
- The only random step is the RoPE sanity test (`tf.random.normal`). It checks a property that holds for any seed, but the printed score values change between runs.
- To fix it, add this to the first code cell:

```python
import random, numpy as np, tensorflow as tf
SEED = 0
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

---

## How it works

1. **Why window attention fails:** the first tokens act as attention sinks that soak up surplus softmax mass. Evicting them pushes the model out of distribution.
2. **StreamingLLM cache:** keep the first `s` tokens permanently, plus the last `w` tokens, giving a constant cache of `s + w` entries.
3. **Positions inside the cache:** keys are cached *before* RoPE and re-rotated by their current cache slot at every step, so the model never sees a relative distance beyond the cache size.

The notebook has two code paths: `forward_full` (whole sequence, causal mask, used as a reference and for attention maps) and `step` (one token at a time through `RollingKVCache`).

### Built-in correctness checks

| Check | Result |
|---|---|
| RoPE: equal offsets give equal scores | Identical at offsets (10, 3), (1010, 1003), (3000, 2993) |
| `RollingKVCache(2, 3)` over 8 tokens | Keeps tokens `[0, 1, 5, 6, 7]` |
| `step` vs `forward_full` with a dense cache (48 tokens) | Max logit difference 0.055 (fp16) |

---

## Results

### Main comparison

TinyLlama-1.1B, PG-19 book 1, first 6,000 tokens, cache budget 1024.

| Method | Cache | PPL (all) | PPL (after token 1024) | Paper (Llama-2-13B, 65K tokens) |
|---|---|---|---|---|
| Dense | unbounded | 663.96 | 1673.6 | fails past pre-training length |
| Window | 0 + 1024 | 164.43 | 311.0 | 5158.07 |
| **StreamingLLM** | **4 + 1020** | **8.39** | **8.6** | 5.40 |

All three methods are identical until the cache fills (running perplexity 7.59 at token 1000). The model and stream length differ from the paper, so compare the ordering and the recovery, not the absolute numbers.

### Ablation A: number of sink tokens

First 4,000 tokens, total cache fixed at 1024.

| Sinks + window | PPL (all) | PPL (after 1024) |
|---|---|---|
| 0 + 1024 | 139.37 | 382.17 |
| 1 + 1023 | 8.22 | 8.50 |
| 2 + 1022 | 8.22 | 8.50 |
| 4 + 1020 | 8.21 | 8.50 |
| 8 + 1016 | 8.20 | 8.48 |
| 16 + 1008 | 8.20 | 8.49 |

A single sink recovers almost all the quality; more sinks add very little on this model.

### Ablation B: sink content

| Config (4 + 1020) | PPL (all) | PPL (after 1024) |
|---|---|---|
| Real first 4 tokens | 8.21 | 8.50 |
| First 4 tokens replaced by newline | 8.46 | 8.76 |

Replacing the sink content costs only about 0.26 PPL after token 1024, so position matters far more than content. The "all" column for the newline run excludes the first three positions in the notebook, so compare the "after 1024" column.

### Attention sink analysis

- Layers 0–2 put almost no attention on token 0 and behave locally.
- Layers 3–8 put roughly 0.7–0.9 of attention on token 0, and later layers about 0.4–0.6.
- Inside the rolling cache, StreamingLLM keeps about 0.5–0.85 of attention on the four sink slots, while plain window attention puts about 0.01 on its first four slots.

---

## Limitations

- One book (6K tokens, 4K for ablations), one model and one cache budget. The paper uses up to 4M-token streams on several model families.
- No sliding-window-with-recomputation baseline, so parity with recomputation is not claimed.
- Perplexity does not measure long-range memory, and no retrieval test (e.g. StreamEval) was run.
- Latency and memory were not measured. The per-token Python loop re-rotates the whole cache at every layer, so the paper's speed-ups are not reproduced.

---

## References

- G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. *Efficient Streaming Language Models with Attention Sinks.* [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- P. Zhang et al. *TinyLlama: An Open-Source Small Language Model.* [arXiv:2401.02385](https://arxiv.org/abs/2401.02385)
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- J. Rae et al. *Compressive Transformers for Long-Range Sequence Modelling.* ICLR 2020 (PG-19)
