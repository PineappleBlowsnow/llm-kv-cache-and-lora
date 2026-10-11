# Implementing KV Caching and LoRA for Transformers

**MVA course — LLMs for Code and Proof**

**KV-cached decoding and low-rank adaptation implemented in PyTorch on a small minGPT decoder.** This is Ying JIN's MVA *LLMs for Code and Proof* coursework, organized as a compact, inspectable implementation portfolio.

## What I implemented

- **KV cache:** thread per-layer keys/values through attention and transformer blocks, offset positional embeddings and causal masks during incremental decoding, and rebuild the cache when generation exceeds the context window.
- **LoRA:** initialize and apply low-rank updates, implement layer-level merge/unmerge, and complete the adapter-training stage for a synthetic sorting task. The base model is frozen with the inherited `get_lora_model` utility.
- **Evaluation:** exercise cached-generation equivalence, compare full-context recomputation with incremental decoding, and test adaptation from length-6 to length-10 sorting sequences.

The course supplied the minGPT model/trainer, demonstrations, helper utilities and grading framework. This is an implementation exercise, not a new KV-cache or LoRA method. [Provenance and exact contribution boundaries](PROVENANCE.md) distinguish inherited scaffolding from my changes.

## Quick start

Use Python 3.9+ with a compatible PyTorch installation. The small public smoke check uses CPU, random tensors and no downloaded weights or datasets.

```bash
python -m pip install -e .
python smoke_check.py
```

The check compares cached and ordinary decoding (including context overflow), verifies the LoRA formula and layer merge/unmerge, and checks that an adapter training step leaves base weights frozen. **It is newly added packaging code, not the historical classroom grader.** See the validation status below before interpreting it as a passed test.

For the original training demonstrations, run from the repository root:

```bash
python -m kv_cache.demo_sort_kv
python -m lora.demo_sort_lora
```

These commands train tiny models; they are not instant smoke tests. The LoRA example runs 1,000 base-training steps and 2,000 adapter-training steps. It generates synthetic examples locally. Set `PYTHONHASHSEED` to a fixed value **before starting Python** for repeatable train/test partitioning; the original demo uses Python's hash and dynamic sampling, so exact historical scores are not guaranteed.

```bash
python -m kv_cache.benchmark
```

This optional benchmark includes `gpt-mini` and an approximately 86.8M-parameter GPT configuration and may take several minutes. It measures a single decode step after prefill, excluding prefill time. The inherited timing code does not synchronize accelerators; use its timings only on CPU. It is not a GPU-serving or throughput benchmark.

`transformers` is optional (`pip install -e ".[pretrained]"`) and is needed only for the inherited GPT-2 pretrained-loading helper; none of the commands above downloads pretrained weights.

## Historical evidence

Saved local results from **28 March 2026**, not newly reproduced results:

| Check / experiment | Recorded result | Scope |
|---|---|---|
| Classroom KV-cache checks | 21 passed | Supplied grading suite, excluded from this public snapshot |
| Classroom LoRA checks | 33 passed | Supplied grading suite, excluded from this public snapshot |
| Length-10 sorting before adaptation | 0 / 5,000 test samples correct | After length-6 base training |
| Length-10 sorting after LoRA | 4,504 / 5,000 = **90.08%** correct | Full-sequence accuracy after 2,000 adapter steps on CPU |

[Historical logs](evidence/) retain the experiment context. Samples are dynamically generated and need not be unique. The sorting result is a toy distribution-shift experiment, not a language-model SFT benchmark, and has no multi-seed uncertainty estimate.

## Validation and limitations

Portfolio preparation on **26 September 2026** parsed all included Python source and checked package contents for common credential patterns. The available packaging runtimes did not contain PyTorch; **the new smoke check, original training demonstrations and performance benchmark were not executed in this preparation**. No checkpoints or grader are required to inspect the code.

- Cache rebuilding on a sliding context preserves the model's learned absolute-position behavior, but gives up incremental-decoding speed beyond the context limit.
- LoRA merging is exposed by `LoRALinear.eval()`; a parent model's `eval()` does not call each child's overridden `eval()` method. The unmerged adapter path remains valid, but do not infer merged-model inference speed from calling `model.eval()` alone.
- This is educational code with dense attention and concatenated caches, not a production serving engine.

## Files and attribution

| Path | Purpose |
|---|---|
| `kv_cache/kv_cache.py` | Cached attention, block/model forward and generation |
| `lora/lora.py` | Low-rank linear layers and GPT integration |
| `mingpt/` | Inherited model, trainer and configuration utilities |
| `smoke_check.py` | Independent checks added for this public snapshot |
| `evidence/` | Historical summaries and synthetic-task logs |

Original classroom source: [llm-2026/llm-efficiency-PineappleBlowsnow](https://github.com/llm-2026/llm-efficiency-PineappleBlowsnow) (access may be restricted). The original [Apache 2.0 license](LICENSE) is preserved, with the upstream [minGPT MIT notice](LICENSE-minGPT). See [PROVENANCE.md](PROVENANCE.md) for credits and packaging changes. Handouts, classroom graders, submission scripts, notebooks, environments, personal paths and generated checkpoints are excluded.


[Back to the MVA project portfolio](https://github.com/PineappleBlowsnow/mva-portfolio)
