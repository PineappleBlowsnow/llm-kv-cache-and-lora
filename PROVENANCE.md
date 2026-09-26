# Provenance and evidence

Prepared on 26 September 2026 as a public snapshot, rather than a rewrite of the original coursework repository.

## Sources and authorship

- Course source: `https://github.com/llm-2026/llm-efficiency-PineappleBlowsnow`.
- Classroom baseline: `acbf43518ab7d564104da44d3addc72d3307a418` (16 March 2026).
- Ying JIN / PineappleBlowsnow implementation commits: `971b282c76ea5af1bd356909ebbd2ee805617e7c` and `a5d95d4df8c66c63ada9763e78661b81e51165eb` (28 March 2026).
- The preceding teaching commit `c089aba` is by Marc Lelarge. The course README credits Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT/); source comments also cite the OpenAI GPT-2 and Hugging Face implementations.

The baseline-to-final source diff identifies these student additions:

| File | Added/completed work |
|---|---|
| `kv_cache/kv_cache.py` | Cached attention and mask offsets; block/model cache propagation; position offsets; autoregressive generation and context overflow handling |
| `lora/lora.py` | Adapter initialization and forward update; layer merge/unmerge; optimizer construction |
| `lora/demo_sort_lora.py` | Adapter-only training stage on length-10 sorting |

The minGPT model/trainer, GPT/LoRA structural classes, `get_lora_model` freezing helper, synthetic dataset/demo scaffolding and benchmark framework are inherited. Git attribution supports the implementation scope above; it is not a claim that every included line was independently authored. Portfolio documentation and the independent smoke check were prepared with AI assistance in September 2026.

## Historical run evidence

The original local directory `results/20260328_152332/` contains:

- `test_kv_cache.txt`: 21 passed in 4.83 seconds.
- `test_lora.txt`: 33 passed in 6.05 seconds.
- `demo_sort_lora.txt`: CPU run, 1,000 base steps on sorting length 6, then 2,000 LoRA steps on length 10; 0/5,000 before and 4,504/5,000 after adaptation on the length-10 test samples.
- `benchmark_kv.txt`: CPU timings for single incremental decoding steps across context lengths.

This package includes a short test summary, the synthetic sorting log and a CPU benchmark log with the machine-specific output path removed. It does not include the classroom grading source. Tests recorded in March were not rerun during packaging in September. No statistical confidence, GPU speedup, or real-world language-task quality is inferred from these records.

## Packaging changes

- Replaced assignment instructions with the project README and this provenance note.
- Kept the core `kv_cache.py`, `lora.py` and `mingpt` source unchanged from the supplied local snapshot.
- Corrected imports in the two demos and benchmark to support repository-root `python -m ...` invocation; added `lora/__init__.py`.
- Moved the optional pretrained helper's Transformers dependency to the `pretrained` extra; added conservative major-version bounds for the minimal Torch/NumPy requirements. These ranges are not a newly tested dependency lock.
- Added an independent `smoke_check.py`; it checks cached-decoding equivalence and adapter behavior without importing grading code.
- Excluded the assignment handout, practical notebooks, course grader/tests, submission scripts, environments, Git metadata, caches and checkpoints.
- Preserved the source Apache 2.0 license and added the upstream minGPT MIT notice, fetched from `https://raw.githubusercontent.com/karpathy/minGPT/master/LICENSE` on 26 September 2026. No original copyright notice was removed.

The original coursework directory was not modified. No explicit publication restriction was found in the reviewed course README or extracted handout; this snapshot preserves source attribution and licenses without claiming a separate instructor endorsement.
