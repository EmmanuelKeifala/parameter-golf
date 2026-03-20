# OpenCode Best Preset

This folder packages the strongest recipe already present in the repo into a cleaner validation preset.

What is baked in by default:
- 10-layer 512-dim tied-embedding model
- overtone spectral embedding init
- phase-transition residual mixing with encoder/decoder-style skip reuse
- sliding-window evaluation enabled by default with `EVAL_STRIDE=64`
- Muon decoupled weight decay exposed as `MUON_WEIGHT_DECAY` and defaulted to `0.02`
- periodic validation disabled by default so the 10-minute wallclock is spent on training, with final evaluation still preserved

Recommended validation command:

```bash
RUN_ID=opencode_best_preset \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_OpenCode_BestPreset/train_gpt.py
```

Optional knobs worth trying without changing the architecture:
- `EVAL_BATCH_SEQS=512` if your box has spare memory and you want faster sliding eval
- `MUON_WEIGHT_DECAY=0.015` to `0.03` for minor robustness tuning
- `SEED=42` or `SEED=7` for reruns against the same preset

Expected behavior:
- artifact should remain under the 16 MB cap
- score should land in the same regime as the current top sliding-window 10-layer recipe, assuming comparable hardware and software stack
