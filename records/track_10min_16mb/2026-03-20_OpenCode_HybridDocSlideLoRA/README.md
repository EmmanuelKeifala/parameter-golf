# OpenCode Hybrid: Doc-Sliding + LoRA TTT

This run starts from the current best public 10-layer recipe and adds two leaderboard-inspired evaluation upgrades on top:

- document-isolated sliding-window evaluation
- lightweight per-document LoRA test-time training

The model artifact stays the same under-16MB compressed export path; only evaluation changes.

## Defaults baked into `train_gpt.py`

- `NUM_LAYERS=10`
- `EVAL_STRIDE=64`
- `EVAL_DOC_ISOLATED=1`
- `TTT_ENABLE=1`
- `TTT_LORA_RANK=4`
- `TTT_CHUNK_SIZE=128`
- `TTT_BATCH_SIZE=32`
- `MUON_WEIGHT_DECAY=0.02`

## 8xH100 validation command

```bash
RUN_ID=opencode_hybrid_docslide_lora \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_OpenCode_HybridDocSlideLoRA/train_gpt.py
```

Useful ablations:

```bash
# Doc-isolated sliding eval only
TTT_ENABLE=0 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_OpenCode_HybridDocSlideLoRA/train_gpt.py

# Faster / lighter TTT
TTT_LORA_RANK=2 TTT_BATCH_SIZE=16 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-20_OpenCode_HybridDocSlideLoRA/train_gpt.py
```

## Colab quickstart

Single-GPU Colab will not reproduce the 8xH100 leaderboard wallclock, but it is enough to smoke-test training and final evaluation.

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

RUN_ID=colab_hybrid_smoke \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=200 \
MAX_WALLCLOCK_SECONDS=0 \
TRAIN_BATCH_TOKENS=65536 \
VAL_LOSS_EVERY=0 \
TTT_BATCH_SIZE=8 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-03-20_OpenCode_HybridDocSlideLoRA/train_gpt.py
```

If Colab runs out of memory, lower one or more of:

- `TRAIN_BATCH_TOKENS`
- `EVAL_BATCH_SEQS`
- `TTT_BATCH_SIZE`
- `TTT_LORA_RANK`

Key output lines to watch:

- `final_int8_zlib_roundtrip_exact`
- `final_int8_docslide_ttt_lora_exact`
