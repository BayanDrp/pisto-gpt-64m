# Pisto GPT 68M

This repo contains a 68M parameter decoder only GPT. It starts with TinyStories pretraining and then gets instruction tuned on Alpaca plus a small manual dataset.

## Weights

If you just want to try the model, download the checkpoints from Hugging Face and place them in `weights/`:

- `pretrain_best.pt` for pretraining: [Download](https://huggingface.co/notpisto/pisto_gpt/resolve/main/weights/pretrain_best.pt)
- `instruct_best.pt` for fine tuning: [Download](https://huggingface.co/notpisto/pisto_gpt/resolve/main/weights/instruct_best.pt)

If Hugging Face warns about unauthenticated downloads, export `HF_TOKEN` before you start training.

## Quick Start

```bash
pip install -r requirements.txt
python cli.py
```

`cli.py` keeps things simple:

- `1` launches the Go web server
- `2` opens the training menu
  - `1` pretraining
  - `2` fine tuning

Once the app is running, open `http://localhost:8080`.

## Web UI

The web UI is a Go server (source in `ui/main.go` + `ui/handlres.go`) that
calls Python for inference via `llm/server_bridge.py`.

```bash
make server-build    # compile the Go server into ui/pisto-server
make server-tunnel   # run the server + expose it via a free Cloudflare tunnel (works on Colab)
# or: ./scripts/run_server.sh
```

## Docker

```bash
docker build -t pisto-gpt .
docker run -p 8080:8080 pisto-gpt
```

## Training

You can train on your own machine or on a free Colab T4 (via the `colab` CLI).

### Local

```bash
python training/pretrain.py    # pretrain from scratch on TinyStories
python training/finetune.py    # instruction-tune weights/pretrain_best.pt
```

Settings live in `config/train.json` and `config/instruct.json`. Both scripts
resume from checkpoints automatically and work on CPU or GPU.

### Colab (free GPU)

```bash
./scripts/colab_pretrain.sh    # Stage 1: pretrain, auto-downloads weights/pretrain_best.pt
./scripts/colab_finetune.sh    # Stage 2: tune, auto-downloads weights/instruct_best.pt
```

Or open `notebooks/colab_train.ipynb` in Colab — it saves checkpoints to your
Google Drive every 30s so a VM recycle never loses progress.

## Model

| Param | Value |
|---|---|
| Architecture | GPT Decoder (Pre-LN) |
| Parameters | ~68M |
| Layers | 10 |
| Heads | 8 |
| d_model | 720 |
| FFN | 2880 |
| Max length | 512 |
| Tokenizer | BPE, ByteLevel (vocab 8192) |
| Weight tying | Yes |

## Project Structure

```
├── cli.py               # CLI entry point
├── config/              # Training/generation configs + bpe_tokenizer.json
├── data/                # Training data
├── llm/                 # Model definition, tokenizer, inference
├── notebooks/           # Colab training notebook
├── scripts/             # Tokenizer training + Colab shell scripts
├── training/            # pretrain.py, finetune.py
├── ui/                  # Go web server (main.go, handlres.go, templates/, static/)
└── weights/             # Checkpoints (pretrain_best.pt, instruct_best.pt)
```
