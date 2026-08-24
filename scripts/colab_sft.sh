#!/usr/bin/env bash
# ============================================================
# pisto-gpt-64m — SFT on Colab T4 via Colab CLI
#   usage: scripts/colab_sft.sh <config-basename> <out-path>
#   example: scripts/colab_sft.sh sft_test.json /tmp/instruct_test.pt
#            scripts/colab_sft.sh sft.json ~/Downloads/instruct_best.pt
# ============================================================
set -euo pipefail

SESSION="pisto"
REPO="/home/fedora/Documents/projects/pisto-gpt-64m"
BUNDLE="/tmp/pisto-gpt-64m.tgz"
REMOTE="/content/pg"
CONF="${1:-sft.json}"
OUT="${2:-$REPO/weights/instruct_best.pt}"
SFT_TIMEOUT="${3:-21600}"

echo "==> [1/7] Ensure session '$SESSION' (T4)..."
colab new -s "$SESSION" --gpu T4 2>/dev/null || echo "    (session already exists)"

echo "==> [2/7] Bundling repo (excl .git, instruct_best, generated data)..."
cd "$REPO"
rm -f "$BUNDLE"
tar czf "$BUNDLE" --exclude=.git --exclude='weights/instruct_best.pt' \
    --exclude='data/sft_*.jsonl' --exclude='__pycache__' --exclude='*.pyc' .
echo "    bundle: $(du -h "$BUNDLE" | cut -f1)"

echo "==> [3/7] Upload + extract..."
echo "import os; os.makedirs('$REMOTE', exist_ok=True); print('ready')" | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" "$REMOTE/bundle.tgz"
echo "import subprocess; subprocess.run('cd $REMOTE && tar xzf bundle.tgz && rm bundle.tgz', shell=True, check=True); print('extracted')" | colab exec -s "$SESSION"

echo "==> [4/7] Install deps..."
colab install -s "$SESSION" flask torch datasets tokenizers

echo "==> [5/7] Build SFT dataset (SFT_CONFIG=$CONF)..."
echo "import subprocess; subprocess.run('cd $REMOTE && SFT_CONFIG=$CONF python3 scripts/build_sft_data.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout 1800

echo "==> [6/7] Train SFT (SFT_CONFIG=$CONF)..."
echo "import subprocess; subprocess.run('cd $REMOTE && SFT_CONFIG=$CONF python3 training/sft.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout "$SFT_TIMEOUT"

echo "==> [7/7] Download weights -> $OUT"
mkdir -p "$(dirname "$OUT")"
colab download -s "$SESSION" "$REMOTE/weights/instruct_best.pt" "$OUT"

echo ""
echo "DONE -> $OUT"
echo "Free the GPU later: colab stop -s $SESSION"
