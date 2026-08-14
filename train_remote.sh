#!/usr/bin/env bash
# ============================================================
# pisto-gpt-64m — Stage 1: Pretrain on Colab T4 (via Colab CLI)
# Configs untouched. Weights auto-downloaded to your PC on completion.
# NOTE: colab exec runs PYTHON code, so all remote steps are Python.
# ============================================================
set -euo pipefail

SESSION="pisto"
REPO="/home/fedora/Documents/projects/pisto-gpt-64m"
BUNDLE="/tmp/pisto-gpt-64m.tgz"
REMOTE="/content/pg"

echo "==> [1/6] Checking for existing session '$SESSION'..."
if colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
    echo "    Reusing existing session (no new assignment needed)"
else
    echo "    Provisioning Colab T4 session '$SESSION'..."
    colab new -s "$SESSION" --gpu T4
fi

echo "==> [2/6] Waiting for session to be ready..."
for i in $(seq 1 30); do
    if colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
        echo "    session ready"
        break
    fi
    sleep 5
done

echo "==> [3/6] Uploading repo to VM..."
cd "$REPO"
tar czf "$BUNDLE" --exclude=.git --exclude=weights/best.pt --exclude=weights/instruct --exclude=weights/instruct_best.pt .
echo "import os; os.makedirs('$REMOTE', exist_ok=True); print('dir ready')" | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" "$REMOTE/bundle.tgz"
echo "import subprocess; subprocess.run('cd $REMOTE && tar xzf bundle.tgz && rm bundle.tgz', shell=True, check=True); print('extracted')" | colab exec -s "$SESSION"

echo "==> [4/6] Installing dependencies..."
colab install -s "$SESSION" flask torch datasets tokenizers

echo "==> [5/6] Running pretraining (this runs until max_hours cap)..."
echo "import subprocess; subprocess.run('cd $REMOTE && python3 training/pretrain.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout 21600

echo "==> [6/6] Downloading weights to your PC..."
mkdir -p "$REPO/weights"
colab download -s "$SESSION" "$REMOTE/weights/best.pt" "$REPO/weights/best.pt"

echo ""
echo "DONE. Pretrained weights saved to:"
echo "  $REPO/weights/best.pt"
echo ""
echo "Next: run ./train_finetune.sh to instruction-tune this checkpoint."