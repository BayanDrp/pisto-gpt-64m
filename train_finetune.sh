#!/usr/bin/env bash
# ============================================================
# pisto-gpt-64m — Stage 2: Instruction Tuning on Colab T4
# Uses the pretrained best.pt (from Stage 1). Configs untouched.
# Weights auto-downloaded to your PC on completion.
# NOTE: colab exec runs PYTHON code, so all remote steps are Python.
# ============================================================
set -euo pipefail

SESSION="pisto"
REPO="/home/fedora/Documents/projects/pisto-gpt-64m"
BUNDLE="/tmp/pisto-gpt-64m.tgz"
REMOTE="/content/pg"

echo "==> [1/6] Checking session '$SESSION'..."
if ! colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
    echo "    Session not running — provisioning a new T4..."
    colab new -s "$SESSION" --gpu T4
    for i in $(seq 1 30); do
        if colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
            break
        fi
        sleep 5
    done
fi

echo "==> [2/6] Uploading repo + pretrained weights to VM..."
cd "$REPO"
tar czf "$BUNDLE" --exclude=.git --exclude=weights/instruct --exclude=weights/instruct_best.pt .
echo "import os; os.makedirs('$REMOTE', exist_ok=True); print('dir ready')" | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" "$REMOTE/bundle.tgz"
echo "import subprocess; subprocess.run('cd $REMOTE && tar xzf bundle.tgz && rm bundle.tgz', shell=True, check=True); print('extracted')" | colab exec -s "$SESSION"

echo "==> [3/6] Verifying pretrained checkpoint exists on VM..."
echo "import subprocess; subprocess.run('ls -la $REMOTE/weights/best.pt', shell=True)" | colab exec -s "$SESSION"

echo "==> [4/6] Installing dependencies..."
colab install -s "$SESSION" flask torch datasets tokenizers

echo "==> [5/6] Running instruction tuning..."
echo "import subprocess; subprocess.run('cd $REMOTE && python3 training/finetune.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout 21600

echo "==> [6/6] Downloading tuned weights to your PC..."
mkdir -p "$REPO/weights/instruct"
colab download -s "$SESSION" "$REMOTE/weights/instruct/instruct_best.pt" "$REPO/weights/instruct/instruct_best.pt"

echo ""
echo "DONE. Instruction-tuned weights saved to:"
echo "  $REPO/weights/instruct/instruct_best.pt"
echo ""
echo "Optional: colab stop -s $SESSION   (frees the GPU, stops billing)"