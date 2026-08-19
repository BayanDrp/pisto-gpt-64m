#!/usr/bin/env bash
# ============================================================
# pisto-gpt-64m — Stage 1: Pretrain on Colab T4 (via Colab CLI)
# Retrains Arabic tokenizer + patches config before pretraining.
# Weights auto-downloaded to your PC on completion.
# NOTE: colab exec runs PYTHON code, so all remote steps are Python.
# ============================================================
set -euo pipefail

SESSION="pisto"
REPO="/home/fedora/Documents/projects/pisto-gpt-64m"
BUNDLE="/tmp/pisto-gpt-64m.tgz"
REMOTE="/content/pg"

echo "==> [1/8] Checking for existing session '$SESSION'..."
if colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
    echo "    Reusing existing session (no new assignment needed)"
else
    echo "    Provisioning Colab T4 session '$SESSION'..."
    colab new -s "$SESSION" --gpu T4
fi

echo "==> [2/8] Waiting for session to be ready..."
for i in $(seq 1 30); do
    if colab status -s "$SESSION" 2>/dev/null | grep -qiE "busy|running|ready|idle"; then
        echo "    session ready"
        break
    fi
    sleep 5
done

echo "==> [3/8] Uploading repo to VM..."
cd "$REPO"
tar czf "$BUNDLE" --exclude=.git --exclude=weights/pretrain_best.pt --exclude=weights/instruct_best.pt .
echo "import os; os.makedirs('$REMOTE', exist_ok=True); print('dir ready')" | colab exec -s "$SESSION"
colab upload -s "$SESSION" "$BUNDLE" "$REMOTE/bundle.tgz"
echo "import subprocess; subprocess.run('cd $REMOTE && tar xzf bundle.tgz && rm bundle.tgz', shell=True, check=True); print('extracted')" | colab exec -s "$SESSION"

echo "==> [4/8] Installing dependencies..."
colab install -s "$SESSION" flask torch datasets tokenizers

echo "==> [5/8] Retraining Arabic tokenizer..."
echo "import subprocess; subprocess.run('cd $REMOTE && python3 scripts/train_tokenizer.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout 3600

echo "==> [6/8] Patching config/train.json for Arabic dataset..."
echo "
import json
cfg_path = '$REMOTE/config/train.json'
with open(cfg_path) as f:
    cfg = json.load(f)
cfg['dataset'] = {'name': 'gagan3012/arabictext', 'split': 'train', 'max_docs': 200000, 'min_len': 50, 'train_split': 0.95}
with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=4, ensure_ascii=False)
print('config/train.json patched for Arabic')
" | colab exec -s "$SESSION"

echo "==> [7/8] Running pretraining (this runs until max_hours cap)..."
echo "import subprocess; subprocess.run('cd $REMOTE && python3 training/pretrain.py', shell=True, check=True)" | colab exec -s "$SESSION" --timeout 21600

echo "==> [8/8] Downloading weights to your PC..."
mkdir -p "$REPO/weights"
colab download -s "$SESSION" "$REMOTE/weights/pretrain_best.pt" "$REPO/weights/pretrain_best.pt"

echo ""
echo "DONE. Pretrained weights saved to:"
echo "  $REPO/weights/pretrain_best.pt"
echo ""
echo "Next: run ./scripts/colab_finetune.sh to instruction-tune this checkpoint."
echo ""
echo "Optional: colab stop -s $SESSION   (frees the GPU, stops billing)"
