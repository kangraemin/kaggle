#!/bin/bash
# push_fold.sh <fold_number>
# TRAIN_FOLDS를 [N]으로 교체 후 EffNetV2-S multiwindow train kernel push
set -e

FOLD=${1:?"Usage: push_fold.sh <fold_number>"}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NOTEBOOK="$REPO_ROOT/birdclef/notebooks/birdclef2026-effnet-multiwindow-train.ipynb"
METADATA_SRC="$REPO_ROOT/birdclef/notebooks/effnet-multiwindow-train-kernel-metadata.json"
METADATA_DST="$REPO_ROOT/birdclef/notebooks/kernel-metadata.json"

echo "[1/3] Setting TRAIN_FOLDS = [$FOLD] in notebook..."
python3 - <<PYEOF
import json, re
nb = json.load(open('$NOTEBOOK'))
src = nb['cells'][3]['source']
nb['cells'][3]['source'] = [
    re.sub(r'TRAIN_FOLDS = \[.*?\]', 'TRAIN_FOLDS = [$FOLD]', line)
    for line in src
]
json.dump(nb, open('$NOTEBOOK', 'w'), ensure_ascii=False)
src_check = ''.join(nb['cells'][3]['source'])
assert 'TRAIN_FOLDS = [$FOLD]' in src_check, 'TRAIN_FOLDS update failed'
print(f'OK: TRAIN_FOLDS = [$FOLD]')
PYEOF

echo "[2/3] Copying training kernel metadata..."
cp "$METADATA_SRC" "$METADATA_DST"

echo "[3/3] Pushing kernel (fold=$FOLD)..."
cd "$REPO_ROOT/birdclef/notebooks"
caffeinate -s kaggle kernels push

echo "Done! Fold $FOLD training kernel pushed."
echo "Check status: kaggle kernels status ramkang/birdclef2026-effnet-multiwindow-train"
