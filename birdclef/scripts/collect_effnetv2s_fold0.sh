#!/bin/bash
# collect_effnetv2s_fold0.sh
# kernel v11 output에서 best_fold0.pth 다운로드 → ramkang/birdclef2026-effnetv2s-xeno 업로드
set -e

KERNEL_SLUG="ramkang/birdclef2026-effnet-multiwindow-train"
DATASET_ID="ramkang/birdclef2026-effnetv2s-xeno"
OUTDIR="/tmp/effnetv2s_fold0_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTDIR"

echo "[1/3] Downloading kernel output from $KERNEL_SLUG..."
kaggle kernels output "$KERNEL_SLUG" -p "$OUTDIR"

# Verify best_fold0.pth
if [ ! -f "$OUTDIR/best_fold0.pth" ]; then
    echo "ERROR: best_fold0.pth not found in kernel output"
    ls -la "$OUTDIR"
    exit 1
fi
echo "Found best_fold0.pth: $(du -h "$OUTDIR/best_fold0.pth" | cut -f1)"

echo "[2/3] Writing dataset-metadata.json..."
cat > "$OUTDIR/dataset-metadata.json" <<'EOF'
{
  "title": "BirdCLEF2026 EffNetV2-S Xeno Fold0",
  "id": "ramkang/birdclef2026-effnetv2s-xeno",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "[3/3] Uploading to $DATASET_ID..."
kaggle datasets version -p "$OUTDIR" -m "trial_049 fold0 EffNetV2-S + Xeno pretrain" 2>/dev/null \
  || kaggle datasets create -p "$OUTDIR"

echo "Done! Cleaning up $OUTDIR"
rm -rf "$OUTDIR"
echo "Dataset ready: https://www.kaggle.com/datasets/$DATASET_ID"
