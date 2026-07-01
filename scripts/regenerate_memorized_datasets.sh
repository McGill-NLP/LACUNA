#!/bin/bash
set -euo pipefail

DATASET_DIR="${1:-$HOME/OLMoBenchOutputs/saves/Train_95frozen_FullSubset/memorized_datasets}"
EXPERIMENT="${2:-OLMo_Mask_Train_FullSubset}"
BACKUP_DIR="${DATASET_DIR}_backup"

echo "=== Dataset dir: $DATASET_DIR"
echo "=== Experiment:  $EXPERIMENT"

# 1. Backup
if [ -d "$BACKUP_DIR" ]; then
    echo "Backup already exists at $BACKUP_DIR — remove it first if you want a fresh backup."
    exit 1
fi
echo "=== Creating backup at $BACKUP_DIR"
cp -r "$DATASET_DIR" "$BACKUP_DIR"

# 2. Save checksums of existing parquet files
echo "=== Computing checksums of existing files"
find "$BACKUP_DIR/data" -name '*.parquet' -print0 | sort -z | xargs -0 md5sum | sed 's|.*/||' > /tmp/checksums_before.txt

# 3. Regenerate
echo "=== Regenerating datasets"
cd "$(dirname "$0")/../src"
python create_memorized_datasets.py experiments="$EXPERIMENT"

# 4. Verify old splits are unchanged
echo "=== Verifying existing splits are unchanged"
find "$DATASET_DIR/data" -name '*.parquet' ! -name 'relearn*' -print0 | sort -z | xargs -0 md5sum | sed 's|.*/||' > /tmp/checksums_after.txt

if diff /tmp/checksums_before.txt /tmp/checksums_after.txt > /dev/null; then
    echo "=== OK: All pre-existing splits are identical"
    echo "=== Backup can be safely removed with: rm -rf $BACKUP_DIR"
else
    echo "=== WARNING: Some splits differ!"
    diff /tmp/checksums_before.txt /tmp/checksums_after.txt
    echo "=== Restore with: rm -rf $DATASET_DIR && mv $BACKUP_DIR $DATASET_DIR"
    exit 1
fi

# 5. Check new relearn file exists
if ls "$DATASET_DIR/data"/relearn* &>/dev/null; then
    echo "=== New relearn split created:"
    ls -lh "$DATASET_DIR/data"/relearn*
else
    echo "=== WARNING: No relearn split was created"
fi
