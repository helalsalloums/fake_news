#!/usr/bin/env bash
# Downloads the ARAFA dataset (CC-BY-NC-SA-4.0) from Zenodo and verifies its checksum.
# Source: https://zenodo.org/records/16762969
#
# Citation (required if you use this dataset):
#   Khalil, C., Elbassuoni, S., & Assaf, R. (2025).
#   ARAFA: An LLM Generated Arabic Fact-Checking Dataset.
#   Research Square. https://doi.org/10.21203/rs.3.rs-7335564/v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/../artifacts/raw"
DEST_FILE="$DEST_DIR/ARAFA.json"
URL="https://zenodo.org/records/16762969/files/ARAFA.json?download=1"
EXPECTED_MD5="87b89f1813cd808ac49a20d35d86bc2c"

mkdir -p "$DEST_DIR"

if [ -f "$DEST_FILE" ]; then
  echo "Found existing file at $DEST_FILE, verifying checksum before skipping download..."
  ACTUAL_MD5=$(md5sum "$DEST_FILE" | awk '{print $1}')
  if [ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]; then
    echo "Checksum matches. Skipping download."
    exit 0
  else
    echo "Checksum mismatch on existing file. Re-downloading..."
  fi
fi

echo "Downloading ARAFA.json from Zenodo (188.8 MB)..."
curl -L --fail "$URL" -o "$DEST_FILE"

echo "Verifying checksum..."
ACTUAL_MD5=$(md5sum "$DEST_FILE" | awk '{print $1}')
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
  echo "ERROR: checksum mismatch. Expected $EXPECTED_MD5, got $ACTUAL_MD5" >&2
  echo "The downloaded file may be corrupted or the source has changed. Aborting." >&2
  rm -f "$DEST_FILE"
  exit 1
fi

echo "Done. Saved to $DEST_FILE"
echo ""
echo "Reminder: ARAFA is licensed CC-BY-NC-SA-4.0 (noncommercial, share-alike)."
echo "Review the license before using this data: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"