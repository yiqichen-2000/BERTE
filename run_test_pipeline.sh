#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INPUT_FASTA="${INPUT_FASTA:-input_fasta/demo_test.fasta}"
PREFIX="${PREFIX:-test}"
FRAGMENT_SIZE="${FRAGMENT_SIZE:-1500}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/test_pipeline}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/work_data_${FRAGMENT_SIZE}bp}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
PROTEIN_DIR="${PROTEIN_DIR:-${OUTPUT_DIR}/protein_outputs}"
CDD_DB="${CDD_DB:-CDD/RPSTBLASTN_LIB/db_large.pn}"
RF_MODEL="${RF_MODEL:-models/RF.pkl}"
THREADS="${THREADS:-4}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEQ_LEN="${SEQ_LEN:-502}"
SAVE_NAME="${SAVE_NAME:-berte_predictions}"
EVALUATE="${EVALUATE:-0}"
MODE_ARGS=()
case "$EVALUATE" in
  0) ;;
  1) MODE_ARGS=(--evaluate) ;;
  *) echo "EVALUATE must be 0 (prediction) or 1 (labelled evaluation)." >&2; exit 1 ;;
esac
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Required file not found or empty: $1" >&2
    exit 1
  fi
}

echo "[1/6] Checking inputs"
require_file "$INPUT_FASTA"
require_file "$CDD_DB"
require_file "$RF_MODEL"
for k in 3 4 5 6; do
  require_file "models/berte_${k}mer_finetuned.h5"
done

mkdir -p "$WORK_DIR" "$LOG_DIR" "$PROTEIN_DIR"

echo "[2/6] Preprocessing test FASTA"
python -u code/preprocessing/preprocess_test_fasta.py "$INPUT_FASTA" \
  --fragment-size "$FRAGMENT_SIZE" \
  --output-dir "$WORK_DIR" \
  --prefix "$PREFIX" \
  "${MODE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/01_preprocess_test_fasta.log"

echo "[3/6] Extracting CDD protein-domain features"
python -u code/cdd_features/extract_cdd_features.py \
  -i "$INPUT_FASTA" \
  -d "$CDD_DB" \
  -t "$THREADS" \
  --out1 "$PROTEIN_DIR/${PREFIX}_cdd_raw_hits.tsv" \
  --out2 "$WORK_DIR/${PREFIX}_cdd_features.txt" \
  2>&1 | tee "$LOG_DIR/02_extract_cdd_features.log"

echo "[4/6] Exporting BERTE embeddings"
for k in 3 4 5 6; do
  echo "  - ${k}-mer embeddings"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u code/finetuning/extract_berte_embeddings.py \
    "models/berte_${k}mer_finetuned.h5" \
    "$WORK_DIR" \
    --prefix "$PREFIX" \
    --k "$k" \
    --stride "$k" \
    --batch_size "$BATCH_SIZE" \
    --seq_len "$SEQ_LEN" \
    --output "$WORK_DIR/${PREFIX}_${k}mer_embeddings.pkl" \
    2>&1 | tee "$LOG_DIR/03_extract_${k}mer_embeddings.log"
done

echo "[5/6] Checking RF model"
require_file "$RF_MODEL"

echo "[6/6] Predicting with the hierarchical RF model"
python -u code/hierarchical_classifier/predict_hierarchical_rf.py \
  "$RF_MODEL" \
  "$WORK_DIR" \
  --filePDB "$CDD_DB" \
  --split "$PREFIX" \
  --save_name "$SAVE_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --threads "$THREADS" \
  --store_predictions \
  "${MODE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/04_predict_hierarchical_rf.log"

echo "Test inference pipeline completed. Outputs are under: $OUTPUT_DIR"
