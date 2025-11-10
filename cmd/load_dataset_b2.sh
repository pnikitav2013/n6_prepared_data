#!/usr/bin/env bash
set -euo pipefail

# Configure target bucket and destination directory.
BUCKET_NAME="dataset460"
DEST_DIR="output"

files=(
	"y_labels_phoneme.npy"
	"y_labels.npy"
	"y_label_length_phoneme.npy"
	"y_label_length.npy"
	"xy_sample_names.npy"
	"x_data.npy"
)

mkdir -p "${DEST_DIR}"

for file_name in "${files[@]}"; do
	remote_path="output/${file_name}"
	local_path="${DEST_DIR}/${file_name}"
	echo "Downloading ${remote_path} -> ${local_path}"
	b2 download-file-by-name "${BUCKET_NAME}" "${remote_path}" "${local_path}"
done

echo "Download complete."
