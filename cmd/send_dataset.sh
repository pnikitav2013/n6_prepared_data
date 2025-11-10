#!/usr/bin/env bash
set -euo pipefail

# CHANGE THESE CONSTANTS TO MATCH YOUR SETUP
SOURCE_DIR="/workspace/data/output"   # Local directory with files to upload
B2_BUCKET="dataset460"                # B2 bucket name
B2_REMOTE_PREFIX="row"                # Folder prefix inside the bucket (use empty string for root)

if ! command -v b2 >/dev/null 2>&1; then
	echo "Error: b2 CLI not found in PATH." >&2
	exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
	echo "Error: SOURCE_DIR '$SOURCE_DIR' does not exist." >&2
	exit 1
fi

mapfile -d '' files < <(find "$SOURCE_DIR" -type f -print0)

if [ ${#files[@]} -eq 0 ]; then
	echo "No files to upload in '$SOURCE_DIR'."
	exit 0
fi

for file in "${files[@]}"; do
	relative_path="${file#$SOURCE_DIR/}"
	if [ -n "$B2_REMOTE_PREFIX" ]; then
		remote_path="$B2_REMOTE_PREFIX/$relative_path"
	else
		remote_path="$relative_path"
	fi

	echo "Uploading '$file' to '${B2_BUCKET}/${remote_path}'"
	b2 file upload "$B2_BUCKET" "$file" "$remote_path"
done

echo "Upload complete."
