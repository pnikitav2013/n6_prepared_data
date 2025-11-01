#!/usr/bin/env bash
set -euo pipefail

# Скрипт скачивает и распаковывает train-clean-100.tar.gz
# Использование: ./load_dataset_small.sh /path/to/destination
# По умолчанию скачивается https://www.openslr.org/resources/12/train-clean-100.tar.gz

# ARCHIVE_URL_DEFAULT="https://www.openslr.org/resources/12/train-clean-100.tar.gz"
ARCHIVE_URL_DEFAULT="https://www.openslr.org/resources/12/test-clean.tar.gz"
print_usage() {
	cat <<EOF
Использование: $(basename "$0") DEST_DIR [--keep-archive]

Параметры:
	DEST_DIR         Папка, в которую будет скачан и распакован архив
	--keep-archive   Не удалять скачанный .tar.gz после распаковки

Пример:
	$(basename "$0") /data/librispeech
EOF
}

if [ "$#" -lt 1 ]; then
	print_usage
	exit 1
fi

DESTDIR="$1"
KEEP_ARCHIVE=false
if [ "${2:-}" = "--keep-archive" ]; then
	KEEP_ARCHIVE=true
fi

ARCHIVE_URL="$ARCHIVE_URL_DEFAULT"
ARCHIVE_NAME=$(basename "$ARCHIVE_URL")

mkdir -p "$DESTDIR"

# Make DESTDIR absolute so tar -C works even after we change directory to TMPDIR
if command -v realpath >/dev/null 2>&1; then
	DESTDIR="$(realpath -m "$DESTDIR")"
else
	# fallback: cd in a subshell and print pwd
	DESTDIR="$(cd "$DESTDIR" && pwd)"
fi

# Use a temporary working directory for download
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading $ARCHIVE_URL to temporary folder..."
cd "$TMPDIR"

if command -v wget >/dev/null 2>&1; then
	wget -c "$ARCHIVE_URL" -O "$ARCHIVE_NAME"
elif command -v curl >/dev/null 2>&1; then
	curl -C - -L "$ARCHIVE_URL" -o "$ARCHIVE_NAME"
else
	echo "Ошибка: neither wget nor curl found. Install wget or curl and re-run." >&2
	exit 2
fi

echo "Скачано: $TMPDIR/$ARCHIVE_NAME"
echo "Распаковка в: $DESTDIR"

# Extract into destination directory
tar -xzf "$ARCHIVE_NAME" -C "$DESTDIR"

if [ "$KEEP_ARCHIVE" = false ]; then
	rm -f "$ARCHIVE_NAME"
fi

echo "Готово. Содержимое распаковано в: $DESTDIR"

