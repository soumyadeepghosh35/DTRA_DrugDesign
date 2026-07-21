#!/usr/bin/env bash

set -euo pipefail

maxSizeMb="${1:-60}"
maxSizeBytes=$((maxSizeMb * 1024 * 1024))

addedCount=0
skippedCount=0
deletedCount=0

echo "Staging changed files smaller than or equal to ${maxSizeMb} MiB..."

# Find modified tracked files and untracked files.
# Null delimiters safely handle spaces and unusual filenames.
while IFS= read -r -d '' filePath; do
    # Git stores a symbolic link itself, not the linked file contents.
    if [[ -L "${filePath}" ]]; then
        git add -- "${filePath}"
        printf 'Added symlink: %s\n' "${filePath}"
        ((addedCount += 1))
        continue
    fi

    if [[ ! -f "${filePath}" ]]; then
        continue
    fi

    fileSizeBytes=$(wc -c < "${filePath}")
    fileSizeBytes=${fileSizeBytes//[[:space:]]/}

    if (( fileSizeBytes <= maxSizeBytes )); then
        git add -- "${filePath}"
        printf 'Added: %s\n' "${filePath}"
        ((addedCount += 1))
    else
        fileSizeMb=$(awk \
            -v size="${fileSizeBytes}" \
            'BEGIN { printf "%.2f", size / 1024 / 1024 }')

        printf 'Skipped: %s (%s MiB)\n' "${filePath}" "${fileSizeMb}"
        ((skippedCount += 1))
    fi
done < <(
    git ls-files \
        --modified \
        --others \
        --exclude-standard \
        -z
)

# Stage tracked files that were deleted.
while IFS= read -r -d '' filePath; do
    git add -u -- "${filePath}"
    printf 'Staged deletion: %s\n' "${filePath}"
    ((deletedCount += 1))
done < <(git ls-files --deleted -z)

echo
echo "Staging complete:"
echo "  Added files:       ${addedCount}"
echo "  Staged deletions:  ${deletedCount}"
echo "  Oversized skipped: ${skippedCount}"

echo
git status --short
