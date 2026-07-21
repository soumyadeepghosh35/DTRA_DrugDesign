#!/usr/bin/env bash

set -uo pipefail

rootDir="."
maxSizeMb="${2:-60}"
commitMessage="${3:-Update files}"
remoteName="${REMOTE_NAME:-origin}"

maxSizeBytes=$((maxSizeMb * 1024 * 1024))

rootDir=$(cd "${rootDir}" 2>/dev/null && pwd -P) || {
    echo "Error: Cannot access root directory: ${rootDir}"
    exit 1
}

# Discover all normal repositories and Git worktrees.
mapfile -d '' gitMarkers < <(
    find "${rootDir}" -name .git -print0 -prune
)

repoDirs=()

for gitMarker in "${gitMarkers[@]}"; do
    repoDirs+=("${gitMarker%/.git}")
done

if (( ${#repoDirs[@]} == 0 )); then
    echo "No Git repositories found under: ${rootDir}"
    exit 0
fi

isInsideNestedRepo() {
    local currentRepo="$1"
    local fullPath="$2"
    local candidateRepo

    for candidateRepo in "${repoDirs[@]}"; do
        [[ "${candidateRepo}" == "${currentRepo}" ]] && continue

        if [[ "${candidateRepo}" == "${currentRepo}/"* ]]; then
            if [[ "${fullPath}" == "${candidateRepo}" ||
                  "${fullPath}" == "${candidateRepo}/"* ]]; then
                return 0
            fi
        fi
    done

    return 1
}

processRepo() {
    local repoDir="$1"
    local branchName
    local filePath
    local fullPath
    local fileSizeBytes
    local fileSizeMb
    local upstreamBranch

    local addedCount=0
    local skippedCount=0
    local deletedCount=0

    echo
    echo "============================================================"
    echo "Repository: ${repoDir}"
    echo "============================================================"

    branchName=$(
        git -C "${repoDir}" symbolic-ref --quiet --short HEAD 2>/dev/null
    ) || true

    if [[ -z "${branchName}" ]]; then
        echo "Skipped: repository is in detached HEAD state."
        return
    fi

    if ! git -C "${repoDir}" remote get-url "${remoteName}" >/dev/null 2>&1; then
        echo "Skipped: remote '${remoteName}' is not configured."
        return
    fi

    # Clear the staging area without changing working-tree files.
    # This ensures previously staged oversized files are removed.
    if git -C "${repoDir}" rev-parse --verify HEAD >/dev/null 2>&1; then
        if ! git -C "${repoDir}" reset --quiet; then
            echo "Error: Could not reset staging area."
            return
        fi
    else
        git -C "${repoDir}" rm \
            --cached \
            --recursive \
            --ignore-unmatch \
            . >/dev/null 2>&1 || true
    fi

    # Stage modified tracked files and untracked files.
    while IFS= read -r -d '' filePath; do
        fullPath="${repoDir}/${filePath}"

        # Do not stage one nested repository into another.
        if isInsideNestedRepo "${repoDir}" "${fullPath}"; then
            printf 'Skipped nested repository: %s\n' "${filePath}"
            continue
        fi

        # Git stores the symlink itself, not the target contents.
        if [[ -L "${fullPath}" ]]; then
            if git -C "${repoDir}" add -- "${filePath}"; then
                printf 'Added symlink: %s\n' "${filePath}"
                ((addedCount += 1))
            fi

            continue
        fi

        [[ -f "${fullPath}" ]] || continue

        fileSizeBytes=$(wc -c < "${fullPath}")
        fileSizeBytes=${fileSizeBytes//[[:space:]]/}

        if (( fileSizeBytes <= maxSizeBytes )); then
            if git -C "${repoDir}" add -- "${filePath}"; then
                printf 'Added: %s\n' "${filePath}"
                ((addedCount += 1))
            fi
        else
            fileSizeMb=$(awk \
                -v size="${fileSizeBytes}" \
                'BEGIN { printf "%.2f", size / 1024 / 1024 }')

            printf \
                'Skipped oversized file: %s (%s MiB)\n' \
                "${filePath}" \
                "${fileSizeMb}"

            ((skippedCount += 1))
        fi
    done < <(
        git -C "${repoDir}" ls-files \
            --modified \
            --others \
            --exclude-standard \
            -z
    )

    # Stage tracked-file deletions.
    while IFS= read -r -d '' filePath; do
        fullPath="${repoDir}/${filePath}"

        if isInsideNestedRepo "${repoDir}" "${fullPath}"; then
            continue
        fi

        if git -C "${repoDir}" add -u -- "${filePath}"; then
            printf 'Staged deletion: %s\n' "${filePath}"
            ((deletedCount += 1))
        fi
    done < <(
        git -C "${repoDir}" ls-files --deleted -z
    )

    echo
    echo "Staging summary:"
    echo "  Added:            ${addedCount}"
    echo "  Deleted:          ${deletedCount}"
    echo "  Oversized skipped: ${skippedCount}"

    if git -C "${repoDir}" diff --cached --quiet --ignore-submodules --; then
        echo "No eligible changes to commit."
        return
    fi

    echo
    git -C "${repoDir}" status --short

    echo
    echo "Creating commit..."

    if ! git -C "${repoDir}" commit -m "${commitMessage}"; then
        echo "Error: Commit failed for ${repoDir}"
        return
    fi

    upstreamBranch=$(
        git -C "${repoDir}" rev-parse \
            --abbrev-ref \
            --symbolic-full-name \
            '@{upstream}' 2>/dev/null
    ) || true

    echo "Pushing branch: ${branchName}"

    if [[ -n "${upstreamBranch}" ]]; then
        if git -C "${repoDir}" push; then
            echo "Push completed."
        else
            echo "Error: Push failed for ${repoDir}"
        fi
    else
        if git -C "${repoDir}" push \
            --set-upstream \
            "${remoteName}" \
            "${branchName}"; then
            echo "Push completed and upstream configured."
        else
            echo "Error: Push failed for ${repoDir}"
        fi
    fi
}

for repoDir in "${repoDirs[@]}"; do
    processRepo "${repoDir}"
done

echo
echo "Finished processing all repositories."
