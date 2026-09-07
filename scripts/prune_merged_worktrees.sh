#!/bin/bash
# Remove git worktrees whose work has already merged into origin/dev.
#
# A worktree is removed only when ALL of these hold:
#   1. its HEAD is an ancestor of origin/dev (merge commits make this a
#      reliable merged-content check for this repository),
#   2. its tree is clean (no staged, unstaged, or untracked changes),
#   3. it has not been touched for at least MIN_AGE_HOURS (default 48),
#      so an agent's in-flight checkout is never yanked mid-run.
#
# Branches and commits are never deleted; `git worktree remove` only
# deletes the working directory. Pass --dry-run to preview.
#
# Usage: scripts/prune_merged_worktrees.sh [--dry-run] [repo-path]

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi
REPO="${1:-$(git rev-parse --show-toplevel 2>/dev/null || true)}"
if [ -z "$REPO" ]; then
    echo "error: not inside a git repository and no repo path given" >&2
    exit 2
fi
REPO=$(cd "$REPO" && pwd)
MIN_AGE_HOURS="${MIN_AGE_HOURS:-48}"
NOW_EPOCH=$(date +%s)

git -C "$REPO" fetch -q origin dev

# Porcelain lines are "worktree <path>"; strip the prefix with sed so paths
# containing spaces survive intact.
MAIN_WT=$(git -C "$REPO" worktree list --porcelain | sed -n 's/^worktree //p' | head -n 1)

git -C "$REPO" worktree list --porcelain | sed -n 's/^worktree //p' | while IFS= read -r wt_path; do
    [ "$wt_path" = "$MAIN_WT" ] && continue
    # Never remove the worktree this script is running from: when invoked
    # inside a linked worktree, $REPO resolves to that worktree, and
    # removing one's own working directory mid-run is never intended.
    [ "$wt_path" = "$REPO" ] && continue
    if ! wt_head=$(git -C "$wt_path" rev-parse HEAD 2>/dev/null); then
        # Directory or linkage already gone; `git worktree prune` handles it.
        continue
    fi
    if [ -n "$(git -C "$wt_path" status --porcelain 2>/dev/null)" ]; then
        echo "keep (dirty):    $wt_path"
        continue
    fi
    if ! git -C "$REPO" merge-base --is-ancestor "$wt_head" origin/dev; then
        echo "keep (unmerged): $wt_path"
        continue
    fi
    # Activity signal: the linked worktree's admin dir (its real git-dir)
    # has entries (HEAD, index, logs) whose mtimes move on checkouts,
    # commits, and resets; the .git linkage file alone is written once at
    # creation and never again, so it cannot distinguish an active checkout
    # from an abandoned one. Take the newest of the linkage file and the
    # admin dir's HEAD/index.
    wt_gitdir=$(git -C "$wt_path" rev-parse --absolute-git-dir 2>/dev/null || echo "")
    wt_mtime=0
    for probe in "$wt_path/.git" "$wt_gitdir/HEAD" "$wt_gitdir/index"; do
        [ -e "$probe" ] || continue
        # GNU stat first (-c %Y); BSD stat second (-f %m). The reverse
        # order is a trap: on GNU, -f is filesystem mode and %m is the
        # mount point, which SUCCEEDS with a non-mtime string.
        probe_mtime=$(stat -c %Y "$probe" 2>/dev/null || stat -f %m "$probe" 2>/dev/null || echo 0)
        if [ "$probe_mtime" -gt "$wt_mtime" ]; then
            wt_mtime=$probe_mtime
        fi
    done
    age_hours=$(( (NOW_EPOCH - wt_mtime) / 3600 ))
    if [ "$age_hours" -lt "$MIN_AGE_HOURS" ]; then
        echo "keep (recent, ${age_hours}h): $wt_path"
        continue
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "would remove:    $wt_path"
    else
        git -C "$REPO" worktree remove "$wt_path"
        echo "removed:         $wt_path"
    fi
done

if [ "$DRY_RUN" -eq 0 ]; then
    git -C "$REPO" worktree prune
fi
git -C "$REPO" worktree list
