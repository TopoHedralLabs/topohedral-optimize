#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-local-test}"
TEST_BRANCH="docs-local-build"
WORKTREE_PATH="/tmp/docs-local-build"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_PATH" 2>/dev/null || true
    git -C "$REPO_ROOT" branch -D "$TEST_BRANCH" 2>/dev/null || true
}
trap cleanup EXIT

# Remove stale test branch if it exists from a previous run
git -C "$REPO_ROOT" branch -D "$TEST_BRANCH" 2>/dev/null || true

echo "==> Building cargo doc..."
cd "$REPO_ROOT"
RUSTDOCFLAGS="--html-in-header $(pwd)/docs/rustdoc-html/custom-header.html" cargo doc --no-deps

echo "==> Installing Python dependencies..."
pip install -r "$REPO_ROOT/docs/website/requirements.txt" -q

echo "==> Deploying docs with mike (local branch: $TEST_BRANCH, version: $VERSION)..."
cd "$REPO_ROOT/docs/website"
mike deploy --branch "$TEST_BRANCH" --update-aliases "$VERSION" latest
mike set-default --branch "$TEST_BRANCH" latest

echo "==> Injecting API docs into $VERSION/api/ and latest/api/..."
CRATE_NAME=$(grep '^name' "$REPO_ROOT/Cargo.toml" | head -1 | sed 's/.*= *//' | tr -d '"' | tr '-' '_')
API_INDEX="<meta http-equiv='refresh' content='0;url=${CRATE_NAME}/index.html'>"

git -C "$REPO_ROOT" worktree add "$WORKTREE_PATH" "$TEST_BRANCH"
mkdir -p "$WORKTREE_PATH/$VERSION/api" "$WORKTREE_PATH/latest/api"
cp -r "$REPO_ROOT/target/doc/." "$WORKTREE_PATH/$VERSION/api/"
cp -r "$REPO_ROOT/target/doc/." "$WORKTREE_PATH/latest/api/"
echo "$API_INDEX" > "$WORKTREE_PATH/$VERSION/api/index.html"
echo "$API_INDEX" > "$WORKTREE_PATH/latest/api/index.html"
git -C "$WORKTREE_PATH" add .
git -C "$WORKTREE_PATH" commit -m "Local: inject API docs"
git -C "$REPO_ROOT" worktree remove "$WORKTREE_PATH"

echo "==> Serving at http://localhost:8000 (Ctrl+C to stop)"
cd "$REPO_ROOT/docs/website"
mike serve --branch "$TEST_BRANCH"
