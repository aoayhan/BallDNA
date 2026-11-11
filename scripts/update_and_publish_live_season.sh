#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
python_bin="${BALLDNA_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ${python_bin}. Create .venv and install requirements-live.txt first." >&2
  exit 1
fi

git -C "${repo_root}" pull --ff-only origin main
"${python_bin}" "${repo_root}/scripts/update_live_season.py"
live_assets=(
  data/live/player_style_features.parquet
  data/live/player_style_embeddings.parquet
  data/live/siamese_offensive_embeddings.parquet
  data/live/metadata.json
)
git -C "${repo_root}" add -- "${live_assets[@]}"

if git -C "${repo_root}" diff --cached --quiet -- "${live_assets[@]}"; then
  echo "Live-season snapshot is unchanged."
else
  git -C "${repo_root}" commit -m "Refresh live NBA season data" -- "${live_assets[@]}"
fi

# A failed upload can leave a committed snapshot waiting even when no files changed.
for attempt in 1 2 3; do
  if git -C "${repo_root}" push origin HEAD:main; then
    echo "Live-season snapshot published successfully."
    exit 0
  fi
  if (( attempt < 3 )); then
    echo "Upload failed (attempt ${attempt}/3); retrying in 5 seconds." >&2
    sleep 5
  fi
done

echo "Upload failed after 3 attempts; the snapshot remains committed locally for retry." >&2
exit 1
