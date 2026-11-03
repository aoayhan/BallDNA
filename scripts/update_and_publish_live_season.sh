#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h}"
python_bin="${BALLDNA_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ${python_bin}. Create .venv and install requirements-live.txt first." >&2
  exit 1
fi

"${python_bin}" "${repo_root}/scripts/update_live_season.py"
git -C "${repo_root}" add \
  data/live/player_style_features.parquet \
  data/live/player_style_embeddings.parquet \
  data/live/siamese_offensive_embeddings.parquet \
  data/live/metadata.json

if git -C "${repo_root}" diff --cached --quiet; then
  echo "Live-season snapshot is unchanged."
  exit 0
fi

git -C "${repo_root}" commit -m "Refresh live NBA season data"
git -C "${repo_root}" push origin HEAD:main
