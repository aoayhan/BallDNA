"""Exercise the scheduled publisher without contacting NBA.com or GitHub."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize(
    ("failures", "unchanged", "expected_pushes", "expected_exit"),
    [(1, False, 2, 0), (99, False, 3, 1), (1, True, 2, 0)],
)
def test_publisher_retries_and_keeps_pending_uploads(
    tmp_path: Path,
    failures: int,
    unchanged: bool,
    expected_pushes: int,
    expected_exit: int,
) -> None:
    repo = tmp_path / "updater with spaces"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "update_and_publish_live_season.sh"
    shutil.copyfile(Path(__file__).parents[1] / "scripts" / script.name, script)
    executables = tmp_path / "bin"
    executables.mkdir()
    for name in ("python", "sleep"):
        executable = executables / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    fake_git = executables / "git"
    fake_git.write_text(
        '''#!/bin/sh
echo "$*" >> "$LIVE_TEST_CALLS"
case "$*" in
  *" diff --cached --quiet "*) exit "$LIVE_TEST_DIFF_EXIT" ;;
  *" push origin HEAD:main")
    count=0
    if [ -f "$LIVE_TEST_PUSH_COUNT" ]; then count=$(cat "$LIVE_TEST_PUSH_COUNT"); fi
    count=$((count + 1))
    echo "$count" > "$LIVE_TEST_PUSH_COUNT"
    if [ "$count" -le "$LIVE_TEST_FAILURES" ]; then exit 1; fi
    ;;
esac
exit 0
'''
    )
    fake_git.chmod(0o755)
    calls = tmp_path / "calls"
    push_count = tmp_path / "push_count"
    result = subprocess.run(
        ["/bin/zsh", str(script)],
        env={
            **os.environ,
            "PATH": f"{executables}{os.pathsep}{os.environ['PATH']}",
            "BALLDNA_PYTHON": str(executables / "python"),
            "LIVE_TEST_CALLS": str(calls),
            "LIVE_TEST_PUSH_COUNT": str(push_count),
            "LIVE_TEST_DIFF_EXIT": "0" if unchanged else "1",
            "LIVE_TEST_FAILURES": str(failures),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected_exit, result.stderr
    assert int(push_count.read_text()) == expected_pushes
    assert calls.read_text().count('commit -m Refresh live NBA season data') == int(not unchanged)
    if expected_exit:
        assert "snapshot remains committed locally" in result.stderr
