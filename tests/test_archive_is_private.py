"""What compaction archives is whole tool output; on a shared host it must not be readable by other accounts."""
import os
import stat
import sys

import pytest

from context_runtime.compression.transcript import Archive

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX file modes")


def test_a_directory_the_archive_creates_and_every_file_in_it_are_private(tmp_path):
    old = os.umask(0o022)                                  # the usual umask: 0755 directories, 0644 files
    try:
        a = Archive(directory=tmp_path / "new" / "archive")
        aid = a.add("AWS_SECRET_ACCESS_KEY=example printed by a tool", kind="output", tool="shell", turn=1)
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(a.directory).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(a.entries[aid].path).st_mode) == 0o600
    assert a.get(aid).startswith("AWS_SECRET_ACCESS_KEY")


def test_an_existing_directory_keeps_its_mode_and_its_new_files_are_still_private(tmp_path):
    d = tmp_path / "shared"
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)
    a = Archive(directory=d)
    aid = a.add("text", kind="output", tool="t", turn=1)
    assert stat.S_IMODE(os.stat(d).st_mode) == 0o755 and stat.S_IMODE(os.stat(a.entries[aid].path).st_mode) == 0o600
