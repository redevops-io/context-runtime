"""Windows access-control lists of the directories and files the platform services create.

What is enforced, and what is not (also written down in docs/native-portability.md):

* ``ensure_private_dir`` and ``atomic_write`` do **not** set an ACL on Windows. A new directory or file
  inherits the ACL of its parent. Under ``%LOCALAPPDATA%`` that is the user-profile ACL (the user, SYSTEM
  and Administrators), which is the intended default location.
* These tests read the effective DACL (``GetNamedSecurityInfoW`` -> SDDL, so the check does not depend on the
  Windows display language) and assert that the code adds **no explicit access entry** and grants nothing to
  Everyone, Users or Authenticated Users that the parent did not already grant.
* They do not prove that a location chosen by the operator (for example ``CONTEXT_RUNTIME_HOME=C:\\shared``)
  is private: it is exactly as private as that location's ACL.

The SDDL parser is tested on every OS; the ACL reads run on Windows only.
"""

from __future__ import annotations

import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from context_runtime.platform_services import atomic_write, ensure_private_dir

BROAD = {"WD", "BU", "AU", "S-1-1-0", "S-1-5-11", "S-1-5-32-545"}     # Everyone, Users, Authenticated Users
_WRITE_TOKENS = {"FA", "FW", "GA", "GW", "WD", "WO", "SD", "DT"}
_WRITE_MASK = 0x2 | 0x4 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000
_ACE = re.compile(r"\(([^)]*)\)")


def parse_sddl(sddl: str) -> list[dict]:
    """The access entries of the DACL part of an SDDL string."""
    dacl = sddl.split("D:", 1)[1] if "D:" in sddl else ""
    dacl = dacl.split("S:", 1)[0]
    out = []
    for body in _ACE.findall(dacl):
        f = body.split(";")
        if len(f) < 6:
            continue
        out.append({"type": f[0], "flags": f[1], "rights": f[2], "sid": f[5],
                    "inherited": "ID" in re.findall("..", f[1])})
    return out


def grants_write(rights: str) -> bool:
    if rights.lower().startswith("0x"):
        return bool(int(rights, 16) & _WRITE_MASK)
    return any(t in _WRITE_TOKENS for t in re.findall("..", rights))


def broad_allows(aces: list[dict]) -> list[dict]:
    return [a for a in aces if a["type"] == "A" and a["sid"] in BROAD]


# ───────────────────────── the parser (every OS) ─────────────────────────

SAMPLE = ("O:BAG:S-1-5-21-1-2-3-513D:PAI(A;OICIID;FA;;;SY)(A;OICIID;FA;;;BA)"
          "(A;OICIID;0x1301bf;;;S-1-5-21-1-2-3-1001)(A;;FR;;;BU)(A;OICI;GR;;;WD)(D;;FA;;;AU)")


def test_sddl_parser_reads_types_flags_rights_and_sids():
    aces = parse_sddl(SAMPLE)
    assert [a["sid"] for a in aces] == ["SY", "BA", "S-1-5-21-1-2-3-1001", "BU", "WD", "AU"]
    assert [a["inherited"] for a in aces] == [True, True, True, False, False, False]
    assert [a["sid"] for a in broad_allows(aces)] == ["BU", "WD"]       # the deny entry for AU is not an allow


def test_write_detection_covers_tokens_and_masks():
    assert grants_write("FA") and grants_write("GW") and grants_write("0x1301bf")
    assert not grants_write("FR") and not grants_write("GR") and not grants_write("0x1200a9")


# ───────────────────────── real ACLs (Windows only) ─────────────────────────

def _sddl_of(path: Path) -> str:
    import ctypes
    from ctypes import wintypes
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    kern = ctypes.WinDLL("kernel32", use_last_error=True)
    get = adv.GetNamedSecurityInfoW
    get.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get.restype = wintypes.DWORD
    conv = adv.ConvertSecurityDescriptorToStringSecurityDescriptorW
    conv.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR),
                     ctypes.c_void_p]
    conv.restype = wintypes.BOOL
    free = kern.LocalFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_void_p
    sd, dacl = ctypes.c_void_p(), ctypes.c_void_p()
    SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, SDDL_REVISION_1 = 1, 4, 1
    rc = get(str(path), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, None, None, ctypes.byref(dacl), None,
             ctypes.byref(sd))
    assert rc == 0, f"GetNamedSecurityInfoW failed: {rc}"
    try:
        text = wintypes.LPWSTR()
        assert conv(sd, SDDL_REVISION_1, DACL_SECURITY_INFORMATION, ctypes.byref(text), None)
        try:
            return text.value
        finally:
            free(text)
    finally:
        free(sd)


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows ACLs (not applicable to POSIX permission bits)")


@windows_only
def test_created_directory_and_file_add_no_explicit_entries_and_no_broad_grants(tmp_path):
    parent_sddl = _sddl_of(tmp_path)
    d = ensure_private_dir(tmp_path / "cr" / "sub")
    f = d / "state.json"
    atomic_write(f, "{}")
    parent_broad = {(a["sid"], a["rights"]) for a in broad_allows(parse_sddl(parent_sddl))}
    for path in (tmp_path / "cr", d, f):
        sddl = _sddl_of(path)
        aces = parse_sddl(sddl)
        assert aces, sddl
        explicit = [a for a in aces if not a["inherited"]]
        assert explicit == [], f"{path} has explicit entries the code did not intend: {sddl}"
        extra = [a for a in broad_allows(aces) if (a["sid"], a["rights"]) not in parent_broad]
        assert extra == [], f"{path} grants more than its parent to a broad group: {sddl} (parent {parent_sddl})"


@windows_only
def test_default_user_profile_location_grants_nothing_to_broad_groups():
    import os
    base = Path(os.environ["LOCALAPPDATA"]) / f"context-runtime-acl-test-{uuid.uuid4().hex[:8]}"
    try:
        d = ensure_private_dir(base / "state")
        atomic_write(d / "state.json", "{}")
        for path in (d, d / "state.json"):
            sddl = _sddl_of(path)
            print(f"SDDL {path}: {sddl}")              # kept in the log as evidence of the effective ACL
            assert broad_allows(parse_sddl(sddl)) == [], f"{path}: {sddl}"
    finally:
        shutil.rmtree(base, ignore_errors=True)
