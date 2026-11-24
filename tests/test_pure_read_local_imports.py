"""A ``pure_read`` claim is checked against the callable's source. An ``import`` inside the callable is part of
that source, and what it imports is not: these are the callables that used to be accepted as reads because the
name they call was bound two lines above, and the ones that still are, because the module is inert."""
import pytest

from context_runtime.tools.actions import INERT_MODULES, opaque_callees, pure_read, resolve_action
from context_runtime.tools.base import RegistryConfigurationError

CLAIM = "formats its arguments into a string and touches nothing outside the process"


def _shell(args):
    import subprocess
    return subprocess.getoutput(args["cmd"])


def _post(args):
    import httpx
    return httpx.request("POST", args["url"], json=args).text


def _sql(args):
    import sqlite3
    sqlite3.connect(args["db"]).execute("DELETE FROM t")
    return "ok"


def _kill(args):
    import os
    os.kill(int(args["pid"]), 9)


def _reflective(args):
    return getattr(__import__("requests"), "po" + "st")(args["url"])


def _computed(args):
    table = {"go": print}
    return table[args["verb"]](args)


def _path(args):
    from pathlib import Path
    Path(args["p"]).open("w", encoding="utf-8").close()


def _alias(args):
    import subprocess as json
    return json.getoutput(args["cmd"])


def _from_alias(args):
    from subprocess import getoutput as dumps
    return dumps(args["cmd"])


def _submodule(args):
    import urllib.parse
    return urllib.request.build_opener().open(args["url"])


def _relative(args):
    from . import json
    return json.dumps(args)


@pytest.mark.parametrize("fn", [_shell, _post, _sql, _kill, _reflective, _computed, _path, _alias, _from_alias, _submodule, _relative],
                         ids=lambda f: f.__name__)
def test_a_callable_that_reaches_outside_through_a_local_import_or_a_computed_callee_is_not_a_read(fn):
    assert opaque_callees(fn), "nothing opaque was found in a callable that leaves the process"
    with pytest.raises(RegistryConfigurationError):
        resolve_action(shipped="read", injected=fn, declared=pure_read(CLAIM), tool="t.x")


def _json_local(args):
    import json
    return json.dumps(sorted(args))


def _from_inert(args):
    from urllib.parse import quote
    import re as rx
    return quote(rx.sub(r"\s+", " ", str(args.get("q", ""))).strip().lower())


def _aliased_inert(args):
    import urllib.parse as up
    return up.urlencode({k: str(v) for k, v in args.items()})


@pytest.mark.parametrize("fn", [_json_local, _from_inert, _aliased_inert], ids=lambda f: f.__name__)
def test_a_callable_that_imports_only_inert_modules_is_still_a_read(fn):
    assert opaque_callees(fn) == ()
    assert resolve_action(shipped="read", injected=fn, declared=pure_read(CLAIM), tool="t.x") == "read"


def test_the_inert_modules_are_the_ones_the_inert_roots_name():
    assert "subprocess" not in INERT_MODULES and "os" not in INERT_MODULES and "urllib.parse" in INERT_MODULES
