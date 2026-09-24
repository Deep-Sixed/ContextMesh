"""The version a user sees at import time is the version that was released.

0.1.1 shipped to PyPI while both packages still reported 0.1.0: the release
workflow checks the tag against pyproject.toml and never read __version__.
This fails on the version-bump pull request instead of after publication.
"""

import re
import unittest
from pathlib import Path

import contextmesh
import contextmesh_mcp

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _project_version() -> str:
    # tomllib is 3.11+ and the suite runs on 3.9, so read the one key directly:
    # the first top-level `version = "..."` inside the [project] table.
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    assert table is not None, "pyproject.toml has no [project] table"
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', table.group(1), re.M)
    assert match is not None, "[project] has no static version"
    return match.group(1)


class VersionAgreementTest(unittest.TestCase):
    def test_contextmesh_reports_the_packaged_version(self):
        self.assertEqual(contextmesh.__version__, _project_version())

    def test_contextmesh_mcp_reports_the_packaged_version(self):
        self.assertEqual(contextmesh_mcp.__version__, _project_version())


if __name__ == "__main__":
    unittest.main()
