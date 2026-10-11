"""Build hook: ship GRAPH.md inside the ``contextmesh`` package.

GRAPH.md stays at the repository root because it is the project's schema
document. Package data that points outside a package lands loose in
site-packages, where any other distribution's GRAPH.md can overwrite or delete
it, so the wheel build copies the file into ``contextmesh/`` instead, which is
where ``contextmesh.ontology`` looks first. Everything else is in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithOntology(build_py):
    def run(self):
        super().run()
        # An editable install builds nothing and reads the root file in place.
        if not self.editable_mode:
            self.copy_file("GRAPH.md", str(Path(self.build_lib, "contextmesh", "GRAPH.md")))


setup(cmdclass={"build_py": BuildPyWithOntology})
