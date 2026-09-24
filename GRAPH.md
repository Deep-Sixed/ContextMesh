# GRAPH.md has moved

The graph contract now lives at [`contextmesh/GRAPH.md`](contextmesh/GRAPH.md).

It moved inside the package because it is not only documentation: `contextmesh/ontology.py`
parses it at import time, so it has to ship inside the package that reads it. This file is a
pointer for old links and is not read by anything.
