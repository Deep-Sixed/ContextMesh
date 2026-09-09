# ContextMesh v0.1.1

`v0.1.1` is a release-engineering patch for the first PyPI publication of ContextMesh.
Runtime graph, persistence, MCP authority, and LLM behavior are unchanged from `v0.1.0`.

## Changed

- The PyPI distribution is `contextmesh-graph` because the `contextmesh` distribution name is already owned by an unrelated project.
- The Python import namespace remains `contextmesh`.
- The console scripts remain `contextmesh` and `contextmesh-mcp`.
- PyPI publication uses GitHub Actions Trusted Publishing through OIDC; no long-lived PyPI API token is stored in the repository.
- Publication is triggered only by a published GitHub release whose tag exactly matches the package version.

## Install

```bash
pip install contextmesh-graph
pip install 'contextmesh-graph[mcp]'
```

The core distribution supports Python 3.9+ with no runtime dependencies. The optional MCP
extra requires Python 3.10+.

# ContextMesh v0.1.0

`v0.1.0` is the first public technical release of ContextMesh. It is intentionally not
a 1.0 stability claim.

## Included

- Typed graph/ontology enforcement for entities, claims, sources, decisions, assumptions, and evidence.
- Readable graph traversal with typed dead ends, provenance, health signals, and temporal reconstruction.
- Durable generation-based sessions with crash-safe publication, stale-writer protection, and restart recovery.
- Selective assumption invalidation, execution-plan persistence, repair/resume, and append-only run history.
- MCP read tools plus narrow controlled writes backed by deployment-owned registries.
- Native structured-output adapters with bounded request/response sizes and redirect refusal.
- Windows, Linux, and macOS CI coverage, including cross-process persistence contracts.

## Release boundary

The core package supports Python 3.9+ with no runtime dependencies. The optional MCP
extra requires Python 3.10+ and is installed with `contextmesh-graph[mcp]`. The stdio MCP
server is a single-session-per-process deployment model, not a shared multi-tenant
service.

Package publication was a separate explicit gate and is opened by the `v0.1.1` release.
