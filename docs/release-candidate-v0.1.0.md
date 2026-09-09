# v0.1.0 release-candidate checklist

PR #36 is a release-engineering gate only. It does not authorize tagging or publishing.

- [x] Post-#35 `main` CI and LLM provider contracts passed on `783ef9686d5eca7988ee592fab425862825bdc16`.
- [x] Codex P1 reproduction: retargeted public alias cannot acknowledge a different session's commit.
- [x] Codex P1 reproduction: replacing the canonical session directory cannot redirect checkpoint/CAS.
- [ ] Wheel and sdist build successfully and pass `twine check` on the exact PR head.
- [ ] Built wheel installs in clean Linux and Windows environments.
- [ ] Core wheel install does not pull the MCP SDK.
- [ ] `contextmesh --help` runs from the installed wheel.
- [ ] `contextmesh-graph[mcp]` installs independently on Linux and Windows.
- [x] Distribution name `contextmesh-graph` checked against the real PyPI index (not
      assumed from the import name): `contextmesh` is already an unrelated published
      package, so the distribution is named `contextmesh-graph` while the import
      namespace and console scripts stay `contextmesh` / `contextmesh-mcp`.
- [ ] `contextmesh-mcp --help` and launcher imports run from the installed wheel.
- [ ] Installed MCP launcher can create and restore a persistent session across fresh processes.
- [x] Package/repository metadata points to `Deep-Sixed/Context-Mesh`.
- [x] Release notes and security-reporting guidance are present.
- [ ] Existing full CI and LLM provider contracts pass on the exact PR head.
- [ ] Final P0/P1-only audit passes on the exact merged release-candidate commit.

Tagging `v0.1.0` and publishing packages remain separate explicit gates.
