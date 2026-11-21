"""Command line: build the graph, ask it things, and see what it is hiding.

    python -m contextmesh demo          # the worked example, end to end
    python -m contextmesh ask "..."     # one question, one readable path
    python -m contextmesh health        # graph health signals
    python -m contextmesh export        # write the dashboard's data
    python -m contextmesh invalidate    # break an assumption, show the fallout
    python -m contextmesh execute       # break one, then re-run only what fell
    python -m contextmesh ingest DIR    # build from your own Markdown/text files
    python -m contextmesh ask --folder DIR "..."   # ask your files, not the demo
"""

from __future__ import annotations

import argparse
import builtins
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .demo import run
from .execute import Event
from .execute import demo as execute_demo
from .health import check
from .model import AssumptionStatus, NodeType

DEFAULT_EXPORT = Path(__file__).resolve().parent.parent / "dashboard" / "data" / "mesh.json"
DASHBOARD_HTML = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
DEFAULT_SUITE = Path(__file__).resolve().parent.parent / "bench" / "contextmesh-docs.json"

RULE = "─" * 78

ASCII_FALLBACKS = str.maketrans({
    "─": "-",
    "·": "*",
    "→": "->",
    "—": "-",
    "✓": "+",
    "✗": "x",
    "…": "...",
    "█": "#",
})


def _stream_supports_presentation(stream: Any) -> bool:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "─ · → — ✓ ✗ … █".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _safe_text(value: Any, stream: Any) -> str:
    text = str(value)
    if _stream_supports_presentation(stream):
        return text
    return text.translate(ASCII_FALLBACKS)


def print(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    stream = sys.stdout if file is None else file
    safe_values = [_safe_text(value, stream) for value in values]
    builtins.print(*safe_values, sep=sep, end=end, file=stream, flush=flush)


def _h(title: str) -> None:
    print(f"\n{title}\n{RULE}")


def _fmt(n: float) -> str:
    return f"{n:,}"


# ── commands ─────────────────────────────────────────────────────────────
def cmd_demo(args: argparse.Namespace) -> int:
    result = run(rounds=args.rounds)
    graph, walker, build = result.graph, result.walker, result.build

    _h("BUILD PATH · what earns a node in the graph")
    for stage in build.stages:
        bar = "█" * min(40, stage.admitted // max(1, build.spans_in // 40 or 1))
        print(
            f"  {stage.name:<8} {stage.caption:<22} "
            f"admitted {stage.admitted:>5}  dropped {stage.dropped:>5}  {bar}"
        )
    print(
        f"\n  spans in {_fmt(build.spans_in)} · dropped at resolve "
        f"{_fmt(build.dropped_at_resolve)} · committed → walkable "
        f"{_fmt(build.committed_walkable)}"
    )

    _h("CONTEXT MESH · the live graph")
    for name, count in graph.type_counts().items():
        if count:
            print(f"  {name:<12} {count:>4}")
    print("\n  edges by type")
    for name, count in graph.edge_counts().items():
        if count:
            print(f"  {name:<14} {count:>4}")
    print(f"\n  untyped edges: {graph.untyped_edges}")

    _h("WALKING · every answer walks a path you can read")
    print(
        f"  {len(walker.walks)} walks · {walker.resolved_rate:.1%} resolved · "
        f"median {walker.median_hops()} hops · "
        f"{walker.token_saving():.1%} fewer tokens than flat top-k"
    )
    for walk in [w for w in walker.walks if w.resolved][:3]:
        print(f"\n  Q: {walk.question}")
        for line in walk.path().splitlines():
            print(f"    {line}")

    _h("DEAD ENDS · walks that ended nowhere")
    for reason, count in walker.dead_end_ledger().items():
        print(f"  {reason:<20} {count:>4}")
    for walk in [w for w in walker.walks if w.dead_end][:2]:
        print(f"\n  Q: {walk.question}\n    {walk.dead_end.value}: {walk.detail}")

    _h("DECISION HISTORY · append-only")
    for record in result.decisions.records:
        marker = "→ supersedes " + record.supersedes if record.supersedes else ""
        print(f"  [{record.at_build}] {record.title} {marker}")
        print(f"        {record.rationale}")

    _h("SELECTIVE INVALIDATION · what fell, and what did not")
    inv = result.invalidation
    if inv is None:
        print("  nothing was rejected in this run")
    else:
        print(f"  rejected: {inv.statement}")
        print(f"  blast radius: {inv.blast_radius} node(s); preserved: {len(inv.preserved)}")
        for node_id in inv.invalidated:
            print(f"    ✗ {inv.why(node_id)}")
        if inv.replacement_id:
            replacement = graph.assumptions[inv.replacement_id]
            print(f"  replacement: {replacement.statement}")

    _h("GRAPH HEALTH")
    for signal in check(graph, result.resolver, walker):
        mark = {"error": "!!", "warn": " !", "info": "  "}[signal.severity]
        print(f"  {mark} {signal.kind:<24} {signal.count:>4}  {signal.detail}")
    return 0


def _source_graph(args: argparse.Namespace) -> Any:
    """The demo, or a folder build when ``--folder`` names one.

    Returns something with ``graph``, ``resolver`` and ``walker``.
    """
    folder = getattr(args, "folder", None)
    if folder is None:
        return run(rounds=args.rounds)
    from .ingest import IngestError, build_folder

    try:
        return build_folder(folder, max_total_bytes=_total_bytes(args))
    except IngestError as exc:
        raise SystemExit(f"contextmesh: {exc}") from exc


def _total_bytes(args: argparse.Namespace) -> int:
    from .ingest import DEFAULT_MAX_TOTAL_BYTES

    mb = getattr(args, "max_total_mb", None)
    return DEFAULT_MAX_TOTAL_BYTES if mb is None else int(mb * 1_000_000)


def _megabytes(value: str) -> float:
    try:
        mb = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number of megabytes")
    if not mb > 0:
        raise argparse.ArgumentTypeError("the total size limit must be more than 0 MB")
    return mb


def _where(node: Any) -> str:
    """Where a claim sits in a real file, and how far to trust the lift."""
    location = node.attrs.get("location")
    if not location:
        return ""
    verified = "verified" if node.attrs.get("verified") else "unverified"
    return f"   [{location} · {node.attrs.get('confidence', '?')} confidence · {verified}]"


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import IngestError, build_folder

    try:
        built = build_folder(args.folder, max_total_bytes=_total_bytes(args))
    except IngestError as exc:
        print(f"contextmesh: {exc}", file=sys.stderr)
        return 2
    summary = built.summary()
    if args.json:
        summary["skipped"] = [{"path": s.path, "reason": s.reason} for s in built.skipped]
        builtins.print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    _h(f"INGEST {summary['root']}")
    print(f"  ingested at    {summary['ingested_at']}  (wall clock; not stored in the graph)")
    print(f"  files read     {summary['files_read']}")
    print(f"  files skipped  {summary['files_skipped']}")
    for item in built.skipped[:10]:
        print(f"                 · {item.path}: {item.reason}")
    if len(built.skipped) > 10:
        print(f"                 · … and {len(built.skipped) - 10} more (--json lists all)")
    print(f"  text read      {summary['bytes_read'] / 1e6:.1f} MB of a "
          f"{summary['max_total_bytes'] / 1e6:g} MB limit")
    from .ingest import total_limit_reason

    capped = sum(
        1 for s in built.skipped if s.reason == total_limit_reason(built.max_total_bytes)
    )
    if capped:
        print(f"  ! {capped} files were left out to stay under the total size limit."
              " Point it at a narrower folder, or raise the limit with --max-total-mb"
              " (each MB costs roughly 350 MB of memory and 15 seconds).")
    print(f"  nodes live     {_fmt(summary['nodes_live'])}  {summary['by_type']}")
    print(f"  edges live     {_fmt(summary['edges_live'])}")
    conf = summary["claim_confidence"]
    print(f"  claims         high {conf['high']} · medium {conf['medium']} · low {conf['low']}"
          "  (extraction confidence; none verified)")
    _h("BUILD PATH")
    for stage in built.report.stages:
        print(f"  {stage.name:<8} {stage.caption:<20} admitted {stage.admitted:>6}"
              f"  dropped {stage.dropped:>6}")
    print("\nAsk it something:  python -m contextmesh ask --folder "
          f"{args.folder} \"<question>\"")
    print("Keep it:           python -m contextmesh_mcp --folder "
          f"{args.folder} --save <dir>")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import load_suite, render
    from .bench import run as run_bench

    suite_path = Path(args.suite)
    if not suite_path.is_file():
        print(f"contextmesh: no benchmark suite at {suite_path}; pass --suite FILE",
              file=sys.stderr)
        return 2
    result = run_bench(load_suite(suite_path), base=suite_path.parent)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    print(render(result))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    result = _source_graph(args)
    for question in args.question:
        walk = result.walker.ask(question)
        print(f"\nQ: {question}")
        if walk.resolved:
            answer = result.graph.node(walk.answer_id)
            print(f"A: {answer.label}")
            if _where(answer):
                print(_where(answer))
            print(f"   ({walk.hops} hops, score {walk.score:.2f}, "
                  f"{walk.tokens_walked} tokens against {walk.tokens_flat} flat)")
            print("\n" + walk.path())
            if walk.evidence:
                print("\n   evidence:")
                for eid in walk.evidence:
                    print(f"     · {result.graph.node(eid).label[:72]}")
        else:
            print(f"A: dead end — {walk.dead_end.value}")
            print(f"   {walk.detail}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    result = _source_graph(args)
    signals = check(result.graph, result.resolver, result.walker)
    for signal in signals:
        mark = {"error": "!!", "warn": " !", "info": "  "}[signal.severity]
        print(f"{mark} {signal.kind} ({signal.count})")
        print(f"   {signal.detail}")
        print(f"   remedy: {signal.remedy}")
        for item in signal.items[:5]:
            print(f"     · {item[:80]}")
        print()
    return 0


def cmd_invalidate(args: argparse.Namespace) -> int:
    result = run(rounds=args.rounds)
    inv = result.invalidation
    if inv is None:
        print("nothing rejected")
        return 1
    print(f"rejected assumption: {inv.statement}\n")
    print(f"INVALIDATED ({inv.blast_radius})")
    for node_id in inv.invalidated:
        print(f"  ✗ {result.graph.node(node_id).label[:70]}")
        print(f"      because: {inv.why(node_id)}")
    print(f"\nPRESERVED ({len(inv.preserved)}) — untouched by the rejection")
    for node_id in inv.preserved[:8]:
        print(f"  ✓ {result.graph.node(node_id).label[:70]}")
    if len(inv.preserved) > 8:
        print(f"  … and {len(inv.preserved) - 8} more")

    print("\nASSUMPTION LINEAGE")
    for assumption in result.graph.assumptions.values():
        flag = {
            AssumptionStatus.ACTIVE: "active",
            AssumptionStatus.REJECTED: "REJECTED",
            AssumptionStatus.SUPERSEDED: "superseded",
        }[assumption.status]
        print(f"  v{assumption.version} [{flag}] {assumption.statement}")
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    """Selective invalidation is only half the claim; this is the other half."""
    result = execute_demo()
    runner = result.runner
    graph = runner.graph

    _h("PLAN · tasks, and the ground each one currently stands on")
    for task in runner.tasks:
        needs = f"  needs {', '.join(task.needs)}" if task.needs else ""
        print(f"  {task.name:<12} {task.title}{needs}")
        print(f"  {'':<12} assumes: {task.assumes}")
    if runner.unaudited:
        print(f"\n  unaudited: {', '.join(runner.unaudited)}")

    for report in runner.rounds:
        _h(f"ROUND {report.round} · {len(report.executed)} executed, "
           f"{len(report.cached)} cached")
        for index, layer in enumerate(report.layers, start=1):
            print(f"  layer {index}: {', '.join(layer)}")
        for name in report.cached:
            print(f"  cached  {name} — untouched, previous result stands")
        for name, reason in report.blocked.items():
            print(f"  blocked {name} — {reason}")
        for name, reason in report.failed.items():
            print(f"  FAILED  {name} — {reason}")

    _h("THE GROUND MOVED · discovered by an auditor, not announced")
    for entry in runner.ledger.entries:
        if entry.event is Event.DISPROVED:
            print(f"  {entry.task}: {entry.detail}")
    for inv in result.invalidations:
        print(f"\n  rejected: {inv.statement}")
        print(f"  blast radius {inv.blast_radius}, preserved {len(inv.preserved)}")
        for node_id in inv.invalidated:
            print(f"    ✗ {graph.node(node_id).label[:64]}")
            print(f"        {inv.why(node_id)}")

    _h("ARTEFACTS · rebuilt, abandoned, or never touched")
    for node in graph.nodes.values():
        if node.type is not NodeType.ENTITY:
            continue
        mark = "live" if node.live else "GONE"
        print(f"  [{mark}] {node.label}")

    _h("ASSUMPTION LINEAGE · a rejection is never edited away")
    for assumption in graph.assumptions.values():
        print(f"  v{assumption.version} [{assumption.status.value}] {assumption.statement}")

    _h("LEDGER · append-only, and checkably so")
    for entry in runner.ledger.entries:
        print(f"  {entry.seq:>3} r{entry.round} {entry.event.value:<12} "
              f"{entry.task:<12} {entry.detail[:60]}")
    print(f"\n  head {runner.ledger.short_head} · chain intact: {runner.ledger.verify()}")

    _h("RECEIPT · the whole event, read back off the ledger alone")
    for receipt in runner.ledger.receipts():
        print(f"  ground     {receipt['assumption']}")
        print(f"  disproved  by {receipt['disproved_by']}, evidence {receipt['evidence_id']}")
        print(f"  reason     {receipt['reason']}")
        print(f"  fell       {len(receipt['invalidated'])}")
        for node_id, chain in sorted(receipt["invalidated"].items()):
            print(f"    ✗ {graph.node(node_id).label[:52]}")
            print(f"        {' → '.join(chain)}")
        print(f"  stood      {len(receipt['preserved'])} node(s) untouched")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    result = run(rounds=args.rounds)
    payload = result.payload()
    out = Path(args.out) if args.out else DEFAULT_EXPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")

    if args.inline and DASHBOARD_HTML.exists():
        _inline(DASHBOARD_HTML, payload)
        print(f"inlined into {DASHBOARD_HTML}")
    return 0


def _json_for_html(payload: Dict[str, Any]) -> str:
    """Serialize JSON so HTML parsing cannot terminate its script-data block.

    ``type=application/json`` changes how JavaScript treats the contents, not
    how the HTML parser finds ``</script>``. Any graph-derived string can
    therefore close the element if a raw ``<`` is emitted. Escaping the three
    HTML-significant ASCII characters as JSON unicode escapes preserves the
    value returned by ``JSON.parse`` while making a tag opener impossible.
    ``json.dumps`` keeps its default ``ensure_ascii=True``, so U+2028/U+2029
    and other non-ASCII characters stay escaped exactly as before.
    """
    blob = json.dumps(payload, separators=(",", ":"))
    return (
        blob.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _inline(html_path: Path, payload: Dict[str, Any]) -> None:
    """Replace the dashboard's embedded snapshot so it works from file://."""
    html = html_path.read_text(encoding="utf-8")
    blob = _json_for_html(payload)
    pattern = re.compile(
        r'(<script id="mesh-data" type="application/json">)(.*?)(</script>)',
        re.S,
    )
    if not pattern.search(html):
        raise SystemExit(f"{html_path} has no <script id='mesh-data'> block")
    rewritten = pattern.sub(lambda m: m.group(1) + blob + m.group(3), html)
    html_path.write_text(rewritten, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="contextmesh",
        description="A graph that remembers why things are connected.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="walk rounds to run before reporting (default: 8)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("demo", help="build, walk, break an assumption, report").set_defaults(
        func=cmd_demo
    )
    folder_help = "build from Markdown/text files under DIR instead of the demo"
    cap_help = (
        "most text to read from --folder, in MB (default 4); each MB costs "
        "roughly 350 MB of memory and 15 seconds to build"
    )
    ask = sub.add_parser("ask", help="ask the graph a question")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--folder", metavar="DIR", help=folder_help)
    ask.add_argument("--max-total-mb", type=_megabytes, metavar="MB", help=cap_help)
    ask.set_defaults(func=cmd_ask)
    health = sub.add_parser("health", help="graph health signals")
    health.add_argument("--folder", metavar="DIR", help=folder_help)
    health.add_argument("--max-total-mb", type=_megabytes, metavar="MB", help=cap_help)
    health.set_defaults(func=cmd_health)
    bench = sub.add_parser(
        "bench", help="score answers on labelled questions: graph walk vs BM25 vs reading all"
    )
    bench.add_argument("--suite", default=str(DEFAULT_SUITE),
                       help="benchmark suite JSON (default: this repo's docs suite)")
    bench.add_argument("--json", metavar="FILE", help="also write the full result as JSON")
    bench.set_defaults(func=cmd_bench)
    ingest = sub.add_parser("ingest", help="build a graph from your own files and report it")
    ingest.add_argument("folder", metavar="DIR")
    ingest.add_argument("--json", action="store_true", help="machine-readable summary")
    ingest.add_argument(
        "--max-total-mb", type=_megabytes, metavar="MB",
        help=cap_help.replace("from --folder", "from DIR"),
    )
    ingest.set_defaults(func=cmd_ingest)
    sub.add_parser(
        "invalidate", help="reject an assumption and show the blast radius"
    ).set_defaults(func=cmd_invalidate)
    sub.add_parser(
        "execute", help="break an assumption, then re-run only what fell over"
    ).set_defaults(func=cmd_execute)
    export = sub.add_parser("export", help="write the dashboard payload")
    export.add_argument(
        "--rounds",
        type=int,
        default=45,
        help="walk rounds; the default fills the dashboard's 1,008-cell grid",
    )
    export.add_argument("--out", help=f"output path (default: {DEFAULT_EXPORT})")
    export.add_argument(
        "--inline",
        action="store_true",
        help="also embed the payload in dashboard/index.html",
    )
    export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
