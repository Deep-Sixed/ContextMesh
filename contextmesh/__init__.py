"""ContextMesh — a graph that remembers why things are connected.

    from contextmesh import Pipeline, Walker, Resolver, documents

    pipeline = Pipeline()
    pipeline.build(documents())
    walker = Walker(pipeline.graph, pipeline.resolver)
    print(walker.ask("Why did the Index Builder run out of memory?").path())

The pieces:

- ``ontology``    GRAPH.md parsed into the schema every write is checked against
- ``graph``       typed nodes and typed edges, with no untyped code path,
                  plus the versioned snapshot the graph saves and reloads as
- ``resolve``     entity resolution — one id per real-world thing
- ``pipeline``    CHUNK → EXTRACT → RESOLVE → LINK → EMBED → PRUNE
- ``traverse``    walks that return the evidence path, and classify the failures
- ``assumptions`` versioned assumptions and selective invalidation
- ``evidence``    controlled observation intake; data enters without a verdict
- ``execute``     re-runs exactly the work an invalidation knocked down
- ``llm``         fail-closed provider adapters; model proposals are not verdicts
- ``decisions``   append-only decision history
- ``health``      the conditions that quietly make a graph useless
- ``metrics``     the dashboard payload, computed from the live graph
"""

from .assumptions import AssumptionError, AssumptionLedger, InvalidationReport
from .corpus import documents
from .decisions import DecisionLog, DecisionRecord
from .evidence import (
    EvidenceConflictError,
    EvidenceIntake,
    EvidenceIntakeError,
    EvidenceReceipt,
    submit_evidence,
)
from .execute import (
    AuditContext,
    ExecutionError,
    RunContext,
    RunLedger,
    Runner,
    RunReport,
    Task,
    TaskState,
    Verdict,
)
from .graph import (
    SNAPSHOT_SCHEMA,
    SNAPSHOT_VERSION,
    ContextGraph,
    SnapshotError,
)
from .health import Signal, check, report
from .llm import (
    LIVE_PROVIDERS,
    AuditProposal,
    LLMClient,
    LLMConfig,
    LLMConfigurationError,
    LLMError,
    LLMProvenance,
    LLMProviderError,
    LLMResponseError,
    LLMResult,
    LLMSchemaError,
    LLMTransportError,
    TokenUsage,
    make_audit_proposer,
    make_worker,
    propose_audit,
)
from .metrics import snapshot
from .model import (
    Assumption,
    AssumptionStatus,
    Edge,
    EdgeType,
    Node,
    NodeType,
    Provenance,
)
from .ontology import ONTOLOGY, Ontology, OntologyError
from .pipeline import BuildReport, Document, Pipeline
from .resolve import ResolutionRecord, Resolver
from .telemetry import TelemetryProjection, project_telemetry
from .traverse import DeadEnd, Walk, Walker

__version__ = "0.1.1"

__all__ = [
    "LIVE_PROVIDERS",
    "ONTOLOGY",
    "Assumption",
    "AssumptionError",
    "AssumptionLedger",
    "AssumptionStatus",
    "AuditContext",
    "AuditProposal",
    "BuildReport",
    "ContextGraph",
    "DeadEnd",
    "DecisionLog",
    "DecisionRecord",
    "Document",
    "Edge",
    "EdgeType",
    "EvidenceConflictError",
    "EvidenceIntake",
    "EvidenceIntakeError",
    "EvidenceReceipt",
    "ExecutionError",
    "InvalidationReport",
    "LLMClient",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMError",
    "LLMProvenance",
    "LLMProviderError",
    "LLMResponseError",
    "LLMResult",
    "LLMSchemaError",
    "LLMTransportError",
    "Node",
    "NodeType",
    "Ontology",
    "OntologyError",
    "Pipeline",
    "Provenance",
    "ResolutionRecord",
    "Resolver",
    "RunContext",
    "RunLedger",
    "RunReport",
    "Runner",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_VERSION",
    "Signal",
    "SnapshotError",
    "Task",
    "TaskState",
    "TelemetryProjection",
    "TokenUsage",
    "Verdict",
    "Walk",
    "Walker",
    "check",
    "documents",
    "make_audit_proposer",
    "make_worker",
    "project_telemetry",
    "propose_audit",
    "report",
    "snapshot",
    "submit_evidence",
]
