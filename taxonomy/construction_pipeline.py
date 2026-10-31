from __future__ import annotations

from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
import logging
import os
import time
from typing import Any, Callable, Iterable

from tqdm import tqdm

from .cycle_utils import fix_cycles_collect_unresolved, interactive_manual_edge_removal
from .embeddings import find_similar_terms
from .graph_utils import (
    connect_subroot_to_domain,
    construct_hierarchy,
    filter_nodes_in_edges,
    get_subhierarchy_edges_from_children,
    get_subhierarchy_edges_from_parents,
    is_child,
    print_hierarchy,
    remove_transitive_edges,
)
from .io_utils import StreamToLogger, load_from_pickle, save_to_pickle
from .llm_taxonomy import (
    check_subsumption_relation,
    find_nearest_parents,
    generate_taxonomy,
    resolve_conflict_mapping,
    resolve_transitive_mapping,
)
from .manual_cycles import (
    FileCycleDecisionStore,
    ManualCycleDecisionRequired,
    ManualCycleResolutionAborted,
)
from .merge_utils import merge_similar_nodes
from .telemetry import TaxonomyRunRecorder, atomic_write_json


@dataclass
class TaxonomyPassState:
    """Mutable state saved after each topic group."""

    edges: list[tuple[str, str]] = field(default_factory=list)
    term_mapping: dict[str, str] = field(default_factory=dict)
    evaluated_pairs: dict[tuple[str, str], str] = field(default_factory=dict)
    topic_ids_completed: list[Any] = field(default_factory=list)
    linkage_roots: list[str] | None = None
    subsumption_evaluated_pairs: dict[tuple[str, str], str] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalTaxonomyArtifact:
    """A completed hierarchy constructed within one topic's local context."""

    topic_id: Any
    terms: list[str]
    edges: list[tuple[str, str]]
    term_to_node: dict[str, str]


@dataclass(frozen=True)
class TaxonomyStateFiles:
    """Pickle filenames used by a taxonomy construction pass."""

    edges: str
    term_mapping: str
    evaluated_pairs: str
    topic_ids_completed: str
    linkage_roots: str | None = None
    legacy_names: dict[str, tuple[str, ...]] = field(default_factory=dict)
    subsumption_evaluated_pairs: str | None = None


def configure_taxonomy_logger(
    log_dir: str,
    domain: str,
    pass_name: str,
    experiment_name: str,
    logger_name: str = "taxonomy_logger",
) -> tuple[logging.Logger, str]:
    """Configure the file logger for one pipeline pass."""

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"{domain}_{pass_name}_taxonomy_"
        f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    print(f"Logging to: {log_file}")
    return logger, log_file


def add_term_mapping(
    mapping_store: dict[str, str],
    original: str,
    candidate: str,
    chat_model: Any,
    telemetry: TaxonomyRunRecorder | None = None,
) -> str:
    """Add a canonical term mapping, resolving conflicts with the LLM."""

    original = original.lower().strip()
    candidate = candidate.lower().strip()

    # Skip empty/whitespace-only names. An empty canonical name would otherwise
    # turn "" into a shared alias hub: everything ever mapped to "" resolves to
    # whatever "" last pointed at (via the mapping_store[rejected] = selected
    # line below), silently corrupting unrelated terms.
    if not original:
        return original
    if not candidate:
        return resolve_transitive_mapping(original, mapping_store)

    candidate = resolve_transitive_mapping(candidate, mapping_store)
    current = resolve_transitive_mapping(original, mapping_store)

    if current == candidate:
        mapping_store[original] = candidate
        return candidate

    if original in mapping_store:
        print("already in mapping:", original, current)
        print("new mapping:", original, candidate)
        print("please resolve the conflict:")
        selected = resolve_conflict_mapping(
            original,
            current,
            candidate,
            chat_model,
            telemetry=telemetry,
        )
    else:
        selected = candidate

    rejected = candidate if selected == current else current
    mapping_store[original] = selected
    if rejected and rejected != selected:
        mapping_store[rejected] = selected
    return selected


def normalize_edges(
    edges: Iterable[tuple[str, str]],
    term_mapping: dict[str, str],
) -> list[tuple[str, str]]:
    """Lowercase, resolve mappings, remove self-edges, and deduplicate edges."""

    normalized = set()
    for child, parent in edges:
        resolved_child = resolve_transitive_mapping(child.lower(), term_mapping)
        resolved_parent = resolve_transitive_mapping(parent.lower(), term_mapping)
        if resolved_child != resolved_parent:
            normalized.add((resolved_child, resolved_parent))
    return list(normalized)


def edge_nodes(edges: Iterable[tuple[str, str]]) -> list[str]:
    """Return all unique nodes appearing in an edge list."""

    return list({item for edge in edges for item in edge})


def clean_taxonomy_edges(
    edges: Iterable[tuple[str, str]],
    chat_model: Any,
    domain: str,
    *,
    term_mapping: dict[str, str] | None = None,
    verbose: bool = True,
    connect_to_domain: bool = True,
    telemetry: TaxonomyRunRecorder | None = None,
    cleaning_context: str | None = None,
    cycle_decision_store: FileCycleDecisionStore | None = None,
) -> list[tuple[str, str]]:
    """Resolve mappings, break cycles, remove transitive edges, and connect roots."""

    cleaned = list(edges)
    if term_mapping is not None:
        cleaned = normalize_edges(cleaned, term_mapping)
    else:
        cleaned = list({(child.lower(), parent.lower()) for child, parent in cleaned if child.lower() != parent.lower()})

    telemetry_context = (
        telemetry.context(cleaning_context=cleaning_context)
        if telemetry and cleaning_context
        else nullcontext()
    )
    with telemetry_context:
        cleaned, _removed_edges, unresolved_cycles, _rounds = fix_cycles_collect_unresolved(
            cleaned,
            chat_model,
            verbose=verbose,
            telemetry=telemetry,
        )
        if unresolved_cycles:
            if cycle_decision_store is not None:
                cleaned = cycle_decision_store.resolve_or_pause(
                    cleaned,
                    unresolved_cycles,
                    telemetry=telemetry,
                )
            else:
                cleaned = interactive_manual_edge_removal(
                    cleaned,
                    unresolved_cycles,
                    domain=domain,
                    telemetry=telemetry,
                )
        else:
            print("no unresolved cycles")

    cleaned = remove_transitive_edges(cleaned, verbose=verbose)
    if connect_to_domain:
        cleaned = connect_subroot_to_domain(cleaned, domain)
    return cleaned


def merge_new_nodes_into_mapping(
    evaluated_pairs: dict[tuple[str, str], str],
    existing_edges: Iterable[tuple[str, str]],
    new_edges: Iterable[tuple[str, str]],
    term_mapping: dict[str, str],
    chat_model: Any,
    domain: str,
    similarity_threshold: float,
    *,
    verbose: bool = False,
    embedding_model: str = 'all-mpnet-base-v2',
    telemetry: TaxonomyRunRecorder | None = None,
) -> dict[tuple[str, str], str]:
    """Merge similar nodes across the current accumulated taxonomy and new topic taxonomy."""

    two_set_nodes = list(set(edge_nodes(existing_edges)).union(edge_nodes(new_edges)))
    similar_node_mapping, evaluated_pairs = merge_similar_nodes(
        evaluated_pairs,
        two_set_nodes,
        chat_model,
        domain,
        similarity_threshold=similarity_threshold,
        verbose=verbose,
        embedding_model=embedding_model,
        telemetry=telemetry,
    )
    for original, canonical in similar_node_mapping.items():
        add_term_mapping(
            term_mapping,
            original,
            canonical,
            chat_model,
            telemetry=telemetry,
        )
    return evaluated_pairs


def generate_topic_edges(
    terms: list[str],
    chat_model: Any,
    cfg: Any,
    term_mapping: dict[str, str],
    *,
    max_retries: int | None = None,
    verbose: bool = True,
    telemetry: TaxonomyRunRecorder | None = None,
) -> list[tuple[str, str]]:
    """Generate, normalize, repair, and root-connect taxonomy edges for one topic."""

    domain = cfg.paths.domain
    edges, mapping = generate_taxonomy(
        terms,
        chat_model,
        domain,
        verbose=verbose,
        max_retries=cfg.llm.max_retries if max_retries is None else max_retries,
        telemetry=telemetry,
    )

    edges = [(child.lower(), parent.lower()) for child, parent in edges]
    edges = connect_subroot_to_domain(edges, domain)

    mapping = {original.lower(): canonical.lower() for original, canonical in mapping.items()}
    for original, canonical in mapping.items():
        add_term_mapping(
            term_mapping,
            original,
            canonical,
            chat_model,
            telemetry=telemetry,
        )

    resolved_terms = [
        resolve_transitive_mapping(term.lower(), term_mapping)
        for term in terms
    ]
    edges = normalize_edges(edges, term_mapping)
    included_terms = set(edge_nodes(edges))
    missing_terms = set(resolved_terms) - included_terms
    print("number of missing terms:", len(missing_terms), missing_terms)

    if missing_terms and included_terms:
        mapping2 = find_similar_terms(
            missing_terms,
            included_terms,
            threshold=cfg.embedding.similarity_threshold,
            embedding_model=cfg.embedding.model_name,
        )
        for original, canonical in mapping2.items():
            add_term_mapping(
                term_mapping,
                original,
                canonical,
                chat_model,
                telemetry=telemetry,
            )

        edges = normalize_edges(edges, term_mapping)
        included_terms = set(edge_nodes(edges))
        missing_terms = [
            term for term in missing_terms
            if resolve_transitive_mapping(term, term_mapping) not in included_terms
        ]
        print("number of missing terms after merging by similarity:", len(missing_terms))
        edges = edges + [(term, domain.lower()) for term in missing_terms]

    print("Number of edges before removing transitive edges and cycles:", len(edges))
    edges = clean_taxonomy_edges(
        edges,
        chat_model,
        domain,
        term_mapping=None,
        verbose=verbose,
        connect_to_domain=True,
        telemetry=telemetry,
        cleaning_context="local_taxonomy_cleanup",
    )
    print("Number of edges after removing transitive edges:", len(edges))
    return edges


def construct_local_taxonomy(
    topic: dict[str, Any],
    chat_model: Any,
    cfg: Any,
    *,
    max_retries: int | None = None,
    verbose: bool = True,
    telemetry: TaxonomyRunRecorder | None = None,
) -> LocalTaxonomyArtifact:
    """Construct one topic hierarchy without consulting global merge state."""

    terms = list(topic["terms"])
    local_mapping: dict[str, str] = {}
    edges = generate_topic_edges(
        terms,
        chat_model,
        cfg,
        local_mapping,
        max_retries=max_retries,
        verbose=verbose,
        telemetry=telemetry,
    )
    term_to_node = {
        term.lower(): resolve_transitive_mapping(term.lower(), local_mapping)
        for term in terms
    }
    return LocalTaxonomyArtifact(
        topic_id=topic["topic_id"],
        terms=terms,
        edges=edges,
        term_to_node=term_to_node,
    )


def fine_grained_state_files() -> TaxonomyStateFiles:
    """Return pickle filenames for the fine-grained pass."""

    return TaxonomyStateFiles(
        edges="fine_grained_edges.pkl",
        term_mapping="fine_grained_term_mapping.pkl",
        evaluated_pairs="fine_grained_evaluated_pairs.pkl",
        topic_ids_completed="fine_grained_topic_ids_completed.pkl",
        subsumption_evaluated_pairs="fine_grained_subsumption_evaluated_pairs.pkl",
        legacy_names={
            "evaluated_pairs": ("evaluated_pairs.pkl",),
            "topic_ids_completed": ("topic_ids_completed.pkl",),
        },
    )


def fine_grained_merge_state_files() -> TaxonomyStateFiles:
    """Return pickle filenames for the explicit fine-grained merge phase."""

    return TaxonomyStateFiles(
        edges="fine_grained_merge_edges.pkl",
        term_mapping="fine_grained_merge_term_mapping.pkl",
        evaluated_pairs="fine_grained_merge_evaluated_pairs.pkl",
        topic_ids_completed="fine_grained_merge_topic_ids_completed.pkl",
        subsumption_evaluated_pairs=(
            "fine_grained_merge_subsumption_evaluated_pairs.pkl"
        ),
    )


def broad_state_files(output_suffix: str = "") -> TaxonomyStateFiles:
    """Return pickle filenames for one broad/coarse pass."""

    pass_prefix = f"coarse{output_suffix}" if output_suffix else "coarse"
    legacy_names = {
        "edges": (f"existing_edges{output_suffix}.pkl",),
        "term_mapping": (f"term_mapping{output_suffix}.pkl",),
        "linkage_roots": (f"s_f_roots{output_suffix}.pkl",),
    }
    if output_suffix:
        legacy_names["topic_ids_completed"] = (f"topic_ids_completed{output_suffix}.pkl",)

    return TaxonomyStateFiles(
        edges=f"{pass_prefix}_edges.pkl",
        term_mapping=f"{pass_prefix}_term_mapping.pkl",
        evaluated_pairs=f"{pass_prefix}_evaluated_pairs.pkl",
        topic_ids_completed=f"{pass_prefix}_topic_ids_completed.pkl",
        subsumption_evaluated_pairs=(
            f"{pass_prefix}_subsumption_evaluated_pairs.pkl"
        ),
        linkage_roots=f"{pass_prefix}_linkage_roots.pkl",
        legacy_names=legacy_names,
    )


def _state_path(cfg: Any, filename: str) -> str:
    return os.path.join(cfg.paths.file_path, cfg.paths.experiment_name, filename)


_FINE_GRAINED_ARTIFACT_NAMESPACE = "fine_grained"
_LOCAL_TAXONOMY_SUBDIR = "fine_grained_local_taxonomies"
_BROAD_LOCAL_TAXONOMY_SUBDIR = "broad_local_taxonomies"
_LOCAL_TAXONOMY_MANIFEST = "manifest.pkl"
_LOCAL_TAXONOMY_SCHEMA_VERSION = 2


def _broad_pass_key(output_suffix: str = "") -> str:
    return f"coarse{output_suffix}" if output_suffix else "coarse"


def _broad_artifact_namespace(output_suffix: str = "") -> str:
    return f"broad_{_broad_pass_key(output_suffix)}"


def _local_taxonomy_dir(
    cfg: Any,
    artifact_namespace: str = _FINE_GRAINED_ARTIFACT_NAMESPACE,
) -> str:
    if artifact_namespace == _FINE_GRAINED_ARTIFACT_NAMESPACE:
        relative_dir = _LOCAL_TAXONOMY_SUBDIR
    else:
        relative_dir = os.path.join(
            _BROAD_LOCAL_TAXONOMY_SUBDIR,
            artifact_namespace,
        )
    return os.path.join(
        cfg.paths.file_path,
        cfg.paths.experiment_name,
        relative_dir,
    )


def _taxonomy_log_dir(cfg: Any) -> str:
    return os.path.join(
        cfg.paths.log_subdir,
        cfg.paths.domain,
        cfg.paths.experiment_name,
    )


def _integration_dir(cfg: Any) -> str:
    return os.path.join(
        _taxonomy_log_dir(cfg),
        "fine_grained_integration",
    )


def _local_telemetry_dir(
    cfg: Any,
    artifact_namespace: str = _FINE_GRAINED_ARTIFACT_NAMESPACE,
) -> str:
    if artifact_namespace == _FINE_GRAINED_ARTIFACT_NAMESPACE:
        stage_dir = "local_construction"
    else:
        stage_dir = f"{artifact_namespace}_local_construction"
    return os.path.join(_taxonomy_log_dir(cfg), stage_dir)


def _broad_integration_dir(cfg: Any, output_suffix: str = "") -> str:
    return os.path.join(
        _taxonomy_log_dir(cfg),
        f"{_broad_artifact_namespace(output_suffix)}_integration",
    )


def _integration_telemetry_dir(cfg: Any) -> str:
    return _integration_dir(cfg)


def _taxonomy_snapshot(
    state: TaxonomyPassState,
    *,
    snapshot_name: str,
    step_index: int,
    topic_id: Any | None,
) -> dict[str, Any]:
    edges = sorted({(child, parent) for child, parent in state.edges})
    nodes = sorted({node for edge in edges for node in edge})
    return {
        "schema_version": 1,
        "snapshot_name": snapshot_name,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "step_index": step_index,
        "topic_id": topic_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": [
            {"child": child, "parent": parent}
            for child, parent in edges
        ],
        "term_mapping": dict(sorted(state.term_mapping.items())),
        "completed_topic_ids": state.topic_ids_completed,
    }


def _save_taxonomy_snapshot(
    cfg: Any,
    state: TaxonomyPassState,
    *,
    snapshot_name: str,
    step_index: int,
    topic_id: Any | None,
) -> str:
    snapshot_dir = os.path.join(_integration_dir(cfg), "snapshots")
    path = os.path.join(snapshot_dir, f"{snapshot_name}.json")
    atomic_write_json(
        _taxonomy_snapshot(
            state,
            snapshot_name=snapshot_name,
            step_index=step_index,
            topic_id=topic_id,
        ),
        path,
    )
    return path


def _atomic_save_to_pickle(obj: Any, path: str) -> None:
    """Write a pickle checkpoint atomically within its destination directory."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp"
    save_to_pickle(obj, temporary_path)
    os.replace(temporary_path, path)


def _new_local_taxonomy_manifest(
    cfg: Any,
    artifact_namespace: str = _FINE_GRAINED_ARTIFACT_NAMESPACE,
) -> dict[str, Any]:
    return {
        "schema_version": _LOCAL_TAXONOMY_SCHEMA_VERSION,
        "artifact_namespace": artifact_namespace,
        "domain": cfg.paths.domain,
        "llm_provider": cfg.llm.provider,
        "llm_model": cfg.llm.model,
        "llm_temperature": cfg.llm.temperature,
        "llm_retry_temperature": cfg.llm.retry_temperature,
        "llm_top_p": cfg.llm.top_p,
        "llm_seed": cfg.llm.seed,
        "llm_max_output_tokens": cfg.llm.max_output_tokens,
        "embedding_model": cfg.embedding.model_name,
        "similarity_threshold": cfg.embedding.similarity_threshold,
        "expected_topic_count": None,
        "complete": False,
        "topics": [],
    }


def _validate_local_taxonomy_manifest(
    manifest: dict[str, Any],
    cfg: Any,
    artifact_namespace: str = _FINE_GRAINED_ARTIFACT_NAMESPACE,
) -> None:
    expected = _new_local_taxonomy_manifest(cfg, artifact_namespace)
    if (
        artifact_namespace != _FINE_GRAINED_ARTIFACT_NAMESPACE
        or "artifact_namespace" in manifest
    ):
        if manifest.get("artifact_namespace") != artifact_namespace:
            raise ValueError(
                "Saved local-taxonomy manifest does not match artifact "
                f"namespace {artifact_namespace!r}."
            )
    for field_name in (
        "schema_version",
        "domain",
        "llm_provider",
        "llm_model",
        "embedding_model",
        "similarity_threshold",
    ):
        if manifest.get(field_name) != expected[field_name]:
            raise ValueError(
                "Saved local-taxonomy manifest does not match the current "
                f"configuration for {field_name!r}: "
                f"{manifest.get(field_name)!r} != {expected[field_name]!r}"
            )
def load_local_taxonomy_artifacts(
    cfg: Any,
    artifact_namespace: str = _FINE_GRAINED_ARTIFACT_NAMESPACE,
) -> list[LocalTaxonomyArtifact]:
    """Load completed local hierarchies in their recorded merge order."""

    artifact_dir = _local_taxonomy_dir(cfg, artifact_namespace)
    manifest_path = os.path.join(artifact_dir, _LOCAL_TAXONOMY_MANIFEST)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No {artifact_namespace!r} local-taxonomy manifest found at "
            f"{manifest_path}. Run local construction first."
        )

    manifest = load_from_pickle(manifest_path)
    _validate_local_taxonomy_manifest(manifest, cfg, artifact_namespace)
    if not manifest.get("complete", False):
        completed_count = len(manifest.get("topics", []))
        expected_count = manifest.get("expected_topic_count")
        raise RuntimeError(
            f"{artifact_namespace!r} local-taxonomy construction is incomplete: "
            f"{completed_count} of {expected_count} topic artifacts are recorded. "
            "Resume local construction before starting the merge phase."
        )
    if len(manifest["topics"]) != manifest["expected_topic_count"]:
        raise ValueError(
            "Completed local-taxonomy manifest has an inconsistent topic count."
        )

    artifacts = []
    for entry in manifest["topics"]:
        artifact_path = os.path.join(artifact_dir, entry["filename"])
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"Missing local-taxonomy artifact for topic "
                f"{entry['topic_id']!r}: {artifact_path}"
            )
        artifact = load_from_pickle(artifact_path)
        if not isinstance(artifact, LocalTaxonomyArtifact):
            raise TypeError(
                f"Invalid local-taxonomy artifact at {artifact_path}: "
                f"expected LocalTaxonomyArtifact, got {type(artifact).__name__}"
            )
        if (
            artifact.topic_id != entry["topic_id"]
            or artifact.terms != entry["terms"]
        ):
            raise ValueError(
                f"Local-taxonomy artifact does not match its manifest entry: "
                f"{artifact_path}"
            )
        artifacts.append(artifact)
    return artifacts


def load_pipeline_state(
    cfg: Any,
    state_files: TaxonomyStateFiles,
    *,
    default_state: TaxonomyPassState | None = None,
) -> TaxonomyPassState:
    """Load pass state from pickle files, falling back to defaults for missing files."""

    state = default_state or TaxonomyPassState()

    def maybe_load(field_name: str, filename: str | None, fallback: Any) -> Any:
        if filename is None:
            return fallback
        for candidate in (filename, *state_files.legacy_names.get(field_name, ())):
            path = _state_path(cfg, candidate)
            if os.path.exists(path):
                return load_from_pickle(path)
        return fallback

    return TaxonomyPassState(
        edges=maybe_load("edges", state_files.edges, state.edges),
        term_mapping=maybe_load("term_mapping", state_files.term_mapping, state.term_mapping),
        evaluated_pairs=maybe_load(
            "evaluated_pairs",
            state_files.evaluated_pairs,
            state.evaluated_pairs,
        ),
        subsumption_evaluated_pairs=maybe_load(
            "subsumption_evaluated_pairs",
            state_files.subsumption_evaluated_pairs,
            state.subsumption_evaluated_pairs,
        ),
        topic_ids_completed=maybe_load(
            "topic_ids_completed",
            state_files.topic_ids_completed,
            state.topic_ids_completed,
        ),
        linkage_roots=maybe_load("linkage_roots", state_files.linkage_roots, state.linkage_roots),
    )


def save_pipeline_state(
    state: TaxonomyPassState,
    cfg: Any,
    state_files: TaxonomyStateFiles,
) -> None:
    """Save pass state using pass-specific pickle filenames."""

    output_dir = os.path.join(cfg.paths.file_path, cfg.paths.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    save_to_pickle(state.edges, os.path.join(output_dir, state_files.edges))
    save_to_pickle(state.term_mapping, os.path.join(output_dir, state_files.term_mapping))
    save_to_pickle(state.evaluated_pairs, os.path.join(output_dir, state_files.evaluated_pairs))
    if state_files.subsumption_evaluated_pairs is not None:
        save_to_pickle(
            state.subsumption_evaluated_pairs,
            os.path.join(output_dir, state_files.subsumption_evaluated_pairs),
        )
    save_to_pickle(
        state.topic_ids_completed,
        os.path.join(output_dir, state_files.topic_ids_completed),
    )
    if state_files.linkage_roots is not None:
        save_to_pickle(state.linkage_roots, os.path.join(output_dir, state_files.linkage_roots))


def _save_subsumption_cache(
    state: TaxonomyPassState,
    cfg: Any,
    state_files: TaxonomyStateFiles,
) -> None:
    """Atomically persist directional ``(child, parent)`` LLM decisions."""

    if state_files.subsumption_evaluated_pairs is None:
        return
    _atomic_save_to_pickle(
        state.subsumption_evaluated_pairs,
        _state_path(cfg, state_files.subsumption_evaluated_pairs),
    )


def _topologically_ready_nodes(
    remaining: set[str],
    children: dict[str, set[str]],
    processed: set[str],
) -> list[str]:
    """Return nodes whose remaining descendants have already been processed."""

    overlap = set(remaining)

    def overlap_descendants(node: str) -> set[str]:
        out = set()
        for child in children.get(node, []):
            if child in overlap:
                out.add(child)
            out |= overlap_descendants(child)
        return out

    return [
        node for node in remaining
        if overlap_descendants(node).issubset(processed)
    ]


def integrate_fine_grained_topic(
    fine_grained_edges: list[tuple[str, str]],
    edges: list[tuple[str, str]],
    chat_model: Any,
    domain: str,
    telemetry: TaxonomyRunRecorder | None = None,
    subsumption_cache: dict[tuple[str, str], str] | None = None,
    persist_subsumption_cache: Callable[[], None] | None = None,
) -> list[tuple[str, str]]:
    """Merge a generated fine-grained subhierarchy into the accumulated taxonomy."""

    edges_copy = edges.copy()
    main_parents, main_children = construct_hierarchy(fine_grained_edges)
    new_sub_parents, new_sub_children = construct_hierarchy(edges)
    overlapped_nodes = (
        (set(main_parents.keys()) | set(main_children.keys()))
        & (set(new_sub_parents.keys()) | set(new_sub_children.keys()))
    ) - {domain.lower()}
    print("Number of overlapped nodes:", len(overlapped_nodes))

    processed = set()
    remaining = set(overlapped_nodes)
    while remaining:
        ready = _topologically_ready_nodes(remaining, new_sub_children, processed)
        print("ready:", ready)
        if not ready:
            print("No ready overlap nodes remain; stopping fine-grained integration loop.")
            break

        target = next(iter(ready))
        target_sub_children = new_sub_children.get(target, [])
        print("target:", target)
        for target_sub_child in target_sub_children:
            target_main_children = main_children.get(target, [])
            if target_sub_child in target_main_children:
                print("target_sub_child is the same as target_main_child, skip")
                continue

            for target_main_child in target_main_children:
                print(
                    "target_sub_child:",
                    target_sub_child,
                    " - target_main_child:",
                    target_main_child,
                )

                rel1 = check_subsumption_relation(
                    target_main_child,
                    target_sub_child,
                    chat_model,
                    verbose=False,
                    telemetry=telemetry,
                    subsumption_cache=subsumption_cache,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                print("rel1:", target_sub_child, "->", target_main_child, ":", rel1)
                if rel1 == "subset":
                    nearest_parents = find_nearest_parents(
                        main_children,
                        target_main_child,
                        target_sub_child,
                        chat_model,
                        verbose=True,
                        telemetry=telemetry,
                        subsumption_cache=subsumption_cache,
                        persist_subsumption_cache=persist_subsumption_cache,
                    )
                    if not nearest_parents:
                        nearest_parents = {target_main_child}
                    print("nearest_parents:", nearest_parents)
                    for nearest_parent in nearest_parents:
                        fine_grained_edges.append((target_sub_child, nearest_parent))
                    target_sub_edges = get_subhierarchy_edges_from_children(
                        new_sub_children,
                        target_sub_child,
                    )
                    fine_grained_edges.extend(target_sub_edges)
                    continue

                rel2 = check_subsumption_relation(
                    target_sub_child,
                    target_main_child,
                    chat_model,
                    verbose=False,
                    telemetry=telemetry,
                    subsumption_cache=subsumption_cache,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                print("rel2:", target_main_child, "->", target_sub_child, ":", rel2)
                if rel2 == "subset":
                    nearest_parents = find_nearest_parents(
                        new_sub_children,
                        target_sub_child,
                        target_main_child,
                        chat_model,
                        verbose=True,
                        telemetry=telemetry,
                        subsumption_cache=subsumption_cache,
                        persist_subsumption_cache=persist_subsumption_cache,
                    )
                    if not nearest_parents:
                        nearest_parents = {target_sub_child}
                    print("nearest_parents:", nearest_parents)
                    for nearest_parent in nearest_parents:
                        fine_grained_edges.append((target_main_child, nearest_parent))
                        target_sub_edges = get_subhierarchy_edges_from_parents(
                            new_sub_parents,
                            nearest_parent,
                        )
                        fine_grained_edges.extend(target_sub_edges)
                else:
                    print(
                        "target_sub_child:",
                        target_sub_child,
                        " - target_main_child:",
                        target_main_child,
                        "no subsumption relation.",
                    )
            main_parents, main_children = construct_hierarchy(fine_grained_edges)

        sub_edges = get_subhierarchy_edges_from_children(new_sub_children, target)
        edges = list(set(edges) - set(sub_edges))
        new_sub_parents, new_sub_children = construct_hierarchy(edges)

        remaining.discard(target)
        processed.add(target)

    fine_grained_edges.extend(edges_copy)
    return list(set(fine_grained_edges))


def integrate_broad_topic(
    existing_edges: list[tuple[str, str]],
    edges: list[tuple[str, str]],
    linkage_roots: list[str],
    term_mapping: dict[str, str],
    chat_model: Any,
    domain: str,
    telemetry: TaxonomyRunRecorder | None = None,
    subsumption_cache: dict[tuple[str, str], str] | None = None,
    persist_subsumption_cache: Callable[[], None] | None = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Merge a generated broad/coarse subhierarchy into the accumulated taxonomy."""

    edges_copy = edges.copy()
    existing_parents, existing_children = construct_hierarchy(existing_edges)
    coarse_parents, coarse_children = construct_hierarchy(edges)

    linkage_roots = list({
        resolve_transitive_mapping(node.lower(), term_mapping)
        for node in linkage_roots
    })
    linkage_nodes, _linkage_edges = filter_nodes_in_edges(linkage_roots, edges)
    print("number of linkage nodes:", len(linkage_nodes))

    processed = set()
    remaining = set(linkage_nodes)
    while remaining:
        ready = _topologically_ready_nodes(remaining, coarse_children, processed)
        print("ready:", ready)
        if not ready:
            print("No ready linkage nodes remain; stopping broad integration loop.")
            break

        target = next(iter(ready))
        print("target:", target)
        c_target_parents = coarse_parents.get(target, [])
        for c_target_parent in c_target_parents:
            if c_target_parent == domain.lower():
                print("c_target_parent is the same as domain, skip")
                edges = list(set(edges) - {(target, c_target_parent)})
                continue

            print("c_target_parent:", c_target_parent)
            c_target_parent_children = existing_children.get(c_target_parent, [])
            if target in c_target_parent_children:
                print("target is already a child of c_target_parent, skip")
                edges = list(set(edges) - {(target, c_target_parent)})
                continue

            for c_target_parent_child in c_target_parent_children:
                print("c_target_parent_child:", c_target_parent_child)

                rel1 = check_subsumption_relation(
                    c_target_parent_child,
                    target,
                    chat_model,
                    verbose=False,
                    telemetry=telemetry,
                    subsumption_cache=subsumption_cache,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                print("rel1:", target, "->", c_target_parent_child, ":", rel1)
                if rel1 == "subset":
                    nearest_parents = find_nearest_parents(
                        existing_children,
                        c_target_parent_child,
                        target,
                        chat_model,
                        verbose=True,
                        telemetry=telemetry,
                        subsumption_cache=subsumption_cache,
                        persist_subsumption_cache=persist_subsumption_cache,
                    )
                    if not nearest_parents:
                        nearest_parents = [c_target_parent_child]
                    for nearest_parent in nearest_parents:
                        existing_edges.append((target, nearest_parent))
                    continue

                rel2 = check_subsumption_relation(
                    target,
                    c_target_parent_child,
                    chat_model,
                    verbose=False,
                    telemetry=telemetry,
                    subsumption_cache=subsumption_cache,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                print("rel2:", c_target_parent_child, "->", target, ":", rel2)
                if rel2 == "subset":
                    nearest_parents = find_nearest_parents(
                        existing_children,
                        target,
                        c_target_parent_child,
                        chat_model,
                        verbose=True,
                        telemetry=telemetry,
                        subsumption_cache=subsumption_cache,
                        persist_subsumption_cache=persist_subsumption_cache,
                    )
                    if not nearest_parents:
                        nearest_parents = [target]
                    for nearest_parent in nearest_parents:
                        existing_edges.append((c_target_parent_child, nearest_parent))
                else:
                    print(
                        "target:",
                        target,
                        " - c_target_parent_child:",
                        c_target_parent_child,
                        "no subsumption relation.",
                    )

            existing_parents, existing_children = construct_hierarchy(existing_edges)
            edges = list(set(edges) - {(target, c_target_parent)})
        coarse_parents, coarse_children = construct_hierarchy(edges)

        if not is_child(target, edges):
            remaining.discard(target)
            processed.add(target)

    existing_edges.extend(edges_copy)
    return list(set(existing_edges)), linkage_roots


def _run_with_logging(func: Any, logging_cfg: Any, pass_name: str, *args: Any, **kwargs: Any) -> Any:
    logger, log_file = configure_taxonomy_logger(
        _taxonomy_log_dir(logging_cfg),
        logging_cfg.paths.domain,
        pass_name,
        logging_cfg.paths.experiment_name,
    )
    telemetry = kwargs.get("telemetry")
    if telemetry is not None:
        telemetry.set_console_log(log_file)
    stdout_logger = StreamToLogger(logger, logging.INFO)
    stderr_logger = StreamToLogger(logger, logging.ERROR)
    with redirect_stdout(stdout_logger), redirect_stderr(stderr_logger):
        return func(*args, **kwargs)


def _sorted_topics(topics: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        topics,
        key=lambda topic: topic.get("n_terms", len(topic.get("terms", []))),
        reverse=True,
    )


def _construct_local_topic_job(
    index: int,
    topic: dict[str, Any],
    chat_model: Any,
    cfg: Any,
    *,
    max_retries: int | None,
    verbose: bool,
    telemetry: TaxonomyRunRecorder | None,
    topic_group_label: str,
) -> dict[str, Any]:
    topic_id = topic["topic_id"]
    terms = list(topic["terms"])
    print(f"Constructing Local {topic_group_label} Topic Group {topic_id}")
    print("number of terms:", len(terms))
    print(terms)

    topic_start_timestamp = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )
    topic_start_perf = time.perf_counter()
    try:
        telemetry_context = (
            telemetry.context(topic_id=topic_id, topic_position=index)
            if telemetry
            else nullcontext()
        )
        with telemetry_context:
            artifact = construct_local_taxonomy(
                topic,
                chat_model,
                cfg,
                max_retries=max_retries,
                verbose=verbose,
                telemetry=telemetry,
            )
        return {
            "index": index,
            "topic_id": topic_id,
            "terms": terms,
            "artifact": artifact,
            "start_timestamp": topic_start_timestamp,
            "start_perf": topic_start_perf,
            "construction_seconds": time.perf_counter() - topic_start_perf,
        }
    except BaseException as exc:
        if telemetry:
            telemetry.record_unit({
                "topic_id": topic_id,
                "position": index,
                "term_count": len(terms),
                "status": "failure",
                "start_timestamp": topic_start_timestamp,
                "end_timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "total_wall_clock_seconds": round(
                    time.perf_counter() - topic_start_perf,
                    6,
                ),
                "error": f"{type(exc).__name__}: {exc}",
                **telemetry.usage_for_context(topic_id=topic_id),
            })
        raise


def _run_local_taxonomy_construction_inner(
    topics: Iterable[dict[str, Any]],
    chat_model: Any,
    cfg: Any,
    *,
    resume: bool,
    max_retries: int | None,
    verbose: bool,
    telemetry: TaxonomyRunRecorder | None,
    llm_concurrency: int,
    artifact_namespace: str,
    topic_group_label: str,
) -> list[LocalTaxonomyArtifact]:
    if llm_concurrency < 1:
        raise ValueError("llm_concurrency must be at least 1")

    sorted_topics = _sorted_topics(topics)
    artifact_dir = _local_taxonomy_dir(cfg, artifact_namespace)
    manifest_path = os.path.join(artifact_dir, _LOCAL_TAXONOMY_MANIFEST)

    if resume and os.path.exists(manifest_path):
        manifest = load_from_pickle(manifest_path)
        _validate_local_taxonomy_manifest(manifest, cfg, artifact_namespace)
        if manifest.get("expected_topic_count") != len(sorted_topics):
            raise ValueError(
                "Saved local-taxonomy manifest topic count does not match "
                f"the current {artifact_namespace!r} topic input."
            )
    else:
        manifest = _new_local_taxonomy_manifest(cfg, artifact_namespace)
        manifest["expected_topic_count"] = len(sorted_topics)
        _atomic_save_to_pickle(manifest, manifest_path)

    entries = manifest["topics"]
    if len(entries) > len(sorted_topics):
        raise ValueError(
            "Saved local-taxonomy manifest contains more topics than the "
            "current fine-grained topic input."
        )

    entries_by_position: dict[int, dict[str, Any]] = {}
    for entry in entries:
        position = entry["position"]
        if position in entries_by_position or not 0 <= position < len(sorted_topics):
            raise ValueError(
                f"Invalid or duplicate manifest position: {position!r}"
            )
        topic = sorted_topics[position]
        expected_filename = f"topic_{position:05d}.pkl"
        if (
            entry["topic_id"] != topic["topic_id"]
            or entry["terms"] != list(topic["terms"])
            or entry["filename"] != expected_filename
        ):
            raise ValueError(
                "Saved local-taxonomy topic order/input does not match "
                f"the current topic at position {position}."
            )
        entries_by_position[position] = entry

    pending: list[tuple[int, dict[str, Any]]] = []
    for index, topic in enumerate(sorted_topics):
        topic_id = topic["topic_id"]
        terms = list(topic["terms"])
        filename = f"topic_{index:05d}.pkl"
        artifact_path = os.path.join(artifact_dir, filename)
        entry = entries_by_position.get(index)

        if entry is not None and os.path.exists(artifact_path):
            artifact = load_from_pickle(artifact_path)
            if (
                not isinstance(artifact, LocalTaxonomyArtifact)
                or artifact.topic_id != topic_id
                or artifact.terms != terms
            ):
                raise ValueError(
                    f"Saved local-taxonomy artifact is invalid for topic "
                    f"{topic_id!r}."
                )
            if telemetry:
                telemetry.record_unit({
                    "topic_id": topic_id,
                    "position": index,
                    "term_count": len(terms),
                    "status": "skipped",
                    "total_wall_clock_seconds": 0.0,
                    "artifact_path": artifact_path,
                })
            print(f"Local taxonomy for {topic_id} completed, skip")
            continue
        pending.append((index, topic))

    if pending:
        manifest["complete"] = False
        _atomic_save_to_pickle(manifest, manifest_path)

    def persist_result(result: dict[str, Any]) -> None:
        index = result["index"]
        topic_id = result["topic_id"]
        terms = result["terms"]
        artifact = result["artifact"]
        filename = f"topic_{index:05d}.pkl"
        artifact_path = os.path.join(artifact_dir, filename)
        persistence_start = time.perf_counter()
        _atomic_save_to_pickle(artifact, artifact_path)

        entry = {
            "position": index,
            "topic_id": topic_id,
            "terms": terms,
            "filename": filename,
        }
        entries_by_position[index] = entry
        manifest["topics"] = [
            entries_by_position[position]
            for position in sorted(entries_by_position)
        ]
        _atomic_save_to_pickle(manifest, manifest_path)
        persistence_seconds = time.perf_counter() - persistence_start

        if telemetry:
            nodes = set(edge_nodes(artifact.edges))
            observed_nodes = set(artifact.term_to_node.values())
            telemetry.record_unit({
                "topic_id": topic_id,
                "position": index,
                "term_count": len(terms),
                "status": "success",
                "start_timestamp": result["start_timestamp"],
                "end_timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
                "construction_seconds": round(
                    result["construction_seconds"],
                    6,
                ),
                "persistence_seconds": round(persistence_seconds, 6),
                "total_wall_clock_seconds": round(
                    time.perf_counter() - result["start_perf"],
                    6,
                ),
                "edge_count": len(artifact.edges),
                "observed_node_count": len(observed_nodes),
                "intermediate_node_count": len(
                    nodes - observed_nodes - {cfg.paths.domain.lower()}
                ),
                "artifact_path": artifact_path,
                **telemetry.usage_for_context(topic_id=topic_id),
            })
        print("completed local taxonomy topic id:", topic_id)

    use_concurrency = (
        cfg.llm.provider.lower() == "vllm"
        and llm_concurrency > 1
        and len(pending) > 1
    )
    if use_concurrency:
        worker_count = min(llm_concurrency, len(pending))
        print(
            f"Constructing {len(pending)} local taxonomies with "
            f"{worker_count} concurrent vLLM workers"
        )
        failures: list[tuple[int, Any, BaseException]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_item = {
                executor.submit(
                    _construct_local_topic_job,
                    index,
                    topic,
                    chat_model,
                    cfg,
                    max_retries=max_retries,
                    verbose=verbose,
                    telemetry=telemetry,
                    topic_group_label=topic_group_label,
                ): (index, topic["topic_id"])
                for index, topic in pending
            }
            for future in tqdm(
                as_completed(future_to_item),
                total=len(future_to_item),
                desc="Constructing local taxonomies",
            ):
                index, topic_id = future_to_item[future]
                try:
                    persist_result(future.result())
                except BaseException as exc:
                    failures.append((index, topic_id, exc))

        if failures:
            manifest["complete"] = False
            _atomic_save_to_pickle(manifest, manifest_path)
            first_index, first_topic_id, first_error = min(
                failures,
                key=lambda item: item[0],
            )
            raise RuntimeError(
                f"Local taxonomy construction failed for "
                f"{len(failures)} topic(s); first failure was topic "
                f"{first_topic_id!r} at position {first_index}."
            ) from first_error
    else:
        for index, topic in tqdm(
            pending,
            desc="Constructing local taxonomies",
        ):
            result = _construct_local_topic_job(
                index,
                topic,
                chat_model,
                cfg,
                max_retries=max_retries,
                verbose=verbose,
                telemetry=telemetry,
                topic_group_label=topic_group_label,
            )
            persist_result(result)

    manifest["complete"] = True
    manifest["topics"] = [
        entries_by_position[position]
        for position in range(len(sorted_topics))
    ]
    _atomic_save_to_pickle(manifest, manifest_path)
    return load_local_taxonomy_artifacts(cfg, artifact_namespace)


def _run_local_construction_stage(
    topics: Iterable[dict[str, Any]],
    chat_model: Any,
    cfg: Any,
    *,
    artifact_namespace: str,
    topic_group_label: str,
    telemetry_stage: str,
    stage_summary_filename: str,
    log_pass_name: str,
    resume: bool,
    max_retries: int | None,
    log_to_file: bool,
    verbose: bool,
    enable_telemetry: bool,
    llm_concurrency: int,
) -> list[LocalTaxonomyArtifact]:
    telemetry = TaxonomyRunRecorder(
        enabled=enable_telemetry,
        telemetry_dir=_local_telemetry_dir(cfg, artifact_namespace),
        stage=telemetry_stage,
        stage_summary_filename=stage_summary_filename,
        cfg=cfg,
    )
    kwargs = {
        "topics": topics,
        "chat_model": chat_model,
        "cfg": cfg,
        "resume": resume,
        "max_retries": max_retries,
        "verbose": verbose,
        "telemetry": telemetry,
        "llm_concurrency": llm_concurrency,
        "artifact_namespace": artifact_namespace,
        "topic_group_label": topic_group_label,
    }
    run_error = None
    try:
        if log_to_file:
            return _run_with_logging(
                _run_local_taxonomy_construction_inner,
                cfg,
                log_pass_name,
                **kwargs,
            )
        return _run_local_taxonomy_construction_inner(**kwargs)
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path = os.path.join(
            _local_taxonomy_dir(cfg, artifact_namespace),
            _LOCAL_TAXONOMY_MANIFEST,
        )
        manifest = (
            load_from_pickle(manifest_path)
            if os.path.exists(manifest_path)
            else {}
        )
        entries = manifest.get("topics", [])
        telemetry.finish(
            error=run_error,
            metadata={
                "construction_complete": manifest.get("complete", False),
                "expected_topic_count": manifest.get("expected_topic_count"),
                "completed_topic_count": len(entries),
                "remaining_topic_count": max(
                    0,
                    (manifest.get("expected_topic_count") or 0) - len(entries),
                ),
                "configured_llm_concurrency": llm_concurrency,
                "concurrent_provider_enabled": (
                    cfg.llm.provider.lower() == "vllm"
                    and llm_concurrency > 1
                ),
                "artifact_directory": _local_taxonomy_dir(
                    cfg,
                    artifact_namespace,
                ),
            },
        )


def run_fine_grained_local_construction(
    topics: Iterable[dict[str, Any]],
    chat_model: Any,
    cfg: Any,
    *,
    resume: bool = False,
    max_retries: int | None = None,
    log_to_file: bool = True,
    verbose: bool = True,
    enable_telemetry: bool = True,
    llm_concurrency: int = 256,
) -> list[LocalTaxonomyArtifact]:
    """Construct local hierarchies, concurrently when the provider is vLLM."""

    return _run_local_construction_stage(
        topics,
        chat_model,
        cfg,
        artifact_namespace=_FINE_GRAINED_ARTIFACT_NAMESPACE,
        topic_group_label="Fine-Grained",
        telemetry_stage="local_construction",
        stage_summary_filename="local_construction_stage_summary.json",
        log_pass_name="fine_grain_local",
        resume=resume,
        max_retries=max_retries,
        log_to_file=log_to_file,
        verbose=verbose,
        enable_telemetry=enable_telemetry,
        llm_concurrency=llm_concurrency,
    )


def run_broad_local_construction(
    topics: Iterable[dict[str, Any]],
    chat_model: Any,
    cfg: Any,
    output_suffix: str = "",
    *,
    resume: bool = False,
    max_retries: int | None = None,
    log_to_file: bool = True,
    verbose: bool = True,
    enable_telemetry: bool = True,
    llm_concurrency: int = 256,
) -> list[LocalTaxonomyArtifact]:
    """Construct and persist independent local hierarchies for one broad pass."""

    pass_key = _broad_pass_key(output_suffix)
    artifact_namespace = _broad_artifact_namespace(output_suffix)
    return _run_local_construction_stage(
        topics,
        chat_model,
        cfg,
        artifact_namespace=artifact_namespace,
        topic_group_label=f"Broad {pass_key}",
        telemetry_stage=f"{artifact_namespace}_local_construction",
        stage_summary_filename=(
            f"{artifact_namespace}_local_construction_summary.json"
        ),
        log_pass_name=f"{pass_key}_local",
        resume=resume,
        max_retries=max_retries,
        log_to_file=log_to_file,
        verbose=verbose,
        enable_telemetry=enable_telemetry,
        llm_concurrency=llm_concurrency,
    )


def _run_fine_grained_taxonomy_merge_inner(
    artifacts: Iterable[LocalTaxonomyArtifact],
    chat_model: Any,
    cfg: Any,
    state: TaxonomyPassState,
    state_files: TaxonomyStateFiles,
    *,
    verbose: bool,
    telemetry: TaxonomyRunRecorder | None,
    cycle_decision_store: FileCycleDecisionStore,
) -> TaxonomyPassState:
    def persist_subsumption_cache() -> None:
        _save_subsumption_cache(state, cfg, state_files)

    for step_index, artifact in enumerate(tqdm(list(artifacts)), start=1):
        topic_id = artifact.topic_id
        if topic_id in state.topic_ids_completed:
            if telemetry:
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": "skipped",
                    "total_wall_clock_seconds": 0.0,
                })
            print(f"{topic_id} merged, skip")
            continue

        print(f"Merging Fine-Grained Topic Group {topic_id}")
        print("number of terms:", len(artifact.terms))
        step_start_timestamp = datetime.now().astimezone().isoformat(
            timespec="milliseconds"
        )
        step_start_perf = time.perf_counter()
        initial_edge_count = len(state.edges)
        initial_node_count = len(edge_nodes(state.edges))
        timings: dict[str, float] = {}
        try:
            telemetry_context = (
                telemetry.context(topic_id=topic_id, step_index=step_index)
                if telemetry
                else nullcontext()
            )
            with telemetry_context:
                operation_start = time.perf_counter()
                for original, local_node in artifact.term_to_node.items():
                    if original != local_node:
                        add_term_mapping(
                            state.term_mapping,
                            original,
                            local_node,
                            chat_model,
                            telemetry=telemetry,
                        )
                timings["mapping_import_seconds"] = (
                    time.perf_counter() - operation_start
                )

                operation_start = time.perf_counter()
                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=True,
                    telemetry=telemetry,
                    cleaning_context="global_before_similarity_merge",
                    cycle_decision_store=cycle_decision_store,
                )

                edges = clean_taxonomy_edges(
                    artifact.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=False,
                    telemetry=telemetry,
                    cleaning_context="local_before_similarity_merge",
                    cycle_decision_store=cycle_decision_store,
                )
                timings["pre_merge_normalization_seconds"] = (
                    time.perf_counter() - operation_start
                )

                operation_start = time.perf_counter()
                state.evaluated_pairs = merge_new_nodes_into_mapping(
                    state.evaluated_pairs,
                    state.edges,
                    edges,
                    state.term_mapping,
                    chat_model,
                    cfg.paths.domain,
                    cfg.embedding.similarity_threshold,
                    verbose=False,
                    embedding_model=cfg.embedding.model_name,
                    telemetry=telemetry,
                )
                timings["similarity_merge_seconds"] = (
                    time.perf_counter() - operation_start
                )

                operation_start = time.perf_counter()
                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=False,
                    telemetry=telemetry,
                    cleaning_context="global_after_similarity_merge",
                    cycle_decision_store=cycle_decision_store,
                )

                edges = clean_taxonomy_edges(
                    edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=False,
                    telemetry=telemetry,
                    cleaning_context="local_after_similarity_merge",
                    cycle_decision_store=cycle_decision_store,
                )
                timings["post_mapping_normalization_seconds"] = (
                    time.perf_counter() - operation_start
                )
                print("#new subhierarchy#")
                print_hierarchy(edges)

                operation_start = time.perf_counter()
                state.edges = integrate_fine_grained_topic(
                    state.edges,
                    edges,
                    chat_model,
                    cfg.paths.domain,
                    telemetry=telemetry,
                    subsumption_cache=state.subsumption_evaluated_pairs,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                timings["integration_seconds"] = (
                    time.perf_counter() - operation_start
                )

                operation_start = time.perf_counter()
                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=None,
                    verbose=verbose,
                    connect_to_domain=False,
                    telemetry=telemetry,
                    cleaning_context="final_integrated_taxonomy",
                    cycle_decision_store=cycle_decision_store,
                )
                timings["final_cleanup_seconds"] = (
                    time.perf_counter() - operation_start
                )
                print("#updated fine grained hierarchy#")
                print_hierarchy(state.edges)

            state.topic_ids_completed.append(topic_id)
            persistence_start = time.perf_counter()
            snapshot_path = _save_taxonomy_snapshot(
                cfg,
                state,
                snapshot_name=f"after_step_{step_index:05d}",
                step_index=step_index,
                topic_id=topic_id,
            )
            save_pipeline_state(state, cfg, state_files)
            timings["persistence_seconds"] = (
                time.perf_counter() - persistence_start
            )

            if telemetry:
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": "success",
                    "start_timestamp": step_start_timestamp,
                    "end_timestamp": datetime.now().astimezone().isoformat(
                        timespec="milliseconds"
                    ),
                    "total_wall_clock_seconds": round(
                        time.perf_counter() - step_start_perf,
                        6,
                    ),
                    **{
                        name: round(duration, 6)
                        for name, duration in timings.items()
                    },
                    "initial_node_count": initial_node_count,
                    "initial_edge_count": initial_edge_count,
                    "final_node_count": len(edge_nodes(state.edges)),
                    "final_edge_count": len(state.edges),
                    "node_delta": len(edge_nodes(state.edges)) - initial_node_count,
                    "edge_delta": len(state.edges) - initial_edge_count,
                    "snapshot_path": snapshot_path,
                    **telemetry.usage_for_context(
                        topic_id=topic_id,
                        step_index=step_index,
                    ),
                })
            print("completed topic id:", topic_id)
        except BaseException as exc:
            if telemetry:
                if isinstance(exc, ManualCycleDecisionRequired):
                    unit_status = "awaiting_human_decision"
                elif isinstance(exc, ManualCycleResolutionAborted):
                    unit_status = "aborted"
                else:
                    unit_status = "failure"
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": unit_status,
                    "start_timestamp": step_start_timestamp,
                    "end_timestamp": datetime.now().astimezone().isoformat(
                        timespec="milliseconds"
                    ),
                    "total_wall_clock_seconds": round(
                        time.perf_counter() - step_start_perf,
                        6,
                    ),
                    **{
                        name: round(duration, 6)
                        for name, duration in timings.items()
                    },
                    "error": f"{type(exc).__name__}: {exc}",
                    **telemetry.usage_for_context(
                        topic_id=topic_id,
                        step_index=step_index,
                    ),
                })
            raise

    print("number of fine-grained edges after removing transitive edges:", len(state.edges))
    return state


def run_fine_grained_taxonomy_merge(
    chat_model: Any,
    cfg: Any,
    state: TaxonomyPassState | None = None,
    *,
    resume: bool = False,
    log_to_file: bool = True,
    verbose: bool = True,
    enable_telemetry: bool = True,
) -> TaxonomyPassState:
    """Load and merge persisted fine-grained local hierarchies in order."""

    telemetry = TaxonomyRunRecorder(
        enabled=enable_telemetry,
        telemetry_dir=_integration_telemetry_dir(cfg),
        stage="fine_grained_integration",
        stage_summary_filename="fine_grained_integration_summary.json",
        cfg=cfg,
    )
    loaded_artifacts: list[LocalTaxonomyArtifact] = []
    initial_state = state or TaxonomyPassState()
    run_error = None
    run_status = None
    initial_node_count = 0
    initial_edge_count = 0
    decision_file = os.path.join(
        _integration_dir(cfg),
        "pending_cycle_decisions.json",
    )
    cycle_decision_store = FileCycleDecisionStore(
        decision_file,
        domain=cfg.paths.domain,
        experiment_name=cfg.paths.experiment_name,
    )
    try:
        loaded_artifacts = load_local_taxonomy_artifacts(cfg)
        state_files = fine_grained_merge_state_files()
        if resume:
            initial_state = load_pipeline_state(
                cfg,
                state_files,
                default_state=initial_state,
            )

        initial_node_count = len(edge_nodes(initial_state.edges))
        initial_edge_count = len(initial_state.edges)
        initial_snapshot_path = os.path.join(
            _integration_dir(cfg),
            "snapshots",
            "initial.json",
        )
        if not resume or not os.path.exists(initial_snapshot_path):
            _save_taxonomy_snapshot(
                cfg,
                initial_state,
                snapshot_name="initial",
                step_index=0,
                topic_id=None,
            )

        kwargs = {
            "artifacts": loaded_artifacts,
            "chat_model": chat_model,
            "cfg": cfg,
            "state": initial_state,
            "state_files": state_files,
            "verbose": verbose,
            "telemetry": telemetry,
            "cycle_decision_store": cycle_decision_store,
        }
        if log_to_file:
            return _run_with_logging(
                _run_fine_grained_taxonomy_merge_inner,
                cfg,
                "fine_grain_merge",
                **kwargs,
            )
        return _run_fine_grained_taxonomy_merge_inner(**kwargs)
    except ManualCycleDecisionRequired as exc:
        run_status = "awaiting_human_decision"
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    except ManualCycleResolutionAborted as exc:
        run_status = "aborted"
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    except BaseException as exc:
        run_status = "failure"
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        completed_count = len(initial_state.topic_ids_completed)
        telemetry.finish(
            error=run_error,
            status=run_status,
            metadata={
                "integration_complete": (
                    completed_count == len(loaded_artifacts)
                ),
                "expected_step_count": len(loaded_artifacts),
                "completed_step_count": completed_count,
                "remaining_step_count": max(
                    0,
                    len(loaded_artifacts) - completed_count,
                ),
                "initial_node_count": initial_node_count,
                "initial_edge_count": initial_edge_count,
                "final_node_count": len(edge_nodes(initial_state.edges)),
                "final_edge_count": len(initial_state.edges),
                "snapshot_directory": os.path.join(
                    _integration_dir(cfg),
                    "snapshots",
                ),
                "cycle_decision_file": decision_file,
                "cycle_decision_pending": (
                    run_status == "awaiting_human_decision"
                ),
            },
        )


def run_fine_grained_taxonomy_pass(
    topics: Iterable[dict[str, Any]],
    chat_model: Any,
    cfg: Any,
    state: TaxonomyPassState | None = None,
    *,
    resume: bool = False,
    max_retries: int | None = None,
    log_to_file: bool = True,
    verbose: bool = True,
    enable_telemetry: bool = True,
    llm_concurrency: int = 256,
) -> TaxonomyPassState:
    """Compatibility wrapper that runs local construction and then merging."""

    run_fine_grained_local_construction(
        topics,
        chat_model,
        cfg,
        resume=resume,
        max_retries=max_retries,
        log_to_file=log_to_file,
        verbose=verbose,
        enable_telemetry=enable_telemetry,
        llm_concurrency=llm_concurrency,
    )
    return run_fine_grained_taxonomy_merge(
        chat_model,
        cfg,
        state=state,
        resume=resume,
        log_to_file=log_to_file,
        verbose=verbose,
        enable_telemetry=enable_telemetry,
    )


def _run_broad_taxonomy_merge_inner(
    artifacts: Iterable[LocalTaxonomyArtifact],
    chat_model: Any,
    cfg: Any,
    state: TaxonomyPassState,
    state_files: TaxonomyStateFiles,
    *,
    verbose: bool,
    telemetry: TaxonomyRunRecorder | None,
) -> TaxonomyPassState:
    def persist_subsumption_cache() -> None:
        _save_subsumption_cache(state, cfg, state_files)

    for step_index, artifact in enumerate(tqdm(list(artifacts)), start=1):
        topic_id = artifact.topic_id
        if topic_id in state.topic_ids_completed:
            if telemetry:
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": "skipped",
                    "total_wall_clock_seconds": 0.0,
                })
            print(f"{topic_id} completed, skip")
            continue

        print(f"Processing Coarse Topic Group {topic_id}")
        print(artifact.terms)
        print("number of terms:", len(artifact.terms))
        step_start = time.perf_counter()
        initial_edge_count = len(state.edges)
        try:
            telemetry_context = (
                telemetry.context(topic_id=topic_id, step_index=step_index)
                if telemetry
                else nullcontext()
            )
            with telemetry_context:
                for original, local_node in artifact.term_to_node.items():
                    if original != local_node:
                        add_term_mapping(
                            state.term_mapping,
                            original,
                            local_node,
                            chat_model,
                            telemetry=telemetry,
                        )

                edges = clean_taxonomy_edges(
                    artifact.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=True,
                    telemetry=telemetry,
                    cleaning_context="broad_local_before_similarity_merge",
                )
                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=True,
                    telemetry=telemetry,
                    cleaning_context="broad_global_before_similarity_merge",
                )

                state.evaluated_pairs = merge_new_nodes_into_mapping(
                    state.evaluated_pairs,
                    state.edges,
                    edges,
                    state.term_mapping,
                    chat_model,
                    cfg.paths.domain,
                    cfg.embedding.similarity_threshold,
                    verbose=False,
                    embedding_model=cfg.embedding.model_name,
                    telemetry=telemetry,
                )

                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=True,
                    telemetry=telemetry,
                    cleaning_context="broad_global_after_similarity_merge",
                )
                edges = clean_taxonomy_edges(
                    edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=state.term_mapping,
                    verbose=verbose,
                    connect_to_domain=True,
                    telemetry=telemetry,
                    cleaning_context="broad_local_after_similarity_merge",
                )

                print("#coarse taxonomy#")
                print_hierarchy(edges)

                state.edges, state.linkage_roots = integrate_broad_topic(
                    state.edges,
                    edges,
                    state.linkage_roots or [],
                    state.term_mapping,
                    chat_model,
                    cfg.paths.domain,
                    telemetry=telemetry,
                    subsumption_cache=state.subsumption_evaluated_pairs,
                    persist_subsumption_cache=persist_subsumption_cache,
                )
                state.edges = clean_taxonomy_edges(
                    state.edges,
                    chat_model,
                    cfg.paths.domain,
                    term_mapping=None,
                    verbose=verbose,
                    connect_to_domain=False,
                    telemetry=telemetry,
                    cleaning_context="broad_final_integrated_taxonomy",
                )
                print("#updated hierarchy#")
                print_hierarchy(state.edges)
                print(
                    "number of edges after removing transitive edges:",
                    len(state.edges),
                )

            state.topic_ids_completed.append(topic_id)
            save_pipeline_state(state, cfg, state_files)
            if telemetry:
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": "success",
                    "total_wall_clock_seconds": round(
                        time.perf_counter() - step_start,
                        6,
                    ),
                    "initial_edge_count": initial_edge_count,
                    "final_edge_count": len(state.edges),
                    "edge_delta": len(state.edges) - initial_edge_count,
                    **telemetry.usage_for_context(
                        topic_id=topic_id,
                        step_index=step_index,
                    ),
                })
            print("completed topic id:", topic_id)
        except BaseException as exc:
            if telemetry:
                telemetry.record_unit({
                    "step_index": step_index,
                    "topic_id": topic_id,
                    "status": "failure",
                    "total_wall_clock_seconds": round(
                        time.perf_counter() - step_start,
                        6,
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    **telemetry.usage_for_context(
                        topic_id=topic_id,
                        step_index=step_index,
                    ),
                })
            raise

    return state


def run_broad_taxonomy_merge(
    chat_model: Any,
    cfg: Any,
    initial_edges: list[tuple[str, str]],
    initial_mapping: dict[str, str],
    linkage_roots: list[str],
    output_suffix: str = "",
    initial_evaluated_pairs: dict[tuple[str, str], str] | None = None,
    initial_subsumption_evaluated_pairs: (
        dict[tuple[str, str], str] | None
    ) = None,
    state: TaxonomyPassState | None = None,
    *,
    resume: bool = False,
    log_to_file: bool = True,
    verbose: bool = True,
    enable_telemetry: bool = True,
) -> TaxonomyPassState:
    """Load and integrate persisted local hierarchies for one broad pass."""

    pass_key = _broad_pass_key(output_suffix)
    artifact_namespace = _broad_artifact_namespace(output_suffix)
    loaded_artifacts = load_local_taxonomy_artifacts(
        cfg,
        artifact_namespace,
    )
    state_files = broad_state_files(output_suffix)
    initial_state = state or TaxonomyPassState(
        edges=initial_edges.copy(),
        term_mapping=initial_mapping.copy(),
        evaluated_pairs=(initial_evaluated_pairs or {}).copy(),
        topic_ids_completed=[],
        linkage_roots=linkage_roots.copy(),
        subsumption_evaluated_pairs=(
            initial_subsumption_evaluated_pairs or {}
        ).copy(),
    )
    if resume:
        initial_state = load_pipeline_state(cfg, state_files, default_state=initial_state)

    telemetry = TaxonomyRunRecorder(
        enabled=enable_telemetry,
        telemetry_dir=_broad_integration_dir(cfg, output_suffix),
        stage=f"{artifact_namespace}_integration",
        stage_summary_filename=f"{artifact_namespace}_integration_summary.json",
        cfg=cfg,
    )
    kwargs = {
        "artifacts": loaded_artifacts,
        "chat_model": chat_model,
        "cfg": cfg,
        "state": initial_state,
        "state_files": state_files,
        "verbose": verbose,
        "telemetry": telemetry,
    }
    run_error = None
    try:
        if log_to_file:
            return _run_with_logging(
                _run_broad_taxonomy_merge_inner,
                cfg,
                f"{pass_key}_merge",
                **kwargs,
            )
        return _run_broad_taxonomy_merge_inner(**kwargs)
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        completed_count = len(initial_state.topic_ids_completed)
        telemetry.finish(
            error=run_error,
            metadata={
                "integration_complete": (
                    completed_count == len(loaded_artifacts)
                ),
                "expected_step_count": len(loaded_artifacts),
                "completed_step_count": completed_count,
                "remaining_step_count": max(
                    0,
                    len(loaded_artifacts) - completed_count,
                ),
                "final_node_count": len(edge_nodes(initial_state.edges)),
                "final_edge_count": len(initial_state.edges),
                "artifact_directory": _local_taxonomy_dir(
                    cfg,
                    artifact_namespace,
                ),
            },
        )
