import ast
import re
import time
import networkx as nx

from .llm_client import invoke_chat_model


def _normalize_closed_cycle(cycle):
    """
    Given a CLOSED cycle (last node repeats the first), rotate so the smallest node
    is first, and return a CLOSED tuple.
    Example: ('C','A','B','C') -> ('A','B','C','A')
    """
    assert cycle[0] == cycle[-1], "cycle must be CLOSED (last == first)"
    core = cycle[:-1]  # exclude the repeated last
    min_idx = min(range(len(core)), key=lambda i: core[i])  # natural ordering
    rotated = core[min_idx:] + core[:min_idx]
    return tuple(rotated + [rotated[0]])

def find_next_cycle(edges, skip_cycles=None):
    """
    Find one directed cycle (as a CLOSED normalized tuple).
    If the first found cycle is in skip_cycles, try to find a different cycle by
    temporarily hiding one edge at a time from that cycle (detection-only).
    Return None if no cycle exists or only skipped cycles remain.
    """
    skip_cycles = skip_cycles or set()

    G = nx.DiGraph()
    G.add_edges_from(edges)

    
    def find_cycle_in_graph(graph: nx.DiGraph):
        """
        Find one directed cycle and return it as a CLOSED, normalized tuple
        (n1, n2, ..., nk, n1). Return None if no cycle exists.
        """
        try:
            cyc_edges = nx.find_cycle(graph, orientation="original")
            # Convert edge list to node cycle [u1, u2, ..., uk, u1]
            cycle = [cyc_edges[0][0]] + [v for (_, v, *_) in cyc_edges]
            return _normalize_closed_cycle(cycle)
        except nx.NetworkXNoCycle:
            return None
    # First attempt
    cyc = find_cycle_in_graph(G)
    if cyc is None:
        return None
    if cyc not in skip_cycles:
        return cyc

    # Try to uncover a different cycle by temporarily removing each edge of the skipped cycle
    path_edges = list(zip(cyc[:-1], cyc[1:]))  # [(n1,n2),...,(nk,n1)]
    for e in path_edges:
        if not G.has_edge(*e):
            continue
        G.remove_edge(*e)
        alt = find_cycle_in_graph(G)
        G.add_edge(*e)  # restore
        if alt is not None and alt not in skip_cycles:
            return alt

    # No alternative cycles found (besides already skipped)
    return None

def break_a_cycle(
    cycle,
    model,
    max_retries=5,
    retry_delay=1,
    telemetry=None,
):
    prompt = (
        f"""
In the following cycle, each item is a subcategory of the item that directly follows it. For example, in (A, B, C, A), A is a subcategory of B, B of C, and so on.
A subcategory means "A is a narrower, more specific task or concept than B."
Cycle: {cycle}
Please identify invalid child-parent relationships in the format [('A','B'), ('B','C')] and do not provide any rationale.
        """
    )
    
    logical_call_id = (
        telemetry.new_logical_call_id("cycle_edge_resolution")
        if telemetry
        else None
    )
    for attempt in range(max_retries):
        try:
            response = (
                telemetry.invoke(
                    model,
                    prompt,
                    request_type="cycle_edge_resolution",
                    function_name="break_a_cycle",
                    attempt_number=attempt + 1,
                    logical_call_id=logical_call_id,
                )
                if telemetry
                else invoke_chat_model(model, prompt, attempt + 1)
            )
            if response.content:
                pattern = r"\[\s*(\('.*?'\s*,\s*'.*?'\)\s*(,\s*\('.*?'\s*,\s*'.*?'\)\s*)*)\]"
                
                m = re.search(pattern, response.content, flags=re.S)
                if m:
                    tuple_list = ast.literal_eval(m.group())
                    print(tuple_list)

                    return tuple_list
        except Exception as e:
            # Optional: log the exception e
            print(e)
            pass
        time.sleep(retry_delay)  # Wait before retrying
    return []

def fix_cycles_collect_unresolved(
    edges,
    model,
    retry_attempts=5,    # max times to call break_a_cycle when nothing valid is returned
    verbose=False,
    telemetry=None,
):
    """
    Loop:
      - find a cycle not in skip set,
      - ask model up to retry_attempts times for edges to remove,
      - if it yields existing edges, remove them and continue,
      - else record the cycle as unresolved (for manual review), add to skip set, and continue.

    Stops when either:
      - the graph is acyclic, or
      - all remaining cycles are in the unresolved (skip) set.

    Returns:
      clean_edges, removed_edges, unresolved_cycles (list), rounds
    """
    edges_list = [tuple(e) for e in edges]     # normalize to (u,v)
    removed = []
    unresolved = []
    skip_cycles = set()
    rounds = 0

    while True:
        cycle = find_next_cycle(edges_list, skip_cycles=skip_cycles)
        if cycle is None:
            if verbose:
                if unresolved:
                    print("[done] no more resolvable cycles (only unresolved remain).")
                else:
                    print("[done] acyclic.")
            return edges_list, removed, unresolved, rounds

        rounds += 1
        cycle_id = telemetry.start_cycle(cycle) if telemetry else None
        if verbose:
            print(f"[round {rounds}] cycle: {cycle}")

        edges_set = set(edges_list)
        to_remove = []
        for attempt in range(1, retry_attempts + 1):
            wrong_edges = break_a_cycle(
                cycle,
                model,
                telemetry=telemetry,
            ) or []
            # keep only edges that actually exist in the graph
            to_remove = [tuple(e) for e in wrong_edges if tuple(e) in edges_set]
            if telemetry:
                telemetry.record_cycle_llm_resolution(
                    cycle_id,
                    resolution_attempt=attempt,
                    proposed_edges=wrong_edges,
                    valid_edges=to_remove,
                )
            if to_remove:
                if verbose:
                    print(f"[round {rounds}] attempt {attempt}: removing {to_remove}")
                break
            if verbose:
                print(f"[round {rounds}] attempt {attempt}: no valid edges from model")

        if not to_remove:
            # Couldn’t resolve this cycle now; record and skip it in future searches
            unresolved.append(cycle)
            skip_cycles.add(cycle)
            if verbose:
                print(f"[blocked] stored unresolved cycle and moving on: {cycle}")
            if telemetry:
                telemetry.finish_cycle_automatic(
                    cycle_id,
                    removed_edges=[],
                    resolved=False,
                )
            # Do NOT modify edges; just continue to look for other cycles
            continue

        # Apply removals and continue
        edges_set.difference_update(to_remove)
        edges_list = list(edges_set)
        removed.extend(to_remove)
        if telemetry:
            telemetry.finish_cycle_automatic(
                cycle_id,
                removed_edges=to_remove,
                resolved=True,
            )

        if verbose:
            print(f"[round {rounds}] removed: {to_remove}")

def interactive_manual_edge_removal(
    cleaned_edges,
    unresolved_cycles,
    domain=None,
    telemetry=None,
):
    """
    Interactively remove edges to resolve unresolved taxonomy cycles.

    Parameters
    ----------
    cleaned_edges : list[tuple[str, str]]
        Current edge list (child, parent).
    unresolved_cycles : list[list[str]]
        List of cycles, each represented as a list of node names.

    Returns
    -------
    updated_edges : list[tuple[str, str]]
        Edge list after manual removals.
    """

    edges = set(cleaned_edges)

    print("\n" + "=" * 80)
    print("Manual cycle resolution required")
    if domain:
        print(f"Domain: {domain}")
    print("=" * 80)

    for idx, cycle in enumerate(unresolved_cycles, start=1):
        print(f"\nCycle {idx}/{len(unresolved_cycles)}:")
        print("  Nodes in cycle:")
        for n in cycle:
            print(f"   - {n}")

        # candidate edges involved in this cycle
        cycle_edges = [
            (u, v) for (u, v) in edges
            if u in cycle and v in cycle
        ]

        print("\n  Candidate edges in this cycle:")
        for e in cycle_edges:
            print(f"   {e}")

        print(
            "\nEnter edge(s) to remove in **Python tuple format**:\n"
            "  ('child_topic', 'parent_topic')\n\n"
            "Rules:\n"
            " - Multiple edges: separate by semicolon (;)\n"
            " - Press Enter to skip this cycle\n"
            " - Type 'q' to abort manual resolution\n"
        )

        user_input = input("Edges to remove: ").strip()

        if user_input.lower() == "q":
            print("Exiting manual edge resolution.")
            if telemetry:
                telemetry.record_cycle_manual(
                    cycle,
                    raw_input=user_input,
                    removed_edges=[],
                    status="aborted",
                )
            break

        if not user_input:
            print("Skipping this cycle.")
            if telemetry:
                telemetry.record_cycle_manual(
                    cycle,
                    raw_input=user_input,
                    removed_edges=[],
                    status="skipped",
                )
            continue

        removed_for_cycle = []
        try:
            entries = user_input.split(";")
            for entry in entries:
                edge = ast.literal_eval(entry.strip())

                if (
                    not isinstance(edge, tuple)
                    or len(edge) != 2
                    or not all(isinstance(x, str) for x in edge)
                ):
                    print(f"Invalid edge format (ignored): {entry}")
                    continue

                if edge in edges:
                    edges.remove(edge)
                    removed_for_cycle.append(edge)
                    print(f"Removed edge: {edge}")
                else:
                    print(f"Edge not found (ignored): {edge}")

            if telemetry:
                telemetry.record_cycle_manual(
                    cycle,
                    raw_input=user_input,
                    removed_edges=removed_for_cycle,
                    status="resolved" if removed_for_cycle else "invalid",
                )
        except Exception as e:
            print("Failed to parse input. Skipping this cycle.")
            print("Error:", e)
            if telemetry:
                telemetry.record_cycle_manual(
                    cycle,
                    raw_input=user_input,
                    removed_edges=removed_for_cycle,
                    status="invalid",
                    error=f"{type(e).__name__}: {e}",
                )

    print("\nManual resolution completed.")
    print(f"Remaining edges: {len(edges)}")

    return list(edges)
