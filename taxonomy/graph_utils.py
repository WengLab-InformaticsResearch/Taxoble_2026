from collections import defaultdict, deque
import os

def print_hierarchy(edges, output_txt_path=None):
    """
    Print a tree-style hierarchy from (child, parent) tuples.

    Args:
        edges: Iterable of (child, parent) tuples.
        output_txt_path: Optional file path to save hierarchy text.
    """
    tree = defaultdict(list)
    children = set()
    parents = set()

    # Build mappings
    for child, parent in edges:
        tree[parent].append(child)
        children.add(child)
        parents.add(parent)

    # Find root(s): parents that never appear as children
    roots = list(parents - children)
    hierarchy_lines = []

    def _collect_tree(node, prefix=""):
        line = prefix + node
        print(line)
        hierarchy_lines.append(line)
        node_children = tree.get(node, [])
        for i, child in enumerate(node_children):
            is_last = i == len(node_children) - 1
            branch = "└─ " if is_last else "├─ "
            _collect_tree(child, prefix + branch)

    for root in roots:
        _collect_tree(root)

    if output_txt_path:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(hierarchy_lines) + "\n")

def remove_transitive_edges(edges, verbose=True):
    """
    Remove transitive edges from a taxonomy DAG.
    Edges are assumed to be (child, parent).

    Prints removed edges and the reason.
    """

    # Build adjacency: child -> parents
    graph = defaultdict(set)
    for child, parent in edges:
        graph[child].add(parent)

    def find_alternate_path(start, end):
        """
        Find an alternate path start -> ... -> end (length >= 2).
        Returns the path if found, else None.
        """
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()
            for parent in graph[node]:
                if parent == end:
                    return path + [end]
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, path + [parent]))
        return None

    reduced = []

    for child, parent in edges:
        # Temporarily remove direct edge
        graph[child].remove(parent)

        path = find_alternate_path(child, parent)

        if path:
            if verbose:
                print(
                    f"Removed transitive edge: ({child} → {parent}) "
                    f"because path exists: {' → '.join(path)}"
                )
        else:
            reduced.append((child, parent))

        # Restore edge
        graph[child].add(parent)

    return reduced

def get_roots(edges):
    """
    Given taxonomy edges as (child, parent),
    return the set of root nodes.
    """
    children = set()
    parents = set()

    for child, parent in edges:
        children.add(child)
        parents.add(parent)

    roots = parents - children
    return list(roots)

def connect_subroot_to_domain(edges, domain):
    new_roots = get_roots(edges)
    print("roots:", new_roots)

    for r in new_roots:
        if r != domain.lower():
            edges.append((r, domain.lower()))

    return edges

def construct_hierarchy(edges):
    parents = defaultdict(set)
    children = defaultdict(set)
    for edge in edges:
        child, parent = edge
        parents[child].add(parent)
        children[parent].add(child)
    return parents, children

def filter_nodes_in_edges(nodes, edges):
    node_set = set(nodes)

    # keep edges where both endpoints are valid nodes
    valid_edges = [(child, parent) for child, parent in edges
                   if child in node_set and parent in node_set]

    # get nodes that appear in those edges
    filtered_nodes = list({n for edge in valid_edges for n in edge})

    return filtered_nodes, valid_edges

def is_child(node, edges):
    for child, parent in edges:
        if child == node:
            return True
    return False

def get_subhierarchy_edges_from_children(children, target):
    result = set()
    queue = deque([target])
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        for child in children.get(node, set()):
            result.add((child, node))
            queue.append(child)

    return list(result)

def get_subhierarchy_edges_from_parents(parents, target):
    result = set()
    queue = deque([target])
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        for parent in parents.get(node, set()):
            result.add((node, parent))
            queue.append(parent)

    return list(result)
