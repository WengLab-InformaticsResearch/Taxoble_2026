from itertools import combinations
import networkx as nx
import time
from tqdm import tqdm
from .embeddings import compute_pairwise_cosine_similarity
from .llm_client import invoke_chat_model

def build_similarity_graph(nodes, pairwise_similarity, similarity_threshold):
    """
    Build an undirected graph where edges connect sufficiently similar nodes.
    """
    Gsim = nx.Graph()
    Gsim.add_nodes_from(nodes)

    for (n1, n2), sim in pairwise_similarity.items():
        if sim >= similarity_threshold:
            Gsim.add_edge(n1, n2, weight=float(sim))

    return Gsim

def choose_representative(component, pairwise_similarity):
    """
    Choose the most 'central' node in a component by maximizing
    summed similarity to all other nodes in the component.
    """
    scores = {}

    for n in component:
        score = 0.0
        for m in component:
            if n == m:
                continue
            # similarity dict may store (n,m) or (m,n)
            score += pairwise_similarity.get(
                (n, m),
                pairwise_similarity.get((m, n), 0.0)
            )
        scores[n] = score

    return max(scores, key=scores.get)

def build_node_mapping(nodes, pairwise_similarity, similarity_threshold, domain):
    """
    Build a mapping from each node to its representative node.
    """
    Gsim = build_similarity_graph(
        nodes,
        pairwise_similarity,
        similarity_threshold
    )

    components = [list(c) for c in nx.connected_components(Gsim)]

    node_mapping = {}

    for component in components:
        if domain.lower() in component:
            rep = domain.lower()
        else:
            rep = choose_representative(component, pairwise_similarity)

        for node in component:
            node_mapping[node] = rep

    return node_mapping, components

def check_similarity_by_llm(
    node1,
    node2,
    chat_model,
    try_times=5,
    retry_delay=1,
    verbose=True,
    telemetry=None,
):
    logical_call_id = (
        telemetry.new_logical_call_id("synonym_equivalence_check")
        if telemetry
        else None
    )
    for attempt in range(try_times):
        try:
            prompt = f"""
You are an expert in taxonomy and semantic classification.
Determine whether the following two terms are EXACTLY semantically equivalent and can be safely merged as strict synonyms in a biomedical ontology.
Return "yes" ONLY if:
- both terms refer to exactly the same concept
- neither term is broader or narrower than the other
- neither term adds or removes specificity (such as anatomical site, laterality, subtype, severity, systemic vs localized form, acute vs chronic form, etc.)

Otherwise, return "no".
If you cannot determine whether the two terms are semantically equivalent, return "no".

Term 1: {node1}
Term 2: {node2}
Please return "yes" or "no" only.
            """
            response = (
                telemetry.invoke(
                    chat_model,
                    prompt,
                    request_type="synonym_equivalence_check",
                    function_name="check_similarity_by_llm",
                    attempt_number=attempt + 1,
                    logical_call_id=logical_call_id,
                )
                if telemetry
                else invoke_chat_model(chat_model, prompt, attempt + 1)
            )
            if response.content:
                if verbose:
                    print("prompt:", prompt)
                    print(response.content)
                output = response.content.strip().lower()
                if output in ['yes', 'no']:
                    return output
        except Exception as e:
            print("error:", e)
            pass
        time.sleep(retry_delay) 
    return "no"

def merge_similar_nodes(
    evaluated_pairs,
    nodes,
    chat_modal,
    domain,
    similarity_threshold=0.8,
    verbose=True,
    embedding_model='all-mpnet-base-v2',
    telemetry=None,
):
    pairwise_node_similarity = compute_pairwise_cosine_similarity(nodes, embedding_model=embedding_model)
    pairwise_node_similarity_updated = {}
    candidate_pairs = []
    for pair, sim in pairwise_node_similarity.items():
        if sim >= similarity_threshold:
            candidate_pairs.append(pair)
    
    for pair in tqdm(candidate_pairs):
        if verbose:
            print(pair)
        if pair in evaluated_pairs:
            flag_sim = evaluated_pairs[pair]
        else:
            flag_sim = check_similarity_by_llm(
                pair[0],
                pair[1],
                chat_modal,
                verbose=verbose,
                telemetry=telemetry,
            )
            evaluated_pairs[pair] = flag_sim
            evaluated_pairs[(pair[1], pair[0])] = flag_sim
        if flag_sim == "yes":
            pairwise_node_similarity_updated[pair] = 1
            if verbose:
                print(f"Merging {pair[0]} and {pair[1]}")
            
    node_mapping, similar_components = build_node_mapping(
        nodes,
        pairwise_node_similarity_updated,
        similarity_threshold,
        domain
    )
    for component in similar_components:
        if len(component)>1:
            print(component)
    
    print(f"Number of nodes after merging: {len(set(node_mapping.values()))}")
    
    return node_mapping, evaluated_pairs
