from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def build_similarity_matrix(
    embeddings: np.ndarray,
    k_nn: int = 20,
    shift: bool = False,
    clip=True,
) -> np.ndarray:
    """Build a k-NN cosine similarity matrix from embeddings."""
    similarity = cosine_similarity(embeddings)
    if clip:
        similarity = np.clip(similarity, 0, None)
    if shift:
        similarity = (similarity + 1.0) / 2.0

    similarity_knn = np.zeros_like(similarity)
    for i in range(similarity.shape[0]):
        idx = np.argpartition(similarity[i], -k_nn)[-k_nn:]
        similarity_knn[i, idx] = similarity[i, idx]

    return similarity_knn


def fit_nmf(
    similarity: np.ndarray,
    n_topics: int,
    random_state: int = 42,
):
    """Fit NMF to a similarity matrix."""
    nmf = NMF(
        n_components=n_topics,
        init="nndsvda",
        solver="mu",
        beta_loss="frobenius",
        max_iter=500,
        alpha_W=0.001,
        alpha_H=0.001,
        l1_ratio=0.5,
        random_state=random_state,
    )

    weights = nmf.fit_transform(similarity)
    components = nmf.components_
    return nmf, weights, components


def reconstruction_error(nmf_model) -> float:
    return nmf_model.reconstruction_err_


def topic_coherence(
    components: np.ndarray,
    embeddings: np.ndarray,
    top_n: int = 10,
) -> float:
    """Return average intra-topic cosine similarity of top keywords."""
    scores = []

    for topic_vec in components:
        top_idx = np.argsort(topic_vec)[-top_n:]
        topic_embeds = embeddings[top_idx]
        similarity = cosine_similarity(topic_embeds)
        scores.append(
            similarity[np.triu_indices_from(similarity, k=1)].mean()
        )

    return float(np.mean(scores))


def sparsity_score(weights: np.ndarray, threshold: float = 0.01) -> float:
    """Return the fraction of near-zero topic assignments."""
    return float(np.mean(weights < threshold))


def select_k(
    similarity: np.ndarray,
    embeddings: np.ndarray,
    k_values: list[int],
):
    results = []

    for k in tqdm(k_values, desc="Selecting K"):
        nmf, weights, components = fit_nmf(similarity, k)
        results.append(
            {
                "k": k,
                "reconstruction_error": nmf.reconstruction_err_,
                "coherence": topic_coherence(components, embeddings),
                "sparsity": sparsity_score(weights),
            }
        )

    return results


def find_elbow_index(values: np.ndarray, eps: float = 1e-6) -> int:
    n = len(values)
    if n < 3:
        return n
    if np.max(values) - np.min(values) < eps:
        return min(n, 3)

    locator = KneeLocator(
        np.arange(n),
        values,
        curve="convex",
        direction="decreasing",
        S=2,
    )
    if locator.knee is None:
        return min(n, 5)
    return int(locator.knee) + 1


def extract_topics_elbow(
    terms,
    weights,
    min_terms=1,
    max_terms=10,
    rel_cutoff=0.2,
):
    topics = []

    for topic_id in range(weights.shape[1]):
        component_weights = weights[:, topic_id]
        sorted_term_idx = np.argsort(component_weights)[::-1]
        sorted_weights = component_weights[sorted_term_idx]

        if sorted_weights[0] <= 1e-12:
            topics.append(
                {"topic_id": topic_id, "n_terms": 0, "terms": []}
            )
            continue

        normalized_weights = sorted_weights / sorted_weights[0]
        elbow = find_elbow_index(normalized_weights)
        relative_k = np.sum(normalized_weights >= rel_cutoff)
        k = int(np.clip(min(elbow, relative_k), min_terms, max_terms))

        topics.append(
            {
                "topic_id": topic_id,
                "n_terms": k,
                "terms": [terms[i] for i in sorted_term_idx[:k]],
            }
        )

    return topics


def select_k_combined(
    scores,
    w_recon=0.4,
    w_sparsity=0.3,
    w_coherence=0.3,
    coherence_floor_ratio=0.95,
    show_plots=True,
):
    score_frame = pd.DataFrame(scores).copy()

    def minmax(values):
        values = np.asarray(values, dtype=float)
        low, high = np.min(values), np.max(values)
        if np.isclose(high, low):
            return np.ones_like(values) * 0.5
        return (values - low) / (high - low)

    recon_n = minmax(score_frame["reconstruction_error"].values)
    sparsity_n = minmax(score_frame["sparsity"].values)
    coherence_n = minmax(score_frame["coherence"].values)

    score_frame["score_combined"] = (
        w_recon * (1.0 - recon_n)
        + w_sparsity * sparsity_n
        + w_coherence * coherence_n
    )

    if coherence_floor_ratio is not None:
        coherence = score_frame["coherence"].values.astype(float)
        floor = coherence_floor_ratio * coherence.max()
        feasible = score_frame["coherence"] >= floor
        candidates = (
            score_frame.loc[feasible] if feasible.any() else score_frame
        )
    else:
        candidates = score_frame

    chosen = candidates.sort_values(
        "score_combined", ascending=False
    ).iloc[0]
    best_k = int(chosen["k"])

    plot_frame = score_frame.sort_values("k")
    if show_plots:
        plt.figure(figsize=(4, 3))
        plt.plot(
            plot_frame["k"],
            plot_frame["score_combined"],
            marker="o",
            markersize=4,
        )
        plt.axvline(
            best_k,
            linestyle="--",
            color="red",
            label=f"best_k={best_k}",
        )
        plt.title("Combined Score by K", fontsize=10)
        plt.xlabel("K", fontsize=9)
        plt.ylabel("score_combined", fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

    columns = [
        "k",
        "reconstruction_error",
        "coherence",
        "sparsity",
        "score_combined",
    ]
    return best_k, plot_frame[columns]


def build_k_values_by_size(n_terms: int) -> list[int]:
    if n_terms <= 2:
        return [1] if n_terms == 1 else [2]

    k_min = 2 if n_terms < 50 else 5
    k_max = min(n_terms // 2, 400)

    if k_max <= k_min:
        return [max(2, min(k_min, n_terms - 1))]

    if n_terms < 50:
        values = list(range(k_min, k_max + 1, 2))
    elif n_terms < 100:
        values = list(range(k_min, k_max + 1, 5))
    else:
        values = np.geomspace(k_min, k_max, num=24)
        values = sorted(set(int(round(value)) for value in values))

    values = [value for value in values if 2 <= value < n_terms]
    return values or [min(max(2, n_terms // 2), n_terms - 1)]


def run_nmf_topic_pass(
    terms,
    embeddings,
    *,
    k_nn,
    k_values,
    w_recon,
    w_sparsity,
    w_coherence,
    coherence_floor_ratio,
    min_terms,
    max_terms,
    rel_cutoff,
    add_singletons=True,
    show_plots=True,
):
    similarity = build_similarity_matrix(embeddings, k_nn=k_nn)
    local_k_values = sorted(
        {int(k) for k in k_values if 2 <= k < len(terms)}
    )
    if not local_k_values:
        local_k_values = [2] if len(terms) > 2 else [1]

    scores = select_k(
        similarity=similarity,
        embeddings=embeddings,
        k_values=local_k_values,
    )

    if show_plots:
        ks = [score["k"] for score in scores]
        features = ["reconstruction_error", "coherence", "sparsity"]
        _, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
        for axis, feature in zip(axes, features):
            values = [float(score[feature]) for score in scores]
            axis.plot(ks, values, marker="o")
            axis.set_title(feature.replace("_", " ").title())
            axis.set_xlabel("K")
            axis.set_ylabel("Score")
            axis.grid(alpha=0.3)

        plt.suptitle("Select K Metrics by Feature")
        plt.tight_layout()
        plt.show()

    best_k, score_frame = select_k_combined(
        scores,
        w_recon=w_recon,
        w_sparsity=w_sparsity,
        w_coherence=w_coherence,
        coherence_floor_ratio=coherence_floor_ratio,
        show_plots=show_plots,
    )

    nmf, weights, components = fit_nmf(
        similarity=similarity,
        n_topics=best_k,
    )
    topics = extract_topics_elbow(
        terms=terms,
        weights=weights,
        min_terms=min_terms,
        max_terms=max_terms,
        rel_cutoff=rel_cutoff,
    )
    topics = [topic for topic in topics if topic["n_terms"] > 0]

    if add_singletons:
        assigned_terms = {
            term for topic in topics for term in topic["terms"]
        }
        unassigned_terms = [
            term
            for term in dict.fromkeys(terms)
            if term not in assigned_terms
        ]
        for term in unassigned_terms:
            topics.append(
                {"topic_id": None, "n_terms": 1, "terms": [term]}
            )

    for new_id, topic in enumerate(topics):
        topic["topic_id"] = new_id

    return {
        "terms": terms,
        "embeddings": embeddings,
        "S": similarity,
        "scores": scores,
        "score_df": score_frame,
        "best_k": best_k,
        "nmf": nmf,
        "W": weights,
        "H": components,
        "topics": topics,
    }


def recursive_nmf_split_until_threshold(
    terms,
    embeddings,
    *,
    base_params,
    split_threshold=20,
    depth=0,
):
    """Recursively split topics, recalculating k-NN from each subset's size."""
    n = len(terms)

    if n <= split_threshold:
        return {
            "depth": depth,
            "n_terms_input": n,
            "best_k": None,
            "topics": [],
            "terms": terms,
            "result": None,
            "stop_reason": "size_le_threshold",
        }

    params = deepcopy(base_params)
    params["k_values"] = build_k_values_by_size(n)
    params["k_nn"] = min(max(5, int(np.sqrt(n))), 50, n - 1)

    result = run_nmf_topic_pass(terms, embeddings, **params)

    node = {
        "depth": depth,
        "n_terms_input": n,
        "best_k": result["best_k"],
        "topics": [],
        "result": result,
        "stop_reason": None,
        "terms": terms,
    }

    local_idx = {term: i for i, term in enumerate(terms)}
    any_child_split = False

    for topic in result["topics"]:
        topic_node = deepcopy(topic)
        topic_terms = list(dict.fromkeys(topic["terms"]))

        if len(topic_terms) <= split_threshold or len(topic_terms) < 6:
            topic_node["children"] = None
            topic_node["stop_reason"] = "child_not_large_enough"
            node["topics"].append(topic_node)
            continue

        sub_idx = [local_idx[term] for term in topic_terms if term in local_idx]

        if len(sub_idx) < 2 or len(sub_idx) >= n:
            topic_node["children"] = None
            topic_node["stop_reason"] = "unsplittable_or_no_reduction"
            node["topics"].append(topic_node)
            continue

        sub_embeddings = embeddings[sub_idx]

        print(f"break topic with {len(topic_terms)} terms:", topic_terms)
        child = recursive_nmf_split_until_threshold(
            terms=topic_terms,
            embeddings=sub_embeddings,
            base_params=base_params,
            split_threshold=split_threshold,
            depth=depth + 1,
        )
        topic_node["children"] = child
        topic_node["stop_reason"] = None
        node["topics"].append(topic_node)
        any_child_split = True

    if not any_child_split:
        node["stop_reason"] = "no_children_split"

    return node


def flatten_tree_leaves_only(tree):
    flat = []

    def walk(node, prefix="root"):
        if not node or not node.get("topics"):
            terms = node.get("terms", []) if node else []
            if terms:
                flat.append(
                    {
                        "topic_id": prefix,
                        "n_terms": len(terms),
                        "terms": terms,
                    }
                )
            return

        for i, topic in enumerate(node["topics"]):
            topic_id = f"{prefix}.{i}"
            child = topic.get("children")
            if child is None:
                terms = topic.get("terms", []) or []
                flat.append(
                    {
                        "topic_id": topic_id,
                        "n_terms": len(terms),
                        "terms": terms,
                    }
                )
            else:
                walk(child, prefix=topic_id)

    walk(tree)
    return flat


__all__ = [
    "build_similarity_matrix",
    "fit_nmf",
    "reconstruction_error",
    "topic_coherence",
    "sparsity_score",
    "select_k",
    "find_elbow_index",
    "extract_topics_elbow",
    "select_k_combined",
    "build_k_values_by_size",
    "run_nmf_topic_pass",
    "recursive_nmf_split_until_threshold",
    "flatten_tree_leaves_only",
]
