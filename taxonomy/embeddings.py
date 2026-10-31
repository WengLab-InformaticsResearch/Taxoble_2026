import gc

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def _release_embedding_model(embedder):
    """Move an embedding model off CUDA and release cached GPU memory."""
    if embedder is not None:
        try:
            embedder.to("cpu")
        except Exception:
            pass
    del embedder
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def generate_embeddings_in_batches(
    texts,
    batch_size= 1024,
    embedding_model='all-mpnet-base-v2'
):
    """
    Generate sentence embeddings in batches.
    
    Args:
        texts (List[str]): List of input text strings.
        model_name (str): Pretrained SentenceTransformer model name.
        batch_size (int): Number of texts to process per batch.
    
    Returns:
        np.ndarray: Array of embeddings.
    """
    embeddings = []
    embedder = None
    try:
        embedder = SentenceTransformer(embedding_model)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)
    finally:
        _release_embedding_model(embedder)

def compute_pairwise_cosine_similarity(topic_names, embedding_model='all-mpnet-base-v2'):
    embeddings = generate_embeddings_in_batches(topic_names, embedding_model=embedding_model)
    # Compute full cosine similarity matrix
    print('computing cosine similarity...')
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create dictionary of pairwise similarities (i < j to avoid duplicates)
    similarity_dict = {}
    n = similarity_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            similarity_dict[(topic_names[i], topic_names[j])] = similarity_matrix[i, j]
    
    return similarity_dict

def find_similar_terms(selected_terms, all_terms, embedder=None, threshold=0.9, exclude_self=True, embedding_model='all-mpnet-base-v2', device=None):
    created_embedder = embedder is None
    try:
        if embedder is None:
            if device is not None:
                embedder = SentenceTransformer(embedding_model, device=device)
            else:
                embedder = SentenceTransformer(embedding_model)

        selected_terms = list(selected_terms)
        all_terms = list(all_terms)
        selected_emb = embedder.encode(selected_terms, normalize_embeddings=True)
        all_emb = embedder.encode(all_terms, normalize_embeddings=True)

        sims = selected_emb @ all_emb.T  # cosine because normalized
        result = {}

        for i, term in enumerate(selected_terms):
            row = sims[i].copy()

            if exclude_self:
                same_idx = [j for j, t in enumerate(all_terms) if t == term]
                for j in same_idx:
                    row[j] = -1.0

            j_best = int(np.argmax(row))
            best_sim = float(row[j_best])

            if best_sim >= threshold:
                result[term] = all_terms[j_best]

        return result
    finally:
        if created_embedder:
            _release_embedding_model(embedder)
