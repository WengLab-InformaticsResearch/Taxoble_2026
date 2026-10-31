import numpy as np
import pandas as pd

import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import pickle

import time
import os

import unicodedata
from taxonomy.config import LLMConfig
from taxonomy.llm_client import build_chat_model

import networkx as nx


def save_to_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_latest_cache(file_path, cache_prefix):
    cache_dir = os.path.join(file_path, "cache")
    if not os.path.isdir(cache_dir):
        return {}

    cache_pattern = re.compile(rf"^{re.escape(cache_prefix)}_i_(\d+)\.pkl$")
    cache_files = []
    for filename in os.listdir(cache_dir):
        match = cache_pattern.match(filename)
        if match:
            cache_files.append((int(match.group(1)), filename))

    if not cache_files:
        return {}

    _, latest_filename = max(cache_files)
    latest_path = os.path.join(cache_dir, latest_filename)
    print(f"Resuming from cache: {latest_path}")
    return load_from_pickle(latest_path)


def delete_cache_files(file_path, cache_prefix):
    cache_dir = os.path.join(file_path, "cache")
    if not os.path.isdir(cache_dir):
        return

    cache_pattern = re.compile(rf"^{re.escape(cache_prefix)}_i_\d+\.pkl$")
    for filename in os.listdir(cache_dir):
        if cache_pattern.match(filename):
            os.remove(os.path.join(cache_dir, filename))


def is_abbreviation(keyword: str, domain: str, model, max_attempts: int = 3, verbose=False) -> str:
    for attempt in range(max_attempts):
        try:
            message = f"""
You are an expert in {domain}.
From the text "{keyword}", determine whether it contains one or more abbreviations.

Definition:
- Abbreviation = acronym or initialism formed from initial letters of multiple words (e.g., CKD, T2D)

Rules:
- If the text clearly contains such an abbreviation, return YES.
- If the text is a normal word or phrase (not an acronym or initialism), return NO.
- Ignore brand names and CamelCase or PascalCase words.
- If the text is ambiguous to determine whether it contains an abbreviation, return UNCERTAIN.

Return exactly one value:
YES
NO
UNCERTAIN
"""
            response = model.invoke(message)
            answer = response.content.strip().lower().rstrip(".")
            if verbose:
                print(keyword, "->", answer)
            if answer in {"yes", "no", "uncertain"}:
                return answer
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
    return "no"  # default after max_attempts

def expand_abbreviation(keyword: str, domain: str, model, max_attempts: int = 3, verbose=False) -> str:
    for attempt in range(max_attempts):
        try:
            message = f"""
You are an expert in {domain}.
Given the text "{keyword}" and the domain "{domain}", replace any abbreviation in the text with its most likely full form in this domain.

Rules:
- Abbreviation = acronym or initialism only
- Use domain knowledge to infer the most common expansion in this domain.
- Replace only the abbreviation; keep all other words unchanged
- Preserve numbers, order, and casing as much as possible
- If the expansion is ambiguous or unclear, return UNCERTAIN

Return exactly one value:
- the rewritten full text
- or UNCERTAIN
            """
            response = model.invoke(message)
            answer = response.content.strip().lower().rstrip(".")
            
            if verbose:
                print(keyword, "->", answer)
            if answer == "uncertain" or len(answer) == 0:
                answer = keyword
            return answer
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
    return keyword  # default after max_attempts




def check_keyword_domain_relationship(keyword, domain, model, max_retries=3, retry_delay=1, verbose=False):

    prompt = (
        f"""
You are an expert in {domain}.

Determine whether the concept "{keyword}" belongs to the domain of {domain}.

A concept belongs to the domain if it directly describes any of the following categories in this domain:
- disease or syndrome: A condition that alters or interferes with a normal process, state, or activity of an organism.
- pathophysiology: The biological or pathological processes and functional changes that are associated with or a result of disease or injury.
- diagnosis: A procedure, method, or technique used to determine the nature or identity of a disease or disorder.
- treatment: A procedure, method, or technique designed to prevent a disease or a disorder, or to improve physical function, or used in the process of treating a disease or injury.
- risk factors or etiologies: Factors that contribute to the development or cause of a disease.
- biomarkers: A biological molecule found in blood, other body fluids, or tissues that is a sign of a normal or abnormal process, or of a condition or disease.
- anatomy or biological structures: Physical structures of organisms, including cells, tissues, and organs.
- patient populations: A group of individuals united by a common factor (e.g., geographic location, ethnicity, disease, age, sex).
- subspecialties: Specialized areas of expertise within a medical specialty.
- experimental systems and models: Experimental systems and research techniques used to investigate biological processes or diseases.
- morphological findings: Structural and associated functional alterations in cells or tissues characteristic of pathological processes.
Rules:
- The concept must directly refer to one of the categories above within {domain}.
- Concepts that are only indirectly related to the domain should be classified as NO.
- If the concept is too general, return NO.

Return exactly one value:
- YES
- NO
- UNCERTAIN
        """
    )
    rel1 = 'no'
    for attempt in range(max_retries):
        try:
            response = model.invoke(prompt)
            if response.content:
                rel1 = response.content.strip().lower().rstrip(".")
                if verbose:
                    print(f"Keyword: {keyword}, Response: {rel1}")
                if rel1 in ['yes', 'no', 'uncertain']:
                    break
        except Exception as e:
            # Optional: log the exception e
            pass
        time.sleep(retry_delay)  # Wait before retrying
    if rel1 == 'yes':
        return 'relevant'
    else:
        return 'not relevant'



def preprocess_candidate_terms(
    df,
    column_name,
    domain,
    file_path,
    llm_config=None,
    preprocessing_mode="canonicalize_and_filter_by_domain",
    similarity_threshold=0.75,
    verbose=False
):
    """
    Args:
        preprocessing_mode:
            - "canonicalize": expand abbreviations and merge synonyms.
            - "canonicalize_and_filter_by_domain": canonicalize, then filter by domain.
        similarity_threshold: minimum embedding cosine similarity for synonym candidates.

    Returns:
        keyword_map: dict[str, dict]
    """
    valid_modes = {"canonicalize", "canonicalize_and_filter_by_domain"}
    if preprocessing_mode not in valid_modes:
        raise ValueError(f"preprocessing_mode must be one of {sorted(valid_modes)}")
    run_domain_filter = preprocessing_mode == "canonicalize_and_filter_by_domain"
    llm_config = llm_config or LLMConfig()
    
    chat_model_detect_abbr = build_chat_model(llm_config)
    chat_model_expand_abbr = build_chat_model(llm_config)
    chat_model_in_domain = None
    if run_domain_filter:
        chat_model_in_domain = build_chat_model(llm_config)

    # -------- pass 1: collect & normalize --------
    series = (
        df[column_name]
        .dropna()
        .str.split(";")
        .explode()
        .str.strip()
    )

    raw_keywords = series.unique()
    keyword_map = {}

    for raw_kw in tqdm(raw_keywords):

        keyword_map[raw_kw] = {
            "is_abbreviation": False,
            "expanded": None,
            "merged": None,
            "in_domain": None,
            "final": None,
            }

    
    # -------- pass 2: detect abbreviations --------
    if not os.path.exists(os.path.join(file_path, 'abbr_flags.pkl')):
        abbr_flags = load_latest_cache(file_path, 'abbr_flags')
        for i, kw in enumerate(tqdm(sorted(raw_keywords), desc="Detecting abbreviations")):
            if len(kw) < 3:
                continue
            if kw in abbr_flags:
                continue
            print(kw)
            flag = is_abbreviation(kw, domain, chat_model_detect_abbr, verbose=verbose)
            abbr_flags[kw] = flag in {"yes", "uncertain"}
            if i % 20 == 0:
                os.makedirs(os.path.join(file_path, "cache"), exist_ok=True)
                save_to_pickle(abbr_flags, os.path.join(file_path, "cache", f'abbr_flags_i_{i}.pkl'))
        print("Number of abbreviations detected:", sum(abbr_flags.values()))
        save_to_pickle(abbr_flags, os.path.join(file_path, 'abbr_flags.pkl'))
        delete_cache_files(file_path, 'abbr_flags')
    else:
        abbr_flags = load_from_pickle(os.path.join(file_path, 'abbr_flags.pkl'))
    
    # -------- pass 3: expand abbreviations --------
    
    if not os.path.exists(os.path.join(file_path, 'expansion_cache.pkl')):
        expansion_cache = load_latest_cache(file_path, 'expansion_cache')
        abbr_kw = [kw for kw, is_abbr in abbr_flags.items() if is_abbr]
        for i, kw in enumerate(tqdm(sorted(abbr_kw), desc="Expanding abbreviations")):
            print(kw)
            if kw in expansion_cache:
                continue
            expanded = expand_abbreviation(kw, domain, chat_model_expand_abbr, verbose=verbose)
            expansion_cache[kw] = expanded
            if i % 20 == 0:
                os.makedirs(os.path.join(file_path, "cache"), exist_ok=True)
                save_to_pickle(expansion_cache, os.path.join(file_path, "cache", f'expansion_cache_i_{i}.pkl'))
        save_to_pickle(expansion_cache, os.path.join(file_path, 'expansion_cache.pkl'))
    else:
        expansion_cache = load_from_pickle(os.path.join(file_path, 'expansion_cache.pkl'))

    # -------- pass 4: merging synonyms --------
    processed_keywords = [expansion_cache.get(kw) or kw for kw in raw_keywords]
    def normalize_keyword(kw):
        kw = kw.strip().lower()
        kw = unicodedata.normalize('NFC', kw)
        kw = "".join(c for c in kw if not unicodedata.combining(c))
        kw = re.sub(r"[^a-z0-9\s]", " ", kw)
        kw = re.sub(r"\s+", " ", kw).strip()
        return kw
    processed_keywords = list(set([normalize_keyword(kw) for kw in processed_keywords]))
    
    if not os.path.exists(os.path.join(file_path, 'synonym_mapping.pkl')):
        pairwise_node_similarity = compute_pairwise_cosine_similarity(processed_keywords)
        chat_model_check_synonym = build_chat_model(llm_config)
        
        if not os.path.exists(os.path.join(file_path, 'pairwise_node_similarity_updated.pkl')):
            pairwise_node_similarity_updated = load_latest_cache(file_path, 'pairwise_node_similarity_updated')
            evaluated_synonym_pairs = load_latest_cache(file_path, 'evaluated_synonym_pairs')
            candidate_pairs = []
            for pair, sim in pairwise_node_similarity.items():
                if sim >= similarity_threshold:
                    candidate_pairs.append(pair)
            for i, pair in enumerate(tqdm(candidate_pairs)):
                if verbose:
                    print(pair)
                if pair in evaluated_synonym_pairs:
                    flag_sim = evaluated_synonym_pairs[pair]
                else:
                    flag_sim = check_similarity_by_llm(pair[0], pair[1], chat_model_check_synonym, verbose=verbose)
                    evaluated_synonym_pairs[pair] = flag_sim
                    evaluated_synonym_pairs[(pair[1], pair[0])] = flag_sim
                if i % 100 == 0:
                    os.makedirs(os.path.join(file_path, "cache"), exist_ok=True)
                    save_to_pickle(pairwise_node_similarity_updated, os.path.join(file_path, "cache", f'pairwise_node_similarity_updated_i_{i}.pkl'))
                    save_to_pickle(evaluated_synonym_pairs, os.path.join(file_path, "cache", f'evaluated_synonym_pairs_i_{i}.pkl'))
                if flag_sim != 'yes' or pair in pairwise_node_similarity_updated:
                    continue
                pairwise_node_similarity_updated[pair] = 1
                if verbose:
                    print(f"Merging {pair[0]} and {pair[1]}")
            save_to_pickle(pairwise_node_similarity_updated, os.path.join(file_path, 'pairwise_node_similarity_updated.pkl'))
            save_to_pickle(evaluated_synonym_pairs, os.path.join(file_path, 'evaluated_synonym_pairs.pkl'))
        else:
            pairwise_node_similarity_updated = load_from_pickle(os.path.join(file_path, 'pairwise_node_similarity_updated.pkl'))
        
        print(pairwise_node_similarity_updated)
        synonym_mapping, components = build_node_mapping(
            processed_keywords,
            pairwise_node_similarity_updated,
            domain=domain,
            similarity_threshold=similarity_threshold,
        )
        print("Merging components:")
        for component in components:
            if len(component)>1:
                print(component)
        print(f"Number of keywords after merging: {len(set(synonym_mapping.values()))}")
        save_to_pickle(synonym_mapping, os.path.join(file_path, 'synonym_mapping.pkl'))
    else:
        synonym_mapping = load_from_pickle(os.path.join(file_path, 'synonym_mapping.pkl'))

    if not os.path.exists(os.path.join(file_path, 'merged_keywords.pkl')):
        merged_keywords = {}
        for kw in tqdm(raw_keywords, desc="Merging synonyms"):
            merged_kw = synonym_mapping.get(normalize_keyword(expansion_cache.get(kw, kw)), kw)
            merged_keywords[kw] = merged_kw
        save_to_pickle(merged_keywords, os.path.join(file_path, 'merged_keywords.pkl'))
    else:
        merged_keywords = load_from_pickle(os.path.join(file_path, 'merged_keywords.pkl'))
    

    # -------- pass 5: optionally filter by domain --------
    domain_flags_all = {}
    if run_domain_filter and not os.path.exists(os.path.join(file_path, 'domain_flags.pkl')):
        domain_flags = load_latest_cache(file_path, 'domain_flags')
        merged_keyword_set = set(merged_keywords.values())
        for i, kw in enumerate(tqdm(sorted(merged_keyword_set), desc="Filtering by domain")):
            if len(kw) < 3:
                continue
            if kw in domain_flags:
                continue

            print(kw)
            rel = check_keyword_domain_relationship(kw, domain, chat_model_in_domain, verbose=verbose)
            domain_flags[kw] = rel == 'relevant'
            if i % 20 == 0:
                os.makedirs(os.path.join(file_path, "cache"), exist_ok=True)
                save_to_pickle(domain_flags, os.path.join(file_path, "cache", f'domain_flags_i_{i}.pkl'))
        save_to_pickle(domain_flags, os.path.join(file_path, 'domain_flags.pkl'))
    elif run_domain_filter:
        domain_flags = load_from_pickle(os.path.join(file_path, 'domain_flags.pkl'))

    if run_domain_filter:
        for kw in sorted(raw_keywords):
            domain_flags_all[kw] = domain_flags.get(merged_keywords.get(kw, kw), False)

    
    # -------- pass 6: populate keyword_map --------
    for raw_kw, entry in keyword_map.items():
        entry["is_abbreviation"] = abbr_flags.get(raw_kw, False)
        entry["expanded"] = expansion_cache.get(raw_kw)
        entry["merged"] = merged_keywords.get(raw_kw, normalize_keyword(entry["expanded"] or raw_kw))
        if run_domain_filter:
            entry["in_domain"] = domain_flags_all.get(raw_kw, False)
            entry["final"] = entry["merged"] if entry["in_domain"] else None
        else:
            entry["final"] = entry["merged"]

    return keyword_map






#--synonym merging----
from sklearn.metrics.pairwise import cosine_similarity

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
    embedder = SentenceTransformer(embedding_model)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedder.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)

    return np.vstack(embeddings)
def compute_pairwise_cosine_similarity(topic_names):
    embeddings = generate_embeddings_in_batches(topic_names)
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

def check_similarity_by_llm(node1, node2, chat_model, try_times=5, retry_delay=1, verbose=True):
    for _ in range(try_times):
        try:
            prompt = f"""
You are an expert in taxonomy and semantic classification.
Determine whether the following two terms are similar (i.e., same concept or synonyms indicating the same concept).
Term 1: {node1}
Term 2: {node2}
Please return "yes" or "no" only.
            """
            response = chat_model.invoke(prompt)
            if response.content:
                if verbose:
                    print(response.content)
                output = response.content.strip().lower()
                if output in ['yes', 'no']:
                    return output
        except Exception as e:
            print("error:", e)
            pass
        time.sleep(retry_delay) 
    return "no"



import networkx as nx


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

if __name__ == "__main__":

    preprocessing_mode = "canonicalize_and_filter_by_domain" #or "canonicalize" 

    llm_config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11436",
        model="gpt-oss:20b",
        temperature=0,
        timeout=20,
    )
    # llm_config = LLMConfig(
    #     provider="openai",
    #     model="gpt-5.4",
    #     api_key=os.getenv("OPENAI_API_KEY") or None,
    #     temperature=0,
    #     timeout=60,
    #     max_output_tokens=4096,
    #     max_retries=5,
    #     retry_delay=1,
    #     verbose=False,
    # )
    
    # Input data
    # # JASN input
    domain = 'nephrology'
    folder_name = 'nephrology'
    input_file_path = f'data/raw/{folder_name}'
    output_file_path = f'data/processed/{folder_name}'
    os.makedirs(output_file_path, exist_ok=True)
    df_keywords = pd.read_csv(os.path.join(input_file_path, "jasn_pubmed_1990_2025_w_keywords.csv"))
    column_name = "keywords"
    df_keywords[column_name] = df_keywords[column_name].str.strip()
    
    # ARD input
    # domain = 'rheumatology'
    # folder_name = 'rheumatology'
    # input_file_path = f'data/raw/{folder_name}'
    # output_file_path = f'data/processed/{folder_name}'
    # os.makedirs(output_file_path, exist_ok=True)
    # df_keywords = pd.read_csv(os.path.join(input_file_path, "ard_pubmed_1939_2025_w_keywords.csv"))
    # column_name = "keywords"
    # df_keywords[column_name] = df_keywords[column_name].str.strip()
    
    
    similarity_threshold = 0.75
    
    keyword_mapping = preprocess_candidate_terms(
        df_keywords,
        column_name,
        domain=domain,
        file_path=output_file_path,
        llm_config=llm_config,
        preprocessing_mode=preprocessing_mode,
        similarity_threshold=similarity_threshold,
        verbose=True
    )

    save_to_pickle(keyword_mapping, os.path.join(output_file_path, 'keyword_mapping.pkl'))
    print("Keyword mapping saved to", os.path.join(output_file_path, 'keyword_mapping.pkl'))


   