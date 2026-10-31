import ast
import json
import time
from typing import Callable

from .llm_client import invoke_chat_model


def generate_taxonomy(
    topics,
    model,
    domain,
    verbose=False,
    max_retries=5,
    retry_delay=1,
    telemetry=None,
):
    """
    Generate a taxonomy and mapping dictionary using JSON output.
    Returns:
      - taxonomy: list of (child, parent) tuples
      - mapping: dict {original_term: modified_term}
    """

    topics_str = "\n".join(topics)

    prompt = f"""
You are a taxonomy expert in {domain}.

Generate a single, coherent taxonomy using strict is-a (subsumption) relations between topics in the domain of {domain}.
Definition of is-a relation:
Term B is a child of Term A if Term B is a specific type of Term A and ALL properties of Term A are also properties of Term B. 
Parent node and child node must have the same semantic type (e.g., disease -> disease, treatment -> treatment, pathophysiology -> pathophysiology, diagnosis -> diagnosis, risk factors -> risk factors, biomarkers -> biomarkers, anatomy -> anatomy, patient populations -> patient populations, subspecialties -> subspecialties, experimental systems and models -> experimental systems and models, morphological findings -> morphological findings).

Rules:
- The root node is "{domain}" and MUST NOT appear as a child.
- Every node must be reachable from the root node "{domain}".
- Nodes may have multiple parents if the is-a relation is valid.
- You MAY introduce multiple levels of intermediate nodes of the domain of {domain} if they belong to the domain and preserve semantic type consistency.
- Do NOT create relations that represent other relations such as part-of, treated-by, studied-by, causes, associated-with, used-in, affects, etc. Only produce strict is-a relations.
- If a topic name is not taxonomy-appropriate or not singular, modify it only with a term that is exactly semantically equivalent to the original term, and record the change.
- Do NOT provide any explanations.

Output JSON ONLY with exactly this structure:

{{
  "taxonomy": [
    ["child", "parent"],
    ["child", "parent"]
  ],
  "mapping": {{
    "original term": "exactly semantically equivalent modified term"
  }}
}}

Topics:
{topics_str}
"""

    if verbose:
        print("PROMPT:\n", prompt)

    logical_call_id = (
        telemetry.new_logical_call_id("taxonomy_generation")
        if telemetry
        else None
    )
    for attempt in range(max_retries):
        try:
            response = (
                telemetry.invoke(
                    model,
                    prompt,
                    request_type="taxonomy_generation",
                    function_name="generate_taxonomy",
                    attempt_number=attempt + 1,
                    logical_call_id=logical_call_id,
                )
                if telemetry
                else invoke_chat_model(model, prompt, attempt + 1)
            )
            content = response.content.strip()

            if verbose:
                print(f"\nRAW MODEL OUTPUT (attempt {attempt+1}):\n{content}")

            data = json.loads(content)

            taxonomy_raw = data.get("taxonomy")
            mapping = data.get("mapping")

            # ---- Validation ----
            if not isinstance(taxonomy_raw, list):
                raise ValueError("taxonomy must be a list")

            taxonomy = []
            for edge in taxonomy_raw:
                if (
                    isinstance(edge, list)
                    and len(edge) == 2
                    and all(isinstance(x, str) for x in edge)
                ):
                    taxonomy.append(tuple(edge))
                else:
                    raise ValueError(f"Invalid taxonomy edge: {edge}")

            if not isinstance(mapping, dict):
                raise ValueError("mapping must be a dict")

            if not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in mapping.items()
            ):
                raise ValueError("mapping must be str → str")

            return taxonomy, mapping

        except Exception as e:
            if verbose:
                print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)
    
    taxonomy = [(topic, domain.lower()) for topic in topics]
    mapping = {}
    return taxonomy, mapping

def resolve_conflict_mapping(
    selected_term,
    new_term1,
    new_term2,
    model,
    max_retries=5,
    retry_delay=1,
    verbose=True,
    telemetry=None,
):
    prompt = (
        f"""
There are two new names for the term '{selected_term}'.
T1: {new_term1}
T2: {new_term2}
Please check which new term is more well-defined the term '{selected_term}' and return T1 or T2 only.
"""
    )

    logical_call_id = (
        telemetry.new_logical_call_id("term_mapping_conflict_resolution")
        if telemetry
        else None
    )
    for attempt in range(max_retries):
        try:
            response = (
                telemetry.invoke(
                    model,
                    prompt,
                    request_type="term_mapping_conflict_resolution",
                    function_name="resolve_conflict_mapping",
                    attempt_number=attempt + 1,
                    logical_call_id=logical_call_id,
                )
                if telemetry
                else invoke_chat_model(model, prompt, attempt + 1)
            )
            
            if response.content:
                if verbose:
                    print("prompt:", prompt)
                    print(response.content)
                output = response.content.strip().lower()[:2]
                if output in ['t1', 't2']:
                    if output == 't1':
                        return new_term1
                    else:
                        return new_term2
        except Exception as e:
            print("error:", e)
            # Optional: log the exception e
            pass
        time.sleep(retry_delay)  # Wait before retrying
    return new_term1

def resolve_transitive_mapping(term, mapping):
    visited = set()
    while term in mapping and term not in visited:
        visited.add(term)
        term = mapping[term]
    return term

def find_nearest_parents(
    children,
    target_parent,
    target,
    chat_model,
    verbose=True,
    visited=None,
    telemetry=None,
    subsumption_cache=None,
    persist_subsumption_cache: Callable[[], None] | None = None,
):
    if visited is None:
        visited = set()
    if target_parent in visited:
        return set()
    visited.add(target_parent)

    nearest = set()

    for c in children[target_parent]:
        if check_subsumption_relation(
            c,
            target,
            chat_model,
            verbose=verbose,
            telemetry=telemetry,
            subsumption_cache=subsumption_cache,
            persist_subsumption_cache=persist_subsumption_cache,
        ) == "subset":
            deeper = find_nearest_parents(
                children,
                c,
                target,
                chat_model,
                verbose=verbose,
                visited=visited,
                telemetry=telemetry,
                subsumption_cache=subsumption_cache,
                persist_subsumption_cache=persist_subsumption_cache,
            )
            if deeper:
                # A more specific parent exists below c; keep only those.
                nearest.update(deeper)
                if verbose:
                    print(f"deeper: {deeper}")
            else:
                # c is a leaf match -> nearest parent.
                nearest.add(c)
                if verbose:
                    print(f"nearest: {nearest}")


    return nearest

def check_subsumption_relation(
    node,
    target,
    model,
    max_retries=3,
    retry_delay=1,
    verbose=True,
    telemetry=None,
    subsumption_cache=None,
    persist_subsumption_cache: Callable[[], None] | None = None,
):
    cache_key = (target.lower().strip(), node.lower().strip())
    if subsumption_cache is not None and cache_key in subsumption_cache:
        result = subsumption_cache[cache_key]
        if verbose:
            print(
                f"cached subsumption: child={cache_key[0]}, "
                f"parent={cache_key[1]}, result={result}"
            )
        return result

    print(target, "-?>", node)
    prompt = (f"""
You are a taxonomy expert.
Determine whether Term 1 is a valid parent of Term 2 using a strict is-a (subsumption) relation. 
Definition of is-a relation: Term 2 is a specific type or subtype of Term 1. All properties of Term 1 are also properties of Term 2.

Evaluation steps:
Step 1: Identify canonical definition
Identify the widely accepted scientific definition of each term based on standard biomedical usage (e.g., commonly used definitions in major biomedical ontologies, textbooks, peer-reviewed scientific literature, authoritative databases).

Step 2: Identify semantic types
Determine the semantic type of Term 1 and Term 2 (e.g., disease, disease group, syndrome, anatomical structure, cell type, experimental model, treatment, biological process).

Step 3: Semantic type compatibility
If the semantic types are different or incompatible, return NO.
Examples of incompatible types include: disease vs experimental model, cell type vs tissue, disease vs treatment, anatomical structure vs biological process.

Step 4: Determine generality
Determine which term is more general.
- A parent term must be broader and more genral than its child.
- If Term 2 is broader or equal in scope to Term 1, return NO.

Step 5: Strict subtype check
Check whether Term 2 is truly a subtype of Term 1.
If they are in other relations (e.g., part-of, treated-by, studied-by, causes, associated-with, used-in, affects), return NO.

Step 6: Final decision
Return YES only if:
- semantic type match
- Term 1 is broader than Term 2
- Term 2 is strictly a subtype of Term 1
- Term 2 can not exist without being Term 1

If the relationship cannot be confidently determined, return UNCERTAIN.

Terms:
Term 1: {node}
Term 2: {target}

Output:
Return one value only:
- YES
- NO
- UNCERTAIN
    
# Do not provide explanations.
"""
    )
    rel = 'no'
    received_valid_response = False
    logical_call_id = (
        telemetry.new_logical_call_id("subsumption_check")
        if telemetry
        else None
    )
    for attempt in range(max_retries):
        try:
            response = (
                telemetry.invoke(
                    model,
                    prompt,
                    request_type="subsumption_check",
                    function_name="check_subsumption_relation",
                    attempt_number=attempt + 1,
                    logical_call_id=logical_call_id,
                )
                if telemetry
                else invoke_chat_model(model, prompt, attempt + 1)
            )
            if response.content:
                rel = response.content.strip().lower().rstrip(".")
                if verbose:
                    print(f"candidate node: {node}, target: {target}, Response: {rel}")
                if rel in ['yes', 'no', 'uncertain']:
                    received_valid_response = True
                    break
        except Exception as e:
            # Optional: log the exception e
            print(e)
            pass
        time.sleep(retry_delay)
    result = 'subset' if rel == 'yes' else 'not subset'
    if received_valid_response and subsumption_cache is not None:
        subsumption_cache[cache_key] = result
        if persist_subsumption_cache is not None:
            persist_subsumption_cache()
    return result
