# Taxoble

## Workflow

### 1. Optional: Download PubMed Data

Use `download_data_from_pubmed.ipynb` to fetch PubMed records.

You can skip this step if you already have a CSV file with candidate terms.

### 2. Preprocess Candidate Terms

Edit the configuration block at the bottom of `preprocess_input.py`.

Then run:

```bash
python preprocess_input.py
```

This creates `keyword_mapping.pkl`.

Preprocessing modes:

- `canonicalize`: expands abbreviations and merges synonyms.
- `canonicalize_and_filter_by_domain`: also removes terms that are not relevant to the configured domain.

### 3. Construct the Taxonomy

Open `construct_taxonomy.ipynb` and edit the `TaxonomyConfig` cell.

Run the notebook cells in order.

## Outputs

Final outputs are written to:

```text
results/<folder_name>/<experiment_name>/
```

The directory contains:

- `final_taxonomy.txt`: the readable taxonomy.
- `edges.json`: the taxonomy edge list, represented as `(child, parent)` pairs.
- `nodes.json`: the complete list of taxonomy nodes.
- `term_to_node_mappings.json`: the mapping from terms to their assigned taxonomy nodes.



## Notes

- The pipeline relies on LLM. Review final outputs before using them in downstream analyses.
- Some cycle resolution may require manual input when the LLM cannot confidently repair a graph.
- Large keyword sets can require substantial memory for embedding and similarity computations.



## Citation

If you use this code in a publication or shared project, please cite the repository.