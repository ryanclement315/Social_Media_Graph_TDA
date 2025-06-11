# Reddit TDA Project

This project analyzes the structure of Reddit's subreddit hyperlink network using Topological Data Analysis (TDA), persistent homology, and graph-theoretic methods.

## Project Structure

- **build_script.py**  
  Script to build the main directed graph (`G`) from the raw Reddit hyperlink data (`soc-redditHyperlinks-body.tsv`).  
  - Reads the TSV file in chunks.
  - Constructs a weighted directed graph using NetworkX.
  - Saves the graph as `reddit_graph.pkl` for later analysis.

- **Build.ipynb**  
  Jupyter notebook version of `build_script.py` for interactive graph construction and saving.  
  - Useful for step-by-step inspection and debugging.

- **reddit_graph.pkl**  
  Pickled NetworkX DiGraph object containing the subreddit hyperlink network with edge weights.

- **soc-redditHyperlinks-body.tsv**  
  Raw Reddit hyperlink data (tab-separated values).  
  - Each row represents a hyperlink from one subreddit to another.

- **TDA.ipynb**  
  Main notebook for topological and graph-theoretic analysis.  
  - Loads the graph from `reddit_graph.pkl`.
  - Computes persistent homology (using Ripser and Gudhi).
  - Visualizes persistence diagrams, representative cycles, cliques, and subgraphs.
  - Contains functions for clique analysis, subgraph visualization, and Mapper (KeplerMapper) analysis.

- **PH_analysis.ipynb**  
  Notebook focused on persistent homology (PH) cluster analysis.  
  - Extracts significant clusters using H0 features.
  - Provides interactive widgets for exploring clusters and visualizing their structure as directed trees.

- **Reddit_TDA.ipynb**  
  Alternative or earlier notebook for TDA and persistent homology analysis.  
  - Similar in spirit to `TDA.ipynb`, with code for extracting cycles and persistent features.

- **analysis_script.py**  
  Python script with reusable analysis functions.  
  - Includes functions for graph summary, distance matrix computation, persistent homology, clique analysis, and visualization.
  - Can be imported into notebooks or run as a standalone script for batch analysis.

- **README.md**  
  This file. Describes the project structure and the purpose of each file.

## Typical Workflow

1. **Build the Graph**  
   Use `build_script.py` or `Build.ipynb` to process the raw TSV data and create `reddit_graph.pkl`.

2. **Analyze the Graph**  
   Open `TDA.ipynb` or `PH_analysis.ipynb` to:
   - Load the graph.
   - Compute persistent homology and extract topological features.
   - Visualize cycles, clusters, cliques, and Mapper graphs.
   - Use interactive widgets for cluster exploration.

3. **Reusable Analysis**  
   Use functions from `analysis_script.py` for custom or batch analyses.

## Dependencies

- Python 3.7+
- pandas, numpy, networkx, matplotlib, pickle
- ripser, persim, gudhi
- scikit-learn, kmapper (for Mapper analysis)
- ipywidgets, IPython (for interactive widgets in notebooks)

## Data

- `soc-redditHyperlinks-body.tsv` must be present in the project directory to build the graph.
- The processed graph is saved as `reddit_graph.pkl` for efficient reuse.

---

For more details on each analysis step, see the code and markdown cells in the notebooks.