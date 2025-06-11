# %%
# Core libraries
import pandas as pd
import numpy as np
import networkx as nx
import pickle


# %%
# Parameters
chunk_size = 100_000
input_file = 'soc-redditHyperlinks-body.tsv'

# Build directed graph from Reddit hyperlink data
G = nx.DiGraph()
total_rows = sum(1 for _ in open(input_file)) - 1
reader = pd.read_csv(input_file, sep='\t', comment='#', chunksize=chunk_size)
rows_processed = 0

for chunk in reader:
    for _, row in chunk.iterrows():
        src = row['SOURCE_SUBREDDIT']
        tgt = row['TARGET_SUBREDDIT']
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += 1
        else:
            G.add_edge(src, tgt, weight=1)
    rows_processed += len(chunk)
    print(f"Processed {rows_processed}/{total_rows} rows ({rows_processed/total_rows:.2%})")

print(f"Graph construction complete. Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")


# %%
# Save the graph object for later use
with open('reddit_graph.pkl', 'wb') as f:
    pickle.dump(G, f)
print("Graph saved to reddit_graph.pkl")


# %%



