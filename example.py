import networkx as nx

import numpy as np

from multicom import load_graph, extract_subgraph
from multicom import approximate_ppr, conductance_sweep_cut
from multicom import multicom
from multicom import load_groundtruth, compute_f1_scores
from multicom import plot_nx_clusters

# Synthetic data

print("Load synthetic data...")
adj_matrix = load_graph("data/example_graph.txt")
groundtruth = load_groundtruth("data/example_groundtruth.txt")

print("Apply MULTICOM on seed node 0")
seeds, communities = multicom(adj_matrix, 0, approximate_ppr, conductance_sweep_cut, explored_ratio=.9)

print("Compute F1-scores for detected communities")
f1_scores = compute_f1_scores(communities, groundtruth)
print (f1_scores)

G = nx.from_numpy_matrix(adj_matrix.toarray())
pos = nx.fruchterman_reingold_layout(G)

print("Plot new seeds found by MULTICOM")
plot_nx_clusters(G, [[seed] for seed in seeds], pos)
print("Plot detected communities")
plot_nx_clusters(G, communities, pos, plot_overlaps=True)
print("Plot ground-truth communities")
plot_nx_clusters(G, groundtruth, pos)

# Amazon data
# Data available on: https://snap.stanford.edu/data/com-Amazon.html

print("Load Amazon data...")
adj_matrix = load_graph("data/com-amazon.ungraph.txt")
groundtruth = load_groundtruth("data/com-amazon.all.dedup.cmty.txt")

print("Filter the nodes with degree 0")
degree = np.array(np.sum(adj_matrix, axis=0))[0]
new_adj_matrix, new_groundtruth, node_map = extract_subgraph(adj_matrix, groundtruth, np.where(degree > 0)[0])

print("Apply MULTICOM on seed node 0")
scoring = lambda adj_matrix, seed_set: approximate_ppr(adj_matrix, seed_set, alpha=0.5, epsilon=1e-3)
seeds, communities = multicom(new_adj_matrix, 0, approximate_ppr, conductance_sweep_cut, explored_ratio=.9)

print("Compute the average F1-Score for detected communities")
print(np.mean(compute_f1_scores(communities, new_groundtruth)))
