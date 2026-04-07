#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperdimensional Memory with RG‑Based Cluster Splitting on Real Text
===================================================================
Uses golden‑ratio bundling (α, β) to convert text into hypervectors.
Maintains clusters online; splits when similarity < 1/φ².
Queries retrieve the best matching cluster.

Data: 20 Newsgroups (subset) – install scikit-learn if not available.
"""

import numpy as np
import math
import hashlib
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Golden‑ratio constants
PHI = (1 + math.sqrt(5)) / 2
ALPHA = 1 / PHI
BETA = 1 / PHI**2
RG_THRESHOLD = 1 / PHI**2          # ≈ 0.382
DIM = 3819                         # optimal dimension (reduce to 100 for speed)

def hv_from_bytes(data):
    """Convert bytes to hypervector using golden‑ratio bundling (α, β)."""
    hv = np.zeros(DIM)
    n = len(data)
    for i in range(n):
        # Deterministic base hypervector for each byte value (0-255)
        seed = data[i]
        np.random.seed(seed)
        base = np.random.randn(DIM)
        base /= np.linalg.norm(base)
        hv += ALPHA * base
        if i < n-1:
            np.random.seed(data[i+1])
            base_next = np.random.randn(DIM)
            base_next /= np.linalg.norm(base_next)
            hv += BETA * base_next
    norm = np.linalg.norm(hv)
    if norm > 0:
        hv /= norm
    return hv

def hv_from_text(text):
    """Convert a text string to hypervector (UTF-8 bytes)."""
    return hv_from_bytes(text.encode('utf-8'))

def causal_similarity(u, v):
    diff = np.abs(u - v)
    return np.mean(np.exp(-diff / PHI))

class HyperdimensionalMemory:
    def __init__(self, threshold=RG_THRESHOLD):
        self.threshold = threshold
        self.clusters = []          # list of (cluster_hv, list_of_docs)
        self.doc_hvs = []           # store each doc's hv for retrieval (optional)

    def add_document(self, text, metadata=None):
        hv = hv_from_text(text)
        if not self.clusters:
            # first document: create first cluster
            self.clusters.append((hv.copy(), [metadata]))
            self.doc_hvs.append((hv, metadata))
            return

        # Find closest cluster
        best_sim = -1
        best_idx = -1
        for idx, (cluster_hv, _) in enumerate(self.clusters):
            sim = causal_similarity(hv, cluster_hv)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.threshold:
            # Add to existing cluster: update cluster hypervector (bundling)
            cluster_hv, docs = self.clusters[best_idx]
            # Update cluster hypervector: weighted average
            new_hv = (cluster_hv + hv) / np.linalg.norm(cluster_hv + hv)
            self.clusters[best_idx] = (new_hv, docs + [metadata])
        else:
            # Create new cluster
            self.clusters.append((hv.copy(), [metadata]))
        self.doc_hvs.append((hv, metadata))

    def query(self, query_text, top_k=1):
        """Return the most similar cluster and its documents."""
        q_hv = hv_from_text(query_text)
        best_sim = -1
        best_cluster = None
        for cluster_hv, docs in self.clusters:
            sim = causal_similarity(q_hv, cluster_hv)
            if sim > best_sim:
                best_sim = sim
                best_cluster = docs
        return best_cluster, best_sim

# ============================================================
# Demonstration with 20 Newsgroups dataset
# ============================================================
def main():
    print("Loading 20 Newsgroups dataset (subset)...")
    # Use a small subset for demonstration (e.g., 500 documents)
    newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    texts = newsgroups.data[:500]          # take first 500 documents
    labels = newsgroups.target[:500]
    target_names = newsgroups.target_names

    print(f"Loaded {len(texts)} documents. Building hyperdimensional memory...")
    memory = HyperdimensionalMemory(threshold=RG_THRESHOLD)

    for i, text in enumerate(texts):
        memory.add_document(text, metadata={'index': i, 'label': target_names[labels[i]]})
        if (i+1) % 100 == 0:
            print(f"  Processed {i+1} documents, now {len(memory.clusters)} clusters")

    print(f"\nMemory built with {len(memory.clusters)} clusters (threshold = {RG_THRESHOLD:.3f})")

    # Example query
    query = "What is the best way to invest money?"
    print(f"\nQuery: '{query}'")
    cluster_docs, sim = memory.query(query)
    print(f"Best matching cluster (similarity {sim:.3f}) contains {len(cluster_docs)} documents.")
    print("Sample documents from that cluster:")
    for doc in cluster_docs[:3]:
        print(f"  - {doc['label']} (index {doc['index']})")
        # Print a snippet of the original text (first 200 chars)
        snippet = texts[doc['index']][:200].replace('\n', ' ')
        print(f"    Snippet: {snippet}...")

    # Show cluster statistics
    print("\nCluster sizes:")
    sizes = [len(docs) for _, docs in memory.clusters]
    sizes.sort(reverse=True)
    for i, sz in enumerate(sizes[:10]):
        print(f"  Cluster {i+1}: {sz} documents")

if __name__ == "__main__":
    main()
