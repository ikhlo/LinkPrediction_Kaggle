import numpy as np
import pandas as pd
from pathlib import Path

from random import randint
from tqdm import tqdm
from gensim.models import Word2Vec

def random_walk(G, node, walk_length):
    walk = [node]
    for _ in range(walk_length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        neighbor = neighbors[randint(0, len(neighbors) - 1)]
        walk.append(neighbor)
    walk = [str(node) for node in walk]
    return walk

def generate_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        permuted_nodes = np.random.permutation(nodes)
        for node in permuted_nodes:
            walks.append(random_walk(G, node, walk_length))
    return walks

def deepwalk_model(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)
    print("Training Word2Vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)
    return model

def deepwalk_emb(G, n_dim=100, n_walks=50, walk_length=20):

    n = G.number_of_nodes()
    model = deepwalk_model(G, n_walks, walk_length, n_dim)

    nodes = model.wv.index2entity[:n]
    vecs = np.empty(shape=(n, n_dim))

    save_path = Path('..') / 'processed_data' / 'node_embedding_deepwalk.npy'
    with open(save_path, 'wb') as f:
        np.save(f, vecs)