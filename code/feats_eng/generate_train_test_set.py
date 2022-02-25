import numpy as np
import os
from tqdm import tqdm
from random import randint, sample, seed
from feats_eng.utils import pref_attachment, ressource_allocation_measure, adamic_adar_measure, all_l_path, len_shortest_path, salton_index
from networkx import dispersion, pagerank, eigenvector_centrality
from pathlib import Path
from sklearn.decomposition import PCA


def generate(graph, nodes_df, test_set_path, n_rows=50000, valid_set=False, valid_ratio=0.2, seed_value=42):
    """Use the following features for each pair of nodes:
    (1) sum of number of unique terms of the two nodes' abstracts
    (2) absolute value of difference of number of unique terms of the two nodes' abstracts
    (3) number of common terms between the abstracts of the two nodes
    (4) jaccard similarity between the abstracts of the two nodes
    (5) sum of degrees of two nodes
    (6) absolute value of difference of degrees of two nodes
    (7) number of common authors of the two nodes
    (8) jaccard similarity between the authors of the two nodes 
    (9) sum of authors of two nodes 
    (10) number of common linked paper of the two nodes
    (11) jaccard similarity between linked paper of the two nodes
    (12) ressource allocation index of the two nodes
    (13) adamic / adar index of the two nodes 
    (14) number of path of length 3 between the two nodes
    (15) dispersion between the two nodes
    (16) check if the size of the sortest path between the two nodes is less than k
    (17) preferential attachment of the two nodes
    (18) salton index of the two nodes
    (19) absolute difference of the pageranks of the two nodes
    (20) absolute difference of the eigenvector centrality of the two nodes
    (21) sum of absolute difference of two papers deepwalk embedding
    (22) sum of absolute difference of two papaers abstract bert embedding"""

     # SET SEED
    seed(seed_value)

    n = graph.number_of_nodes()
    print(f'Number of ndoes : {n}')
    m = graph.number_of_edges()
    print(f'Number of edges : {m}')
    
    n_rows = min(n_rows, m)
    X_train = np.zeros((2 * n_rows, 22))
    y_train = np.zeros(2 * n_rows)

    #DEEPWALK node embeddings
    deepwalk_emb_fname = Path('processed_data') / 'node_embedding_deepwalk.npy'
    deepwalk_emb = np.load(deepwalk_emb_fname.absolute())

    #BERT abstract embeddings
    bert_emb_fname = Path('processed_data') / 'bert_embeddings.npy'
    bert_emb = np.load(bert_emb_fname.absolute())
    print('Embeddings have been loaded !')

    pageranks = pagerank(graph) # dictionnary of nodes with PageRank as value
    print('Pagerank have been loaded !')

    eigenvectors_centralities = eigenvector_centrality(graph) # dictionnary of nodes with eigenvector centrality as value
    print('Eigenvectors have been loaded !')

    fname = Path('processed_data') / 'pair_of_nodes.txt'
    graph_edges = graph.edges()
    sample_train = sample(list(graph_edges), n_rows)

    with open(fname, 'w') as file:
        for i, edge in enumerate(tqdm(sample_train)):
            # an edge
            file.write(f'{edge[0]},{edge[1]}\n')
            node1 = nodes_df.loc[edge[0]]
            node2 = nodes_df.loc[edge[1]]

            X_train[2 * i, 0] = len(set(node1['abstract'])) + len(set(node2['abstract']))
            X_train[2 * i, 1] = abs(len(set(node1['abstract'])) - len(set(node2['abstract'])))
            X_train[2 * i, 2] = len(set(node1['abstract']).intersection(set(node2['abstract'])))
            X_train[2 * i, 3] = len(set(node1['abstract']).intersection(set(node2['abstract']))) / (len(set(node1['abstract']).union(set(node2['abstract']))) + 1E-8)
            X_train[2 * i, 4] = node1['degree'] + node2['degree']
            X_train[2 * i, 5] = abs(node1['degree'] - node2['degree'])
            X_train[2 * i, 6] = len(set(node1['authors']).intersection(set(node2['authors'])))
            X_train[2 * i, 7] = len(set(node1['authors']).intersection(set(node2['authors']))) / len(set(node1['authors']).union(set(node2['authors'])))
            X_train[2 * i, 8] = len(set(node1['authors'])) + len(set(node2['authors']))
            X_train[2 * i, 9] = len(set(node1['neighbors']).intersection(set(node2['neighbors'])))
            X_train[2 * i, 10] = len(set(node1['neighbors']).intersection(set(node2['neighbors']))) / len(set(node1['neighbors']).union(set(node2['neighbors'])))
            X_train[2 * i, 11] = ressource_allocation_measure(graph, edge[0], edge[1])
            X_train[2 * i, 12] = adamic_adar_measure(graph, edge[0], edge[1])
            X_train[2 * i, 13] = all_l_path(graph, edge[0], edge[1], (3,))
            X_train[2 * i, 14] = dispersion(graph, edge[0], edge[1])
            X_train[2 * i, 15] = 1.0 if len_shortest_path(graph, edge[0], edge[1]) < 11 else 0.0
            X_train[2 * i, 16] = pref_attachment(graph, edge[0], edge[1])
            X_train[2 * i, 17] = salton_index(graph, edge[0], edge[1])
            X_train[2 * i, 18] = np.abs(pageranks[edge[0]] - pageranks[edge[1]])
            X_train[2 * i, 19] = np.abs(eigenvectors_centralities[edge[0]] - eigenvectors_centralities[edge[1]])
            X_train[2 * i, 20] = np.sum(np.abs(deepwalk_emb[edge[0]] - deepwalk_emb[edge[1]]))
            X_train[2 * i, 21] = np.sum(np.abs(bert_emb[edge[0]] - bert_emb[edge[1]]))
            y_train[2 * i] = 1

            # a randomly generated pair of nodes
            rand1 = randint(0, n-1)
            rand2 = randint(0, n-1)
            while (rand1, rand2) in graph_edges or rand1 == rand2:
                rand1 = randint(0, n-1)
                rand2 = randint(0, n-1)
            file.write(f'{rand1},{rand2}\n')
            n1 = nodes_df.loc[rand1]
            n2 = nodes_df.loc[rand2]

            X_train[2 * i + 1, 0] = len(set(n1['abstract'])) + len(set(n2['abstract']))
            X_train[2 * i + 1, 1] = abs(len(set(n1['abstract'])) - len(set(n2['abstract'])))
            X_train[2 * i + 1, 2] = len(set(n1['abstract']).intersection(set(n2['abstract'])))
            X_train[2 * i + 1, 3] = len(set(n1['abstract']).intersection(set(n2['abstract']))) / (len(set(n1['abstract']).union(set(n2['abstract']))) + 1E-8)
            X_train[2 * i + 1, 4] = n1['degree'] + n2['degree']
            X_train[2 * i + 1, 5] = abs(n1['degree'] - n2['degree'])
            X_train[2 * i + 1, 6] = len(set(n1['authors']).intersection(set(n2['authors'])))
            X_train[2 * i + 1, 7] = len(set(n1['authors']).intersection(set(n2['authors']))) / len(set(n1['authors']).union(set(n2['authors'])))
            X_train[2 * i + 1, 8] = len(set(n1['authors'])) + len(set(n2['authors']))
            X_train[2 * i + 1, 9] = len(set(n1['neighbors']).intersection(set(n2['neighbors'])))
            X_train[2 * i + 1, 10] = len(set(n1['neighbors']).intersection(set(n2['neighbors']))) / len(set(n1['neighbors']).union(set(n2['neighbors'])))
            X_train[2 * i + 1, 11] = ressource_allocation_measure(graph, rand1, rand2)
            X_train[2 * i + 1, 12] = adamic_adar_measure(graph, rand1, rand2)
            X_train[2 * i + 1, 13] = all_l_path(graph, rand1, rand2, (3,))
            X_train[2 * i + 1, 14] = dispersion(graph, rand1, rand2)
            X_train[2 * i + 1, 15] = 1.0 if len_shortest_path(graph, rand1, rand2) < 11 else 0.0
            X_train[2 * i + 1, 16] = pref_attachment(graph, rand1, rand2)
            X_train[2 * i + 1, 17] = salton_index(graph, rand1, rand2)
            X_train[2 * i + 1, 18] = np.abs(pageranks[rand1] - pageranks[rand2])
            X_train[2 * i + 1, 19] = np.abs(eigenvectors_centralities[rand1] - eigenvectors_centralities[rand2])
            X_train[2 * i + 1, 20] = np.sum(np.abs(deepwalk_emb[rand1] - deepwalk_emb[rand2]))
            X_train[2 * i + 1, 21] = np.sum(np.abs(bert_emb[rand1] - bert_emb[rand2]))
            y_train[2 * i + 1] = 0

    print('Size of training matrix:', X_train.shape)
    # SAVE TXT PAIR OF NODES

    # GENERATE TEST MATRIX
    node_pairs = list()
    with open(test_set_path, 'r') as f:
        for line in f:
            t = line.split(',')
            node_pairs.append((int(t[0]), int(t[1])))

    # Create the test matrix. Use the same features as above
    X_test = np.zeros((len(node_pairs), 22))
    for i, node_pair in enumerate(tqdm(node_pairs)):
        node1 = nodes_df.loc[node_pair[0]]
        node2 = nodes_df.loc[node_pair[1]]
        
        X_test[i, 0] = len(set(node1['abstract'])) + len(set(node2['abstract']))
        X_test[i, 1] = abs(len(set(node1['abstract'])) - len(set(node2['abstract'])))
        X_test[i, 2] = len(set(node1['abstract']).intersection(set(node2['abstract'])))
        X_test[i, 3] = len(set(node1['abstract']).intersection(set(node2['abstract']))) / (len(set(node1['abstract']).union(set(node2['abstract']))) + 1E-8)
        X_test[i, 4] = node1['degree'] + node2['degree']
        X_test[i, 5] = abs(node1['degree'] - node2['degree'])
        X_test[i, 6] = len(set(node1['authors']).intersection(set(node2['authors'])))
        X_test[i, 7] = len(set(node1['authors']).intersection(set(node2['authors']))) / len(set(node1['authors']).union(set(node2['authors'])))
        X_test[i, 8] = len(set(node1['authors'])) + len(set(node2['authors']))
        X_test[i, 9] = len(set(node1['neighbors']).intersection(set(node2['neighbors'])))
        X_test[i, 10] = len(set(node1['neighbors']).intersection(set(node2['neighbors']))) / len(set(node1['neighbors']).union(set(node2['neighbors'])))
        X_test[i, 11] = ressource_allocation_measure(graph, node_pair[0], node_pair[1])
        X_test[i, 12] = adamic_adar_measure(graph, node_pair[0], node_pair[1])
        X_test[i, 13] = all_l_path(graph, node_pair[0], node_pair[1], (3,))
        X_test[i, 14] = dispersion(graph, node_pair[0], node_pair[1])
        X_test[i, 15] = 1.0 if len_shortest_path(graph, node_pair[0], node_pair[1]) < 11 else 0.0
        X_test[i, 16] = pref_attachment(graph, node_pair[0], node_pair[1])
        X_test[i, 17] = salton_index(graph, node_pair[0], node_pair[1])
        X_test[i, 18] = np.abs(pageranks[node_pair[0]] - pageranks[node_pair[1]])
        X_test[i, 19] = np.abs(eigenvectors_centralities[node_pair[0]] - eigenvectors_centralities[node_pair[1]])
        X_test[i, 20] = np.sum(np.abs(deepwalk_emb[node_pair[0]] - deepwalk_emb[node_pair[1]]))
        X_test[i, 21] = np.sum(np.abs(bert_emb[node_pair[0]] - bert_emb[node_pair[1]]))

    print('Size of test matrix:', X_test.shape)

    if valid_set:
        from sklearn.model_selection import train_test_split
        X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(X_train, y_train, test_size=valid_ratio, random_state=10)
        return (X_subtrain, y_subtrain), (X_valid, y_valid), X_test
    else:
        return (X_train, y_train), X_test