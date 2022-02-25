# -*- coding: utf-8 -*-
"""DC_ALTEGRAD.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QlVkAOggM73U1e5ZRsTXfTtLpxqxavcr

#Imports
"""

"""from google.colab import drive
drive.mount("MyDrive")

# Commented out IPython magic to ensure Python compatibility.
# %cd MyDrive/MyDrive/Challenge/"""

!pip install transformers
!pip install -U sentence-transformers --quiet
from google.colab import drive
import networkx as nx
import csv
import numpy as np
from tqdm import tqdm
import torch
import transformers
from transformers import DistilBertTokenizerFast
from transformers import DistilBertModel, DistilBertTokenizerFast
from random import randint, sample
import random

"""#Read Graph

In this section we read a graph and form a train/test datasets. In order to prevent the model from training on ground truth, we drop edges for pairs in the dataset.\
First, 15% of edges are removed from the graph, the pairs for these edges form a validation dataset. The same amount of pairs without edges is sampled at random.\
Next, we sample 200k edges from graphto use as a training set, 60% of these edges are also removed. The obtained graph is denoted as ```G_train```
"""

G = nx.read_edgelist('/data/edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

random.seed(43)
EDGE_DROP_TRAIN_RATE = 0.6
G_train = G.copy()
test_size = int(m * 0.15) 
test_pairs = []
test_labels = []
for i,edge in enumerate(tqdm(sample(list(G_train.edges()), test_size))):

    n1 = randint(0, n-1)
    n2 = randint(0, n-1)
    while (n1, n2) in G_train.edges() or n1 == n2:
        n1 = randint(0, n-1)
        n2 = randint(0, n-1)

    test_pairs.append([edge[0], edge[1]])
    test_labels.append(1)
    test_pairs.append([n1, n2])
    test_labels.append(0)
    G_train.remove_edge(edge[0], edge[1]) #drop edge in test set


train_pairs = []
train_labels = []


train_abstracts = {}
for i,edge in enumerate(tqdm(sample(list(G_train.edges()), 200000))):

    n1 = randint(0, n-1)
    n2 = randint(0, n-1)
    while (n1, n2) in G_train.edges() or n1 == n2:
        n1 = randint(0, n-1)
        n2 = randint(0, n-1)

    train_pairs.append([edge[0], edge[1]])
    train_labels.append(1)
    train_pairs.append([n1, n2])
    train_labels.append(0)
    remove_edge = np.random.rand() < EDGE_DROP_TRAIN_RATE #remove edge in train set with some probability
    if remove_edge:
        G_train.remove_edge(edge[0], edge[1])

LOAD_TEXT_FEATURES = False
LOAD_AUTHOR_FEATURES = False
LOAD_PAIR_FEATURES = False

"""
#BERT MODEL

We load a pre-trained BERT to obtain abstract embeddings
"""

import torch.nn as nn
from sentence_transformers import SentenceTransformer

!gdown --id 1Z7rU8crJcgZJoScxPcRre9mdScioxDO1

if not LOAD_TEXT_FEATURES:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #model = SentenceTransformer('average_word_embeddings_glove.6B.300d')

    encodings = model.encode(list(abstracts.values()), device='cuda', show_progress_bar=True, batch_size=256, convert_to_numpy=True)

    #np.save("embeddings.npy", encodings)
else:
    !gdown --id 1233WIYYiiKavkFHIXI9R8niky1V-bBNm
    embeddings = np.load("/processed_data/embeddings.npy")

"""
#PROCESS AUTHORS

In this block we extract all the needed information about aithors to create emperical features and author embeddins
"""

def process_line_of_authors(line):
    paper_id, authors = line.rstrip('\n').split('|--|')
    paper_id = int(paper_id)
    authors = authors.split(',')
    return (paper_id, authors)

def len_shortest_p(graph, src, target):
  try :
    return nx.shortest_path_length(graph, src, target)
  except :
    return 10000

with open('authors.txt', 'r') as f:
  l = f.readlines()
  paper_to_authors = dict(map(process_line_of_authors, l))

# LIST WITH ALL AUTHORS
All_authors = []
for authors in paper_to_authors.values():
    All_authors.extend(authors)
All_authors = set(All_authors)

#DICTIONNARY DICT[AUTHOR] + AUTHOR_ID
author_to_authorid = {author : i+1 for i, author in enumerate(All_authors)}

#DICTIONNARY DICT[PAPE_ID] = LIST[AUTHORS_ID]
paper_to_authorsid = {i : list(map(lambda x: author_to_authorid[x], authors)) for i, authors in paper_to_authors.items()}

"""
AUTHORS GRAPH

In this cell two graphs of authors are created, one is the graph of citation, another is a graph of cooperation"""

def add_edge_or_increase_weight(G, u, v): #create edge if it is not in graph or add +1 to its weight otherwise
    if G.has_edge(u,v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, weight=1)
    return

A_cite_graph = nx.Graph()
for edge in G.edges():
    authors_p1 = paper_to_authorsid[edge[0]]
    authors_p2 = paper_to_authorsid[edge[1]]
    #for each pair of athuros of two papers we create edge in the citation graph
    for a1 in authors_p1:
        for a2 in authors_p2:
            add_edge_or_increase_weight(A_cite_graph, a1, a2)

A_coop_graph = nx.Graph()
for a_list in paper_to_authorsid.values():
    #for each paper we look at all its authors and include information in cooperation graph
    for i in range(len(a_list)-1):
        for j in range(i+1, len(a_list)):
            add_edge_or_increase_weight(A_coop_graph, a_list[i],a_list[j])

"""#Node2Vec

Here we run Node2Vec to extract representations of each author from the graph
"""

!pip install fastnode2vec
from fastnode2vec import Node2Vec, Graph
import gensim

if not LOAD_AUTHOR_FEATURES
    edges_list = [(str(e[0]), str(e[1]), A_cite_graph[e[0]][e[1]]["weight"])
                  for e in A_cite_graph.edges]
    g = Graph(edges_list, directed=False, weighted=True)
    node2vec = Node2Vec(g, dim=128, walk_length=15,
                        context=10, p=1, q=0.5, workers=10)
    node2vec.train(epochs=100)
    node2vec.wv.save_word2vec_format('/processed_data/a_cite.nodevectors')

    edges_list_coop = [(str(e[0]), str(e[1]), A_coop_graph[e[0]][e[1]]["weight"])
              for e in A_coop_graph.edges]
    g = Graph(edges_list, directed=False, weighted=True)
    a_coop_model = Node2Vec(g, dim=128, walk_length=15,
                        context=10, p=1, q=0.5, workers=10)
    a_coop_model.train(epochs=100)
    a_coop_model.wv.save_word2vec_format('/processed_data/a_coop.nodevectors')

else:
    ! gdown --id 1-00RDGpf6bvC7De3K1I2aAP5gErgEO9Y
    !gdown --id 1G-uItGZdwtH5E6vNBOlGlefvL4qEbTAO

a_coop_model = gensim.models.KeyedVectors.load_word2vec_format("/processed_data/a_coop.nodevectors")
a_cite_model = gensim.models.KeyedVectors.load_word2vec_format("/processed_data/a_cite.nodevectors")

node_a_embeddings = {}
for node in G_train.nodes():
    a_ids = paper_to_authorsid[int(node)]
    emb_coop = np.mean([a_coop_model.wv[str(a)] for a in a_ids if str(a) in a_coop_model.wv.vocab],axis = 0)
    emb_cite = np.mean([a_cite_model.wv[str(a)] for a in a_ids if str(a) in a_cite_model.wv.vocab],axis = 0)
    node_a_embeddings[node] = np.hstack([emb_coop, emb_cite])

"""#Pair features"""

NB_PAIR_FEATURES = 12

X_pairs_train = torch.zeros((len(train_pairs), NB_PAIR_FEATURES))
for i, edge in enumerate(tqdm(train_pairs)):
    X_pairs_train[i, 0] = G_train.degree(edge[0]) + G_train.degree(edge[1])
    X_pairs_train[i, 1] = abs(G_train.degree(edge[0]) - G_train.degree(edge[1]))
    X_pairs_train[i, 2] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) 
    X_pairs_train[i, 3] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) / len(set(paper_to_authorsid[edge[0]]).union(set(paper_to_authorsid[edge[1]])))
    X_pairs_train[i, 4] = len({n for n in G_train.neighbors(edge[0])}.intersection({n for n in G_train.neighbors(edge[1])}))
    X_pairs_train[ i, 5] = len({n for n in G_train.neighbors(edge[0])}.intersection({n for n in G_train.neighbors(edge[1])})) / (len({n for n in G_train.neighbors(edge[0])}.union({n for n in G_train.neighbors(edge[1])})) + 1)
    X_pairs_train[i, 6] = list(nx.resource_allocation_index(G_train, [(edge[0], edge[1])]))[0][-1]
    X_pairs_train[i, 7] = list(nx.adamic_adar_index(G_train, [(edge[0], edge[1])]))[0][-1]
    X_pairs_train[i, 8] = len(set(paper_to_authorsid[edge[0]])) + len(set(paper_to_authorsid[edge[1]])) 
    X_pairs_train[i, 9] = nx.algorithms.centrality.dispersion(G_train, u=edge[0], v=edge[1])
    X_pairs_train[i, 10] = nx.has_path(G_train, edge[0], edge[1])
    X_pairs_train[i, 11] = len_shortest_p(G_train, edge[0], edge[1]) < 11

X_pairs_test = torch.zeros((len(test_pairs), NB_PAIR_FEATURES))
for i, edge in enumerate(tqdm(test_pairs)):
    X_pairs_test[i, 0] = G_train.degree(edge[0]) + G_train.degree(edge[1])
    X_pairs_test[i, 1] = abs(G_train.degree(edge[0]) - G_train.degree(edge[1]))
    X_pairs_test[i, 2] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) 
    X_pairs_test[i, 3] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) / len(set(paper_to_authorsid[edge[0]]).union(set(paper_to_authorsid[edge[1]])))
    X_pairs_test[i, 4] = len({n for n in G_train.neighbors(edge[0])}.intersection({n for n in G_train.neighbors(edge[1])}))
    X_pairs_test[ i, 5] = len({n for n in G_train.neighbors(edge[0])}.intersection({n for n in G_train.neighbors(edge[1])})) / (len({n for n in G_train.neighbors(edge[0])}.union({n for n in G_train.neighbors(edge[1])})) + 1)
    X_pairs_test[i, 6] = list(nx.resource_allocation_index(G_train, [(edge[0], edge[1])]))[0][-1]
    X_pairs_test[i, 7] = list(nx.adamic_adar_index(G_train, [(edge[0], edge[1])]))[0][-1]
    X_pairs_test[i, 8] = len(set(paper_to_authorsid[edge[0]])) + len(set(paper_to_authorsid[edge[1]])) 
    
    X_pairs_test[i, 9] = nx.algorithms.centrality.dispersion(G_train, u=edge[0], v=edge[1])
    X_pairs_test[i, 10] = nx.has_path(G_train, edge[0], edge[1])
    X_pairs_test[i, 11] = len_shortest_p(G_train, edge[0], edge[1]) < 11

sub_pairs = list()
with open('test.txt', 'r') as f:
    for line in f:
        t = line.split(',')
        sub_pairs.append((int(t[0]), int(t[1])))

# Create the test matrix. Use the same features as above
X_sub = torch.zeros((len(sub_pairs), NB_PAIR_FEATURES))

for i,edge in enumerate(tqdm(sub_pairs)):
    X_sub[i, 0] = G.degree(edge[0]) + G.degree(edge[1])
    X_sub[i, 1] = abs(G.degree(edge[0]) - G.degree(edge[1]))
    X_sub[i, 2] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) 
    X_sub[i, 3] = len(set(paper_to_authorsid[edge[0]]).intersection(set(paper_to_authorsid[edge[1]]))) / len(set(paper_to_authorsid[edge[0]]).union(set(paper_to_authorsid[edge[1]])))
    X_sub[i, 4] = len({n for n in G.neighbors(edge[0])}.intersection({n for n in G.neighbors(edge[1])}))
    X_sub[ i, 5] = len({n for n in G.neighbors(edge[0])}.intersection({n for n in G.neighbors(edge[1])})) / len({n for n in G.neighbors(edge[0])}.union({n for n in G.neighbors(edge[1])}))
    X_sub[i, 6] = list(nx.resource_allocation_index(G, [(edge[0], edge[1])]))[0][-1]
    X_sub[i, 7] = list(nx.adamic_adar_index(G, [(edge[0], edge[1])]))[0][-1]
    X_sub[i, 8] = len(set(paper_to_authorsid[edge[0]])) + len(set(paper_to_authorsid[edge[1]])) 
    
    X_sub[i, 9] = nx.algorithms.centrality.dispersion(G, u=edge[0], v=edge[1])
    X_sub[i, 10] = nx.has_path(G, edge[0], edge[1])
    X_sub[i, 11] = len_shortest_p(G, edge[0], edge[1]) < 11
    #X_sub[ i, 9] = len([path for path in nx.all_simple_edge_paths(G, source=edge[0], target=edge[1], cutoff=3) if len(path) == 3])

"""#GCN

In this block we use our GNN-based architecture as described in the report
"""

!pip install dgl-cu111 -f https://data.dgl.ai/wheels/repo.html

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
from dgl.nn import SAGEConv, GATConv

# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'gcn')
        self.conv2 = SAGEConv(h_feats, h_feats, 'gcn')
        self.conv3 = SAGEConv(h_feats, h_feats, 'gcn')
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=.5)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv2(g, h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv3(g, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.fc1 = nn.Linear(h_feats * 2 + NB_PAIR_FEATURES, h_feats)
        self.fc2 = nn.Linear(h_feats, 1)

    def forward(self, h, X ):
        feat = torch.cat((h[:,0,:], h[:,1,:], X), dim=1)
        x = F.relu(self.fc1(feat))
        return self.fc2(x)


class MLPPredictorSE(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.fc1 = nn.Linear(h_feats, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x ):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
X = torch.tensor(embeddings).to(device)

test_labels = torch.tensor(test_labels).float().to(device)
train_labels = torch.tensor(train_labels).float().to(device)

X_pairs_test = X_pairs_test.to(device)
X_pairs_train = X_pairs_train.to(device)

X_author = torch.zeros(X.shape[0], node_a_embeddings[0].shape[0])
for i, vec in zip(node_a_embeddings.keys(), node_a_embeddings.values()):
    X_author[i] = torch.tensor(vec)

X = torch.cat([X, X_author.to(device)], dim=1)
g_train = dgl.from_networkx(G_train).to(device)
g_train = dgl.add_self_loop(g_train)


gnn_model = GraphSAGE(X.shape[1], 150).to(device)
mlp_model = MLPPredictorSE(150 + NB_PAIR_FEATURES).to(device)
params = list(gnn_model.parameters()) + list(mlp_model.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0005)

epochs = 1000
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, verbose=True, patience=25)
for e in range(epochs):
    optimizer.zero_grad()
    h = gnn_model(g_train, X) #gnn forward pass
    h_train = h[train_pairs] #pick features from train parirs
    h_test = h[test_pairs]#pick features from test parirs
    se_vec = (h_train[:,0,:] - h_train[:,1,:]) ** 2 #compute squared difference 
    se_vec = torch.cat([se_vec, X_pairs_train], dim=-1) # add emperical features
    #scores = (h_train[:,0,:] * h_train[:,1,:]).sum(dim=-1)
    scores = mlp_model(se_vec.squeeze()).squeeze() #MLP forward pass
    with torch.no_grad():
        #same for test
        se_vec_test = (h_test[:,0,:] - h_test[:,1,:]) ** 2
        se_vec_test = torch.cat([se_vec_test, X_pairs_test], dim=-1)
        scores_test = mlp_model(se_vec_test.squeeze()).squeeze()
        test_loss = F.binary_cross_entropy_with_logits(scores_test, test_labels)
        lr_scheduler.step(test_loss)
    loss  = F.binary_cross_entropy_with_logits(scores, train_labels)
    loss.backward()
    optimizer.step()

    print("epoch {},train_loss = {}, test_loss = {}".format(e, loss.item(), test_loss.item()))

from sklearn.metrics import f1_score, accuracy_score

test_prob = torch.sigmoid(scores_test).cpu().numpy()
pred_labels = (test_prob > 0.3) * 1.
y_test_np = test_labels.cpu().numpy()

print(f1_score(y_test_np, pred_labels))

print(accuracy_score(y_test_np, pred_labels))

with torch.no_grad():
    h = gnn_model(g_train, X)
    h_sub = h[sub_pairs]
    se_vec = (h_sub[:,0,:] - h_sub[:,1,:]) ** 2
    se_vec = torch.cat([se_vec, X_sub[:i+1].to(device)], dim=-1)

    scores = mlp_model(se_vec.squeeze()).squeeze()

probs = torch.sigmoid(scores).cpu().detach().numpy()

predictions = zip(range(len(probs)), probs)