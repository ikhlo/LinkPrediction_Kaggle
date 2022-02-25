import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from time import time, strftime, gmtime
from feats_eng.processing import read_dataframe, graph_feats
from feats_eng.generate_train_test_set import generate

parent_path = r"C:\Users\ikhla\OneDrive - De Vinci\A5_2021_2022\Altegrad\Challenge\code_env\\"
processed_path = r"C:\Users\ikhla\OneDrive - De Vinci\A5_2021_2022\Altegrad\Challenge\code_env\processed_data"

#PART 1 ---------------------

start = time()
print('Create dataframe and process abstracts...')
df = pd.DataFrame.from_dict({
        "authors" : processing.read_authors_txt(os.path.join(parent_path, r'data\authors.txt')),
        "abstract" : processing.read_abstracts_txt(os.path.join(parent_path, r'data\abstracts.txt'))
    })

tqdm.pandas(ncols=99)
df['abstract'] = df['abstract'].progress_apply(processing.process_abstract_text)
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')
df.to_csv(os.path.join(processed_path, 'processed_dataframe.csv'))
"""

#PART 2 -------------------------

"""
start = time()
print('Load abstract and authours data..')
df = read_dataframe(os.path.join(processed_path, 'processed_dataframe.csv'))
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')

start = time()
print('Load graph data...')
edge_file = os.path.join(parent_path, r'data\edgelist.txt')
G = nx.read_edgelist(edge_file, delimiter=',', create_using=nx.Graph(), nodetype=int)
df_graph = graph_feats(G)
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')

node_feats = df.join(df_graph)
node_feats.index.name = "node"

node_feats.to_csv(os.path.join(processed_path, 'node_feats_dataframe.csv'))
print('Dataframe with nodes features saved.\n')
"""

#PART 3 -----------------

"""# LOAD GRAPH AND NODES FEATURES & EMBEDDINGS
start = time()
print('Load graph and nodes features...')
edge_file = os.path.join(parent_path, r'data\edgelist.txt')
G = nx.read_edgelist(edge_file, delimiter=',', create_using=nx.Graph(), nodetype=int)
node_feats = read_dataframe(os.path.join(processed_path, 'node_feats_dataframe.csv'))
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')

#print(node_feats.head(5))

start = time()
print('Create datasets...')
train, test = generate(G, node_feats, os.path.join(parent_path, r'data\test.txt'), n_rows=100000, valid_set=False)
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')

start = time()
print('Dump datasets...')
with open(os.path.join(processed_path, "train_22B_noremove.pkl"),"wb") as f:
    pickle.dump(train[0], f) #X_train
    pickle.dump(train[1], f) #y_train

with open(os.path.join(processed_path, "test_22B_noremove.pkl"), "wb") as f:
    pickle.dump(test, f) #X_test

print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')


#PART 4 -----------------

from models.kfold_fit import fitting

start = time()
print('Training model...')

train = Path('processed_data') / 'train_22B_noremove.pkl'
test = Path('processed_data') / 'test_22B_noremove.pkl'

with open(train, 'rb') as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)

with open(test, 'rb') as f:
    X_test = pickle.load(f)

model_params = {"n_estimators":128, "max_depth":4,
"learning_rate":0.15, "colsample_bytree" : 0.6}

fitting(
    'xgb', X_train, y_train, X_test, 
    submission_name='submission_xgb_22B_nr_col_200k.csv', 
    kfold_value=10, 
    **model_params
)
print(f'DONE - {strftime("%H:%M:%S", gmtime(time()-start))}\n')
## ___________________DRAFT______________________
