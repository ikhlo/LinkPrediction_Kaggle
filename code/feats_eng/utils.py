
import numpy as np


def literal_eval_nan(string):
    import ast
    return ast.literal_eval(string) if len(string) != 0 else np.nan

def pos_tag_filter(X, good_tags=['NN', 'VB', 'JJ', 'RB']):
    import nltk 
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    pos_list = list(
        map(
            lambda x: x[0],
            filter(
                lambda x: x[1] in good_tags,
                nltk.pos_tag(nltk.word_tokenize(X.lower()))
            )
        )
    )
    return pos_list

def get_less_common_words(all_vocab, n_words=10000, min_count=10):
    from collections import Counter

    big_count = Counter(all_vocab)
    filtered_big_count = Counter()

    temp = dict(filter(lambda x : x[1] > min_count, big_count.items()))
    for key, value in temp.items():
        filtered_big_count[key] = value
    
    return list(map(lambda x: x[0], filtered_big_count.most_common()[-n_words:]))

def intersection_less_common(node1, node2, abstracts_list, less_common_list):
    node1_oh = list(map(lambda x: abstracts_list[node1].count(x), less_common_list))
    node2_oh = list(map(lambda x: abstracts_list[node2].count(x), less_common_list))
    return np.dot(node1_oh,  node2_oh)

def len_shortest_path(graph, src, target):
    from networkx import shortest_path_length
    try:
        return shortest_path_length(graph, src, target)
    except:
        return graph.number_of_nodes() 

def ressource_allocation_measure(graph, node1, node2):
    from networkx import resource_allocation_index
    return list(resource_allocation_index(graph, [(node1, node2)]))[0][-1]

def adamic_adar_measure(graph, node1, node2):
    from networkx import adamic_adar_index
    return list(adamic_adar_index(graph, [(node1, node2)]))[0][-1]


def all_l_path(G, src, target, lengths=(3,)):
    from networkx import all_simple_edge_paths
    max_length = max(lengths)
    result = {length : 0 for length in lengths}
    for path in all_simple_edge_paths(G, source=src, target=target, cutoff=max_length):
        if len(path) in lengths:
            result[len(path)] += 1
    return np.fromiter(result.values(), dtype=float)

def pref_attachment(G, node1, node2):
    from networkx import preferential_attachment
    return list(preferential_attachment(G, [(node1, node2)]))[0][-1]

def salton_index(G, node1, node2):
    from networkx import common_neighbors
    nb_of_commons = sum(1 for _ in common_neighbors(G, node1, node2))
    denominator = np.sqrt(G.degree(node1) * G.degree(node2))
    if denominator == 0 : return 0
    else : return nb_of_commons / denominator