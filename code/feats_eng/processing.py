import pandas as pd

def read_authors_txt(file):
    def process_line(line):
        _, authors_string = line.rstrip('\n').split('|--|')
        return authors_string.split(',')

    with open(file, 'r', encoding='utf-8') as f:
        authors_list = f.readlines()
    
    all_papers_authors_list = list(map(process_line, authors_list))
    return all_papers_authors_list

def read_abstracts_txt(file):
    def process_line(line):
        _, abstract_string = line.rstrip('\n').split('|--|')
        return abstract_string

    with open(file, 'r', encoding='utf-8') as f:
        abstracts_list = f.readlines()
    
    all_papers_abstract_list = list(map(process_line, abstracts_list))
    return all_papers_abstract_list

def process_abstract_text(abstract_string):
    import nltk
    nltk.download('stopwords')
    from feats_eng import utils

    sw = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()

    abstract_tokens = utils.pos_tag_filter(abstract_string)
    """
    abstract_stemmed_tokens = list(
        map(stemmer.stem, filter(lambda token : token not in sw, abstract_tokens))
    )"""
    abstract_stemmed_tokens = [stemmer.stem(token) for token in abstract_tokens if token not in sw]
    return abstract_stemmed_tokens
    
def read_dataframe(file):
    from feats_eng import utils
    converter_dict = {'authors': utils.literal_eval_nan, 'abstract': utils.literal_eval_nan}
    return pd.read_csv(file, index_col = 0, converters=converter_dict)

def graph_feats(graph):
    
    nodes = list(graph.nodes())
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    print('Number of nodes:', n)
    print('Number of edges:', m)

    degree_serie = pd.Series(dict(graph.degree()), name="degree")
    neighbors_serie = pd.Series(
        {n : [neighbors_n for neighbors_n in graph.neighbors(n)] for n in nodes},
        name="neighbors"
    )
    return pd.concat([degree_serie, neighbors_serie], axis=1)

