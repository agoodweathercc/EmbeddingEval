import numpy as np
import networkx as nx
import os, sys
dict_data = {0: {'label': (29,), 'neighbors': np.array([])},
 1: {'label': (1,), 'neighbors': np.array([7])},
 2: {'label': (1,), 'neighbors': np.array([8])},
 3: {'label': (1,), 'neighbors': np.array([9])},
 4: {'label': (1,), 'neighbors': np.array([10])},
 5: {'label': (1,), 'neighbors': np.array([11])},
 6: {'label': (1,), 'neighbors': np.array([12])},
 7: {'label': (3,), 'neighbors': np.array([ 1, 15, 17])},
 8: {'label': (3,), 'neighbors': np.array([ 2, 16, 18])},
 9: {'label': (3,), 'neighbors': np.array([ 3, 13, 19])},
 10: {'label': (3,), 'neighbors': np.array([ 4, 14, 20])},
 11: {'label': (3,), 'neighbors': np.array([ 5, 14, 21])},
 12: {'label': (3,), 'neighbors': np.array([ 6, 13, 22])},
 13: {'label': (3,), 'neighbors': np.array([ 9, 12])},
 14: {'label': (3,), 'neighbors': np.array([10, 11])},
 15: {'label': (3,), 'neighbors': np.array([ 7, 23])},
 16: {'label': (3,), 'neighbors': np.array([ 8, 24])},
 17: {'label': (3,), 'neighbors': np.array([ 7, 25])},
 18: {'label': (3,), 'neighbors': np.array([ 8, 26])},
 19: {'label': (3,), 'neighbors': np.array([9])},
 20: {'label': (3,), 'neighbors': np.array([10])},
 21: {'label': (3,), 'neighbors': np.array([11])},
 22: {'label': (3,), 'neighbors': np.array([12])},
 23: {'label': (3,), 'neighbors': np.array([15, 27])},
 24: {'label': (3,), 'neighbors': np.array([16, 28])},
 25: {'label': (3,), 'neighbors': np.array([17, 27])},
 26: {'label': (3,), 'neighbors': np.array([18, 28])},
 27: {'label': (3,), 'neighbors': np.array([23, 25])},
 28: {'label': (3,), 'neighbors': np.array([24, 26])}}

sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/pythoncode/')
sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/GEM/')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode')
debug = 'off'
from preprocessing_global import *
print(GRAPH_TYPE)

def get_dict(graph, id):
    # get the graph i for dataset, represented as a dictionary
    import pickle
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + graph + ".graph"
    if not os.path.isfile(file):
        file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + graph + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    data = data['graph']
    assert type(data[id])==dict
    print data[id]
    return data[id]
if debug=='on':
    dict_data = get_dict('nci1', 72)
    print('get_dict pass')

def dict_to_nx(d):
    # convert a dictionary representation of a graph to nx.graph
    G = nx.Graph()
    for i in range(len(d.items())):
        data = d.items()[i]
        u, label, nbrs = data[0], data[1]['label'][0], data[1]['neighbors']
        G.add_node(u, label=str(label))
        # G[u]['label'] = label
        for v in nbrs:
            G.add_edge(u, v, weight=1)
    return G


def random_insertion(G, n=3):
    nodes = list(G.nodes)
    import random
    e = random.sample(nodes, 2).sort()
    i = 1
    while (i <= n):
        if tuple(e) not in list(G.edges):
            G.add_edge(e[0], e[1])
            i = i + 1
    return G

def random_deletion(G, n = 3):
    import random
    for i in range(n):
        e = random.sample(list(G.edges),1)[0]
        print(e)
        G.remove_edge(e[0],e[1])
    return G

def edge_list_to_nx(graph, id):
    import networkx as nx
    import os.path
    # have not add label yet
    filename = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/emd/edgelist/'  + str(id) + '-1-edgelist'
    assert os.path.exists(filename)==True
    f = open(filename, 'r')
    data = f.readlines()
    assert type(data)==list
    assert len(data)>0
    n = len(data)
    G = nx.Graph()
    for i in range(n):
        tmp = data[i].replace('\n', '').split(' ')
        tmp_data = [int(i) for i in tmp]
        G.add_node(tmp_data[0])
        G.add_node(tmp_data[1])
        G.add_edge(tmp_data[0], tmp_data[1], weight=1)
    return G
if debug == 'on':
    ans = edge_list_to_nx('ptc',3)

if debug=='on':
    G = dict_to_nx(dict_data)
    nx.get_node_attributes(G, 'label')
    print('dict_to_nx pass')

def save_edgelist(graph, id):
    # print ('Saving graph %s'%id)
    # G = get_dict(graph, id)
    import pickle
    file = "/Users/admin/Documents/osu/Research/DeepGraphKernels/datasets/dataset/" + GRAPH_TYPE + ".graph"
    if not os.path.isfile(file):
        file = '/home/cai.507/Documents/DeepLearning/deep-persistence/dataset/datasets/' + GRAPH_TYPE + '.graph'
    f = open(file, 'r')
    data = pickle.load(f)
    for k in range(1,5000):
        G = data['graph'][k]
        print ('.'),
        DIRECTORY = "/Users/admin/Documents/osu/Research/deep-persistence/"
        if os.path.exists(DIRECTORY):
            if not os.path.exists(DIRECTORY + graph + "/emd/edgelist/" ):
                os.makedirs(DIRECTORY + graph + "/emd/edgelist/" )
            DIRECTORY = DIRECTORY + graph + "/emd/edgelist/"
        else:
            DIRECTORY = "/home/cai.507/Documents/DeepLearning/deep-persistence/"
            if not os.path.exists(DIRECTORY + graph + "/emd/edgelist/"):
                os.makedirs(DIRECTORY + graph + "/emd/edgelist/")
            DIRECTORY = DIRECTORY + graph + "/emd/edgelist/"

        filename = str(k) + '-1-edgelist'
        f = open(DIRECTORY + filename, 'w')
        for i in G.keys():
            for j in G[i]["neighbors"]:
                content = str(i) + ' ' + str(j) + '\n'
                f.write(content)
    f.close()

# save_edgelist('ptc', 43)

def G2emd(G, id, graph = 'mutag'):
    DIRECTORY = "/Users/admin/Documents/osu/Research/deep-persistence/"
    if os.path.exists(DIRECTORY):
        if not os.path.exists(DIRECTORY+ graph + "/emd/edgelist/"):
            os.makedirs(DIRECTORY+ graph + "/emd/edgelist/")
        DIRECTORY = DIRECTORY + graph + "/emd/edgelist/"
    else:
        DIRECTORY = "/home/cai.507/Documents/DeepLearning/deep-persistence/"
        if not os.path.exists(DIRECTORY + graph + "/emd/edgelist/"):
            os.makedirs(DIRECTORY + graph + "/emd/edgelist/")
        DIRECTORY = DIRECTORY + graph + "/emd/edgelist/"

    filename = str(id) + '-1-edgelist'
    f = open(DIRECTORY + filename, 'w')
    for e in G.edges():
        content = str(e[0]) + ' ' + str(e[1]) + '\n'
        f.write(content)
        content = str(e[1]) + ' ' + str(e[0]) + '\n'
        f.write(content)
    f.close()
    # f = open(DIRECTORY + filename, 'r')
    # c = f.readlines()
    # f.close()
    # print(c)
if debug=='on':
    G2emd(G,3,'ptc')
    print('G2emd pass')

def emd(GG, graph, model='laplacian', dim=2):
    sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/GEM')
    sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/GEM/gem')
    sys.path.append('/Users/admin/Documents/osu/Research/deep-persistence/GEM/gem/c_exe')
    # import matplotlib.pyplot as plt
    from gem.utils import graph_util, plot_util
    from gem.evaluation import visualize_embedding as viz
    from gem.evaluation import evaluate_graph_reconstruction as gr
    from time import time

    from gem.embedding.gf import GraphFactorization
    from gem.embedding.hope import HOPE
    from gem.embedding.lap import LaplacianEigenmaps
    from gem.embedding.lle import LocallyLinearEmbedding
    from gem.embedding.node2vec import node2vec

    # File that contains the edges. Format: source target
    # Optionally, you can add weights as third column: source target weight
    edge_f = '/Users/admin/Documents/osu/Research/deep-persistence/' + graph + '/emd/edgelist/1-1-edgelist'
    if not os.path.isfile(edge_f):
        edge_f = '/home/cai.507/Documents/DeepLearning/deep-persistence/' + graph + '/emd/edgelist/1-1-edgelist'
    isDirected = True

    # Load graph
    # G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    # G = G.to_directed()
    if GG != None:
        G = GG.to_directed()

    models = []
    if model=='laplacian': models.append(LaplacianEigenmaps(d=dim))
    if model=='lle': models.append(LocallyLinearEmbedding(d=dim))
    if model=='node2vec': models.append(node2vec(d=dim, max_iter=1, walk_len=80, num_walks=10, con_size=10, ret_p=1, inout_p=1))
    if model == 'factorization': models.append(GraphFactorization(d=2, max_iter=1000, eta=1*10**-4, regu=1.0))
    if model == 'hope': models.append(HOPE(d=4, beta=0.01))

    for embedding in models:
        import numpy as np
        # print ('Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges()))
        t1 = time()

        # Learn embedding - accepts a networkx graph or file with edge list
        Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
        # print(np.shape(Y))
        # print (embedding._method_name + ':\n\tTraining time: %f' % (time() - t1))
        # Evaluate on graph reconstruction
        try:
            MAP, prec_curv, err, err_baseline = gr.evaluateStaticGraphReconstruction(G, embedding, Y, None)
        except:
            print ('EvaluateStaticGraphReconstruction Error')
        # print(MAP, err, err_baseline )
        # Visualize
        # viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
        # plt.show()
    assert np.shape(Y)[0] == len(G.nodes)
    return Y

def random_insertion(G, n=3):
    G = G.copy()
    nodes = list(G.nodes)
    import random
    e = random.sample(nodes, 2)
    e.sort()
    i = 1
    while (i <= n):
        if tuple(e) not in list(G.edges):
            G.add_edge(e[0], e[1])
            i = i + 1
        else:
            e = random.sample(nodes, 2)
            e.sort()
    return G
G = dict_to_nx(dict_data)
random_insertion(G)

def random_deletion(G, n = 3):
    G = G.copy()
    import random
    for i in range(n):
        e = random.sample(list(G.edges),1)[0]
        print(e)
        G.remove_edge(e[0],e[1])
    return G

def label_emd(label_info):
    def onehot(a):
        assert type(a) == np.ndarray
        b = np.zeros((a.size, a.max() + 1))
        b[np.arange(a.size), a] = 1
        return b

    def dict2list(d):
        # convert {0: '3', 1: '2'} to a list [3,2]
        # be careful that keys in dict is not ordered
        data = np.zeros((len(d), 2))
        for i in d.keys():
            data[i][0] = i
            data[i][1] = int(d[i])
        # print(data)
        return data
        # implement removing all zero rows
    # onehot(np.np.array([2, 1, 3]))

    label_array = dict2list(label_info)
    label_array = label_array.astype(int)
    label_coordinate = onehot(label_array[:,1])
    # print(np.shape(label_coordinate))
    return label_coordinate
if debug=='on':
    emd_coordinate = emd(dict_to_nx(dict_data), GRAPH_TYPE, 'laplacian', dim=2)
    if isinstance(emd_coordinate, complex):
        emd_coordinate = emd_coordinate.real + 10**(-6)
    G = dict_to_nx(dict_data)
    label_info = nx.get_node_attributes(G, 'label')
    label_coordinate = label_emd(label_info)
    print('label_emd pass')

def coordinate(emd_coordinate, label_coordinate, nolabel=True):
    assert np.shape(emd_coordinate)[0] == np.shape(label_coordinate)[0]
    from sklearn.preprocessing import normalize
    emd_coordinate
    emd_coordinate = normalize(emd_coordinate, axis=0)
    label_coordinate = normalize(label_coordinate, axis=0)
    if nolabel==False:
        coordinate = np.concatenate((emd_coordinate, label_coordinate), axis=1)
    if nolabel == True:
        coordinate = emd_coordinate
    return coordinate
if debug=='on':
    X = coordinate(emd_coordinate, label_coordinate)

def compute_dist(X):
    (n, d) = np.shape(X)
    from scipy.spatial.distance import pdist, squareform
    distance = squareform(pdist(X, 'euclidean'))
    assert np.shape(distance) ==(n,n)
    return distance
if debug=='on':
    distance = compute_dist(X)


def G_emd(G, distance):
    import math
    assert min(G.nodes)==0
    assert len(G.nodes) <= np.shape(distance)[0]
    for e in G.edges():
        assert 'weight' in G[e[0]][e[1]].keys()
        assert e[0] < np.shape(distance)[0]
        assert e[1] < np.shape(distance)[0]
        if math.isnan(distance[e[0]][e[1]]):
            G[e[0]][e[1]]['weight'] = 1 # to aviod weight to be zero
        else:
            G[e[0]][e[1]]['weight'] = distance[e[0]][e[1]] + 10**-4 # to aviod weight to be zero
    # print(dict(nx.shortest_path_length(G, weight='weight')))
    # print(dict(nx.shortest_path_length(G)))

    graph_list = sorted(nx.connected_components(G), key=len, reverse=True)
    G_list = []
    n = len(graph_list)
    # n = 1
    for i in range(n):
        if len(graph_list[i]) > 5:
            G_list.append(G.subgraph(graph_list[i]))
    assert type(G_list) == list
    return G_list
if debug=='on':
    G_emd(G, distance)
    print('G_emd pass')

def graphembedding(graph, id, dim, model='laplacian'):
    # dict_data = get_dict(graph, id)
    # G = dict_to_nx(dict_data)
    G = edge_list_to_nx(graph, id)
    if max(G.nodes()) > len(G.nodes) -1:
        G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    # G2emd(G, id, graph)

    # get the coordinate
    # tmp = dict_to_nx(dict_data)
    # assert nx.is_connected(tmp)
    if len(G.node())< (EMD_DIM + 2):
        return
    emd_coordinate = emd(G, graph, model, dim=dim) + 10**(-6)
    if isinstance(emd_coordinate, complex):
        emd_coordinate = emd_coordinate.real + 10 ** (-6)
    # label_info = nx.get_node_attributes(G, 'label')
    # label_coordinate = label_emd(label_info)

    # X = coordinate(emd_coordinate, label_coordinate)
    X = emd_coordinate
    distance = compute_dist(X)
    print (id)
    G = G_emd(G, distance)
    return G
if debug=='on':
    graph = graphembedding(GRAPH_TYPE, 3, 2, 'lle')
    print('graphembedding pass')






