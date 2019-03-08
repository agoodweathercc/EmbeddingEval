# https://github.com/keras-team/keras/issues/4613
import sys
import os
import json
import networkx as nx
import tensorflow as tf
from time import time
import numpy as np
import argparse

sys.path.append('/home/cai.507/Documents/DeepLearning/EmbeddingEval/node_classification/NRL/src/')
import classification
# sys.path.append('/home/cai.507/Documents/DeepLearning/EmbeddingEval/NRL/src/')
sys.path.append('/home/cai.507/Documents/DeepLearning/deep-persistence/pythoncode/GEM/')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
np.random.seed(1337)

from gem.utils      import graph_util
from gem.utils.logutil      import set_logger, make_direct
from gem.evaluation import evaluate_node_classification as nc
from gem.embedding.lap import LaplacianEigenmaps
from gem.embedding.sdne     import SDNE
from gem.embedding.sdne_utils import model_batch_predictor
from aux import batch_generator_sdne_modified, batch_generator_sdne, get_search_num, get_one_hyper, make_hyper_direct

def load_hyperparameter(method='lap', dataset = 'karate'):
    # method can be ['node2vec', 'sdne', 'gf', 'hope', 'lap', 'lle']
    # hyper_dict = open('./blogcat_hypRange_1.txt').read()
    # hyper_dict = open('./blogcat_hypRange_1.txt').read()
    hyper_dict = open('./parameters.txt').read()
    import ast
    hyper_dict = ast.literal_eval(hyper_dict)
    if dataset in ['texas', 'cornell', 'wisconsin', 'washington']:
        hyper_dict['sdne']['n_batch'] = [50]
    if dataset in ['flickr', 'microsoft', 'p2p-gnutella31']:
        # hyper_dict['sdne']['n_iter'] = [[5], [10]]
        hyper_dict['sdne']['n_iter'] = [[5]]
    if dataset in ['wikipedia']:
        hyper_dict['sdne']['n_iter'] = [[50]]
    if dataset in ['blogcatalog']:
        hyper_dict['sdne']['n_iter'] = [[50],[20]]
        hyper_dict['sdne']['alpha'] = [1e-05]
        hyper_dict['sdne']['beta'] = [5]

    return hyper_dict[method]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='wikipedia', help="load from data/")
parser.add_argument('--method', default='sdne', help='node2vec, sdne, gf, hope, lap, lle')
parser.add_argument('--task', default='nc', help="nc or lp")
parser.add_argument('--fold', default='0', help="0, 1, 2, 3, 4")
parser.add_argument('--newloss', default=False)

if __name__ == '__main__':
    # sys.argv = ['--dataset==co-author']
    args = parser.parse_args()
    method = args.method
    method2func = {'sdne': 'SDNE', 'node2vec': 'node2vec',
                   'gf': 'GraphFactorization', 'hope':'HOPE',
                   'lap':'LaplacianEigenmaps', 'lle': 'LocallyLinearEmbedding'}
    dataset = args.dataset
    edge_f, labels_f, mat_f = './data/' + dataset + '.edgelist', './data/' + dataset + '.npy', './data/' + dataset + '.mat'
    isDirected = True if dataset in ['co-author', 'pubmed', 'wikivote'] else False

    # Load graph
    if args.task == 'nc':
        G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=isDirected)
    elif args.task == 'lp':
        G = graph_util.loadGraphForLP(args)
        if isDirected: G = G.to_directed()
    else:
        raise Exception('No such task')

    print(nx.info(G))
    n_nodes = len(G)
    print('finish converting...')
    if dataset != 'karate': labels = np.load(labels_f)

    searchrange = load_hyperparameter(method = method, dataset=dataset) # a dict
    n_models = get_search_num(searchrange)[0]
    models = []
    for idx in range(n_models):
        hyper = get_one_hyper(searchrange, dataset, idx=idx, method=method)
        make_hyper_direct(method=method, dataset=dataset, make_flag=True, **hyper)
        models.append(eval(method2func[method])(**hyper))
    print('finish easy work')

    for embedding in models:
        t1 = time()
        direct = embedding.__getattribute__('_direct')
        make_direct(direct)
        logger = set_logger(os.path.join(direct, 'train_' + args.task + '.log'), reset_flag=True)
        logger.info('Start Training...')
        logger.info('directory is %s' %direct)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        name = 'fold_' + args.fold if args.task == 'lp' else 'emb_nc'

        # if emb already exists, then skip
        if os.path.exists(direct + name + '.npy'):
            print('The following already exist')
            print(direct + name)
            continue

        if True: # generate embedding or load existing embedding
            if method == 'sdne':
                self = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
                model = self._model
                print('finish getting self')
                S = nx.to_scipy_sparse_matrix(G)

                my_generator = batch_generator_sdne_modified(S, n_nodes, searchrange['beta'][0], searchrange['n_batch'][0], True)
                my_generator_ = batch_generator_sdne(S, n_nodes, searchrange['beta'][0], searchrange['n_batch'][0], True)
                print('finish setting generator')

                history_callback = model.fit_generator(
                    my_generator_,
                    nb_epoch= searchrange['n_iter'][0][0],
                    samples_per_epoch=n_nodes // searchrange['n_batch'][0],
                    validation_data=None, class_weight=None, nb_worker=10,
                    verbose=1, max_queue_size = 5)
                Y, t = model_batch_predictor(self._autoencoder, S, self._n_batch), 0

                for epoch in history_callback.epoch:
                    for key, val in history_callback.history.items():
                        logger.info(str(epoch) + ' ' + str(key) + ': ' + str(val[epoch])),
            else:
                Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True, logger=logger)

            # save embedding
            name = 'fold_' + args.fold  if args.task == 'lp' else 'emb_nc'
            np.save(direct + name, Y)
            # np.savetxt(direct + 'embedding_.txt', Y)
            logger.info (embedding._method_name+':\n\tTraining time: %f, save file at %s' % (time() - t1, direct+name))
            logger.info('Finish evaluating static graph reconstruction\n\n\n')

            # save param
            with open(direct+ 'param_' + args.task + '.json', 'w') as fp:
                json.dump(embedding.__getattribute__('_param'), fp, indent=1)
            continue

        # code that is no longer being used
        try:
            training_percents = np.linspace(0.1, 0.9, 9) if dataset not in ['flickr', 'youtube','co-author'] else np.linspace(0.01, 0.10, 10)
            training_percents = list(training_percents)
            print(direct + name)
            logger.info(classification.classify(emb=direct + name + '.npy', network=mat_f, writetofile=True, classifier='LR',
                                        dataset=dataset, algorithm=method, word2vec_format=False,
                                        num_shuffles=5, output_dir=direct,
                                        training_percents=training_percents))
            #
            # logger.info(classification.classify(emb=direct + name + '.npy', network=mat_f, writetofile=True, classifier='EigenPro',
            #                             test_kernel="eigenpro",
            #                             dataset=dataset, algorithm=method, word2vec_format=False,
            #                             num_shuffles=5, output_dir=direct,
            #                             training_percents=training_percents))
            sys.exit()

        except:
            logger.info('Evaluation Error')
        continue
#######################################################################################################################

