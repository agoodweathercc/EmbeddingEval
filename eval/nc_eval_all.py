import numpy as np
import sys, os
import argparse
import time
sys.path.append('/home/cai.507/Documents/DeepLearning/EmbeddingEval/node_classification/NRL/src/')
sys.path.append('/home/cai.507/Documents/DeepLearning/EmbeddingEval/link_prediction/NRL/')
from link_prediction_main_fn_new import link_prediction
import classification

from gem.utils.logutil  import set_logger
from aux import batch_generator_sdne_modified, batch_generator_sdne, get_search_num, get_one_hyper, make_hyper_direct
from gem.embedding.sdne     import SDNE
from test import load_hyperparameter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ppi', help="load from data/")
parser.add_argument('--method', default='sdne', help='node2vec, sdne, gf, hope, lap, lle')

if __name__ == '__main__':
    # sys.argv = []
    args = parser.parse_args()
    method = args.method
    method2func = {'sdne': 'SDNE', 'node2vec': 'node2vec',
                   'gf': 'GraphFactorization', 'hope': 'HOPE',
                   'lap': 'LaplacianEigenmaps', 'lle': 'LocallyLinearEmbedding'}
    dataset = args.dataset
    searchrange = load_hyperparameter(method=method, dataset=dataset)  # a dict
    n_models = get_search_num(searchrange)[0]
    reeval_flag = False

    models = []
    for idx in range(n_models):
        hyper = get_one_hyper(searchrange, dataset, idx=idx, method=method)
        make_hyper_direct(method=method, dataset=dataset, make_flag=True, **hyper)
        models.append(eval(method2func[method])(**hyper))
    print('finish easy work')

    for embedding in models:
        direct = embedding.__getattribute__('_direct')
        dataset= args.dataset
        # print(direct)
        nc_emb = os.path.join(direct, 'emb_nc.npy')

        edge_f, labels_f, mat_f = './data/' + dataset + '.edgelist', './data/' + dataset + '.npy', './data/' + dataset + '.mat'
        logger = set_logger(os.path.join(direct, 'eval_nc.log'), reset_flag=True)

        training_percents = np.linspace(0.1, 0.9, 9) if dataset not in ['flickr', 'youtube'] else np.linspace(0.01, 0.10, 10)
        training_percents = list(training_percents)
        n_shuffle = 2 if dataset in ['texas', 'cornell', 'washington', 'wisconsin'] else 1 # for quick test
        # training_percents = [0.1]
        # n_shuffle = 2

        f1 = direct + 'LR_' + dataset + '_classi_results_sdne.csv'
        f2 = direct + 'EigenPro_' + dataset + '_classi_results_sdne.csv'

        if (not os.path.exists(f2)) or reeval_flag:
            print('%s does not exist, start evaluating...\n'%f2)
            logger.info(classification.classify(emb=nc_emb, network=mat_f, writetofile=True, classifier = 'EigenPro', test_kernel="eigenpro",
                                            dataset=dataset, algorithm=method, word2vec_format=False,
                                            num_shuffles=n_shuffle, output_dir=direct,
                                            training_percents=training_percents))
        else:
            print('%s already exist\n' % f2)

        if (not os.path.exists(f1)) or reeval_flag:
            print('%s does not exist, start evaluating...\n'%f1)
            logger.info(classification.classify(emb=nc_emb, network=mat_f, writetofile=True, classifier = 'LR',
                                            dataset=dataset, algorithm=method, word2vec_format=False,
                                            num_shuffles=n_shuffle, output_dir=direct,
                                            training_percents=training_percents))
        else:
            print('%s already exist\n' % f1)
            if reeval_flag:
                f1_mask = direct + 'Mask_LR_' + dataset + '_classi_results_sdne.csv'
                os.rename(f1, f1_mask)
                print('Rename the %s to %s'%(f1, f1_mask))


#

