import sys
import os
sys.path.append('/home/cai.507/Documents/DeepLearning/EmbeddingEval/link_prediction/NRL/')
from link_prediction_main_fn_new import link_prediction
sys.argv = ['/home/cai.507/.pycharm_helpers/pydev/pydevconsole.py']

input_graph_dir = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/data/nrl-data/link_prediction/wikipedia_80_20/'
file_name = 'fold'
dataset = 'wikipedia'
writetofile_ = True
input_embedding0_dir = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/wikipedia/sdne/n_iter_[50]/n_batch_500/d_100/n_units_[500, 128]/'
output_file_name = os.path.join(input_embedding0_dir, 'LPresults.mcsv')
# embedding0_file_name = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/wikipedia/sdne/n_iter_[50]/n_batch_500/d_100/n_units_[500, 128]/emb_fold_lp'
embedding0_file_name = 'emb_fold_lp'
num_folds = 5
emb_size = 100
algorithm = 'sdne'
input_embedding1_dir = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/wikipedia/sdne/n_iter_[50]/n_batch_500/d_100/n_units_[500, 128]/'
embedding1_file_name = 'embedding1'
share_embeddings = True
word2vec_format = False
l2_normalize = True
no_gridsearch = False
edge_features = ['hadamard', 'l2', 'concat']
embedding_params = {}

link_prediction(input_graph_dir,
                file_name,
                dataset,
                writetofile_,
                output_file_name,
                input_embedding0_dir,
                embedding0_file_name,
                input_embedding1_dir,
                embedding1_file_name,
	            share_embeddings,
                word2vec_format,
                l2_normalize,
                num_folds,
                emb_size,
                algorithm,
                no_gridsearch,
                embedding_params,
                edge_features)


# "python link_prediction_main_fn.py " \
# "--input_graph_dir ppi_80_20 " \
# "--file_name fold " \
# "--dataset ppi " \
# "--writetofile " \
# "--output_file_name results.csv " \
# "--input_embedding0_dir ppi_80_20_embedding_u --" \
# "embedding0_file_name ppi_embedding \
# --input_embedding1_dir ppi_80_20_v 	" \
# "--embedding1_file_name " \
# "ppi_embedding --num_folds 5 " \
# "--emb_size 128 " \
# "--algorithm deepwalk " \
# "--embedding_params {'num_walks' :40, 'walk_length' : 80, 'window' : 10}"
#

import numpy as np
# f = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/wikipedia/sdne/n_iter_[50]/n_batch_500/d_100/n_units_[500]/emb_fold2_lp.npy'



# f = '/home/cai.507/Documents/DeepLearning/EmbeddingEval/data/nrl-data/link_prediction/' + 'ppi' + '_80_20/' + 'fold_1.mat'
# import scipy.io as sio
# mat = sio.loadmat(f)

