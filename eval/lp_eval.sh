#!/usr/bin/env bash

for dataset in 'texas' 'blogcatalog' 'wisconsin' 'washington' 'cornell' 'texas' 'wikivote' 'wikipedia' 'co-author' 'ppi' 'pubmed'  'p2p-gnutella31' 'microsoft' #'flickr'
do
for d in 64 128 256
do
for alpha in 1e-05 0.2
do
for beta in 5 10
do
if [ "$dataset" = "flickr" ] || [ "$dataset" = "p2p-gnutella31" ] || [ "$dataset" = "microsoft" ];then
   n_iter=5
   n_batch=500
elif [ "$dataset" = "texas" ] || [ "$dataset" = "cornell" ] || [ "$dataset" = "washington" ] || [ "$dataset" = "wisconsin" ];then
   n_iter=20
   n_batch=50
elif [ "$dataset" = "wikipedia" ];then
   n_iter=50
   n_batch=500
else
    n_iter=20
    n_batch=500
fi

if [ ! -f '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/'$dataset'/sdne/newloss_[False]/alpha_'$alpha'/beta_'$beta'/n_iter_['$n_iter']/n_batch_'$n_batch'/d_'$d'/n_units_[500]/LPresults.csv' ]; then
    echo "Evaluating LP /home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/'$dataset'/sdne/newloss_[False]/alpha_'$alpha'/beta_'$beta'/n_iter_['$n_iter']/n_batch_'$n_batch'/d_'$d'/n_units_[500]"
~/anaconda2/bin/python /home/cai.507/Documents/DeepLearning/EmbeddingEval/link_prediction/NRL/link_prediction_main_fn.py \
--input_graph_dir '/home/cai.507/Documents/DeepLearning/EmbeddingEval/data/nrl-data/link_prediction/'$dataset'_80_20' \
--file_name 'fold' \
--dataset $dataset\
--write \
--output_file_name '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/'$dataset'/sdne/newloss_[False]/alpha_'$alpha'/beta_'$beta'/n_iter_['$n_iter']/n_batch_'$n_batch'/d_'$d'/n_units_[500]/LPresults.csv' \
--input_embedding0_dir '/home/cai.507/Documents/DeepLearning/EmbeddingEval/experiments/'$dataset'/sdne/newloss_[False]/alpha_'$alpha'/beta_'$beta'/n_iter_['$n_iter']/n_batch_'$n_batch'/d_'$d'/n_units_[500]' \
--embedding0_file_name 'fold' \
--share_embeddings \
--num_folds 5 \
--emb_size $d \
--algorithm 'sdne' \
--embedding_params '{}'
fi
done
done
done
done

#python link_prediction_main_fn.py --input_graph_dir pubmed_80_20
#--file_name fold
#--dataset pubmed
#--writetofile
#--output_file_name results.csv
#--input_embedding0_dir pubmed_80_20_embedding_u
#--embedding0_file_name pubmed_embedding \
#--share_embeddings --num_folds 5
#--emb_size 128
#--algorithm deepwalk
#--embedding_params '{"num_walks" :40, "walk_length" : 80,"window" : 10}'
