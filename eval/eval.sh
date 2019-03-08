#!/usr/bin/env bash

for  graph in 'blogcatalog' 'wisconsin' 'washington' 'cornell' 'texas' 'wikivote' 'wikipedia' 'co-author' 'ppi' 'pubmed'  'p2p-gnutella31' 'microsoft' 'flickr'
do
    time ~/anaconda2/bin/python -W ignore ./eval/nc_eval_all.py --dataset $graph
done
exit


mv emb_fold0_lp.npy fold_0.npy
mv emb_fold1_lp.npy fold_1.npy
mv emb_fold2_lp.npy fold_2.npy
mv emb_fold3_lp.npy fold_3.npy
mv emb_fold4_lp.npy fold_4.npy
