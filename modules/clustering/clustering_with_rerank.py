#edit for private own dataset
from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import torch

#from .evaluation_metrics import cmc, mean_ap
from ..utils.meters import AverageMeter
from ..utils.rerank import re_ranking

from collections import defaultdict
import threading
import multiprocessing

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

      
def pairwise_distance(features, query=None, gallery=None):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def clustering_all(distmat_gg, gallery=None, top_k = 15, label_clusters=None, hardmore=False, minimum_sample=2):
    if gallery is not None:
        gallery_path = [pid for pid,_, _ in gallery]
    else:
        print("error 404")
        exit()
    
    clusters = {i:[] for i in set(label_clusters)}
    for index, cluster in enumerate(label_clusters):
        clusters[cluster].append(index)
    
    new_clusters = {} #shape path
    indices = np.argsort(distmat_gg, axis=1)
    
    for cluster in clusters:
        new_clusters[cluster] = []
            
        #match = np.zeros((len(clusters[cluster]), len(clusters[cluster])))
        tmp_index = [i for i in clusters[cluster]]
        for i in tmp_index:
            tmp_cp =  tmp_index.copy()
            tmp_cp.remove(i) 
            distvec_i = indices[i]  #rank k-nearest of i
            #with j<>i must in top k of i => j (= cluster(i)
            sorter_dismat_i = np.argsort(distvec_i) 
            rank_tmp_in_dv_i = sorter_dismat_i[np.searchsorted(distvec_i, tmp_cp, sorter=sorter_dismat_i)] #rank other k-point to i-point
            rank_tmp_in_dv_i = rank_tmp_in_dv_i < top_k
            
            num_j_accept_i = len(tmp_cp)
            if hardmore:
                print("using hard sample")
                #i must in top k/2 of j => i (= cluster(j)  ~  significantly same reranking
                for j in tmp_cp:
                    distvec_j = indices[j]
                    if np.where(distvec_j == i)[0][0] > top_k // 2: num_j_accept_i-= 1             
                       
            #print(rank_tmp_in_dv_i)
            constrain_1 = len(tmp_cp) // 5 * 4   #empritical belief :>>
            constrain_2 = len(tmp_cp) // 3 * 2       #empritical belief :>>
            if np.sum(rank_tmp_in_dv_i.numpy()) > constrain_1  and num_j_accept_i  > constrain_2: 
                new_clusters[cluster].append(i)         
            
    clusters = {}
    for label in new_clusters:
        for index in new_clusters[cluster]:
            clusters[gallery_path[index]] = [label]
    return cluster


class DSCluster(object):
    def __init__(self, args):
        super(DSCluster, self).__init__()
        self.args = args

    def evaluate(self, features, gallery, rerank=False):
        #distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
        distmat_gg, _, gallery_features = pairwise_distance(features, gallery, gallery)

        print("run kmeans")
        cf = normalize(gallery_features, axis=1)
        km = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed,max_iter=400).fit(cf)
        target_label = km.labels_              
        
        #calculate dismat
        results = clustering_all(distmat_gg, gallery=gallery, label_clusters=target_label, hardmore=self.args.hard_sample, minimum_sample=4)
        
        if rerank:
            print('Applying person re-ranking ...')
            distmat = re_ranking(distmat_gg.numpy(), distmat_gg.numpy(), distmat_gg.numpy())
            results = clustering_all(distmat, gallery=gallery, label_clusters=target_label, minimum_sample=4)   
          
        return results