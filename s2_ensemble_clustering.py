import argparse
import glob
import os
import os.path as osp
import random
import numpy as np
import torch
from modules.cluster import get_clustor
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from modules.clustering.clustering_with_rerank import DSCluster



def clustering(features_dict, keys, args, final=False):
    x = torch.cat([features_dict[f].unsqueeze(0) for f in keys], 0)
    if not final and args.n_pca > 0:
        x = PCA(n_components=args.n_pca).fit_transform(x)
        clustor = get_clustor(name=args.algorithm, seed=args.seed, 
                num_clusters=args.clusters, #kmeans
                eps=args.dbs_eps, min_samples=args.dbs_min) #dbscan
    else:  
        x = normalize(x, axis=1)
        clustor = get_clustor(name=args.algorithm, seed=args.seed, 
                num_clusters=args.clusters, #kmeans
                eps=args.dbs_eps / 2, min_samples=args.dbs_min) #dbscan


    db = clustor.fit(x)
    labels = db.labels_

    labels_dict = {k:[labels[i]] for i, k in enumerate(keys)}
    del clustor, db, x
    return labels_dict


__src__ = []

def run_simple(args):
    global __src__
    print("!load -->{}".format(__src__[0]))
    feature_dict = torch.load(__src__[0])
    #prepare data
    gallery = [k for k in feature_dict.keys()]

    #Early clustering
    final_label_dict = clustering(feature_dict, gallery, args)
    
    return final_label_dict

__ensemble_dict = {}
def run_ensemble(args):
    global __ensemble_dict
    global __src__
    __flag_init__ = True
    #-------------------------
    #Early clustering stage
    len_pre_dict = 0
    for p in __src__:
        feature_dict = torch.load(p)
        #prepare data
        gallery = [k for k in feature_dict.keys()]
        if len_pre_dict > 0 and len_pre_dict != len(gallery): assert False, "file {} has len not equal to others".format(p)
        else: len_pre_dict = len(gallery)

        #Early clustering
        labels_dict = clustering(feature_dict, gallery, args)
        #Assemble labels
        if __flag_init__:
            __ensemble_dict = labels_dict
            __flag_init__ = False
        else:
            for k in __ensemble_dict.keys():
                if k not in labels_dict.keys(): __ensemble_dict[k] += [-9] #random number, do not be confused
                else: __ensemble_dict[k] += labels_dict[k]
        
        del feature_dict
    #-------------------------
    #Finetune Stage
    ###tranform to generate new feature vector
    gallery = []
    for k in __ensemble_dict.keys():
        gallery += [k]
        __ensemble_dict[k] = torch.Tensor(__ensemble_dict[k]).float()
    ###Re-Clustering
    final_label_dict = clustering(features_dict=__ensemble_dict, keys=gallery, args=args, final=True)
    return final_label_dict

    
def run_restrict_ranking(args):
    global __src__
    print("!load -->{}".format(__src__[0]))
    feature_dict = torch.load(__src__[0])
    gallery = [k for k in feature_dict.keys()]
    clustor = DSCluster(args)
    final_label_dict = clustor.evaluate(feature_dict, gallery=gallery, rerank=args.rerank)
    return final_label_dict



def main():
    args = parser.parse_args()
    if args.algorithm == 'kmeans' and args.clusters == 0:
        assert False, "if using kmeans, set --clusters > 0"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    global __src__
    __src__ = glob.glob(osp.join(args.feature_dir, '*.pth'))
    assert len(__src__) > 0, "!!!Can't load *.pth,\n !!!Please fix feature_path or fill all feature.pth path into __src__" 
    
    if args.flag_ensemble:
        print("run ensemble clustering mode")
        final_label_dict = run_ensemble(args)
    elif args.flag_single_rerank:
        print("run single clustering with rerank mode")
        final_label_dict = run_restrict_ranking(args)
    else:
        print("run simple clustering")
        final_label_dict = run_simple(args)
    #save
    if not osp.exists(args.store_dir): os.makedirs(args.store_dir)
    torch.save(final_label_dict, osp.join(args.store_dir, "ensemble_labels.pth"))
    print("!saved ensemble labels in {}".format(osp.join(args.store_dir, "ensemble_labels.pth")))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering features")
    # data
    parser.add_argument('-a', '--algorithm', type=str, default='kmeans')
    parser.add_argument('--minimum-sample', type=int, default=4, help="min sample in class")
    ###############
    # ensemble clustering
    parser.add_argument('--flag-ensemble', action='store_true', help="Using multicluster or not")
    parser.add_argument('--seed', type=int, default=1)
    ###Kmeans
    parser.add_argument('--clusters', type=int, default=0)
    ###DBsan
    parser.add_argument('--dbs-eps', type=float, default=0.6, help="eps hyperparameter for dbscan")
    parser.add_argument('--dbs-min', type=float, default=4, help="min sample hyperparameter for dbscan")

    ########
    #reranking clustering
    parser.add_argument('--flag-single-rerank', action='store_true', help="Using only one feature file and rerank clustering")
    parser.add_argument('--rerank', action='store_true', help="Using orginal rerank not")
    parser.add_argument('--hard-sample', action='store_true', help="Select clusters containing number of samples in specified range")

    ########
    #PCA
    parser.add_argument('--n-pca', type=int, default=0, help="number components of PCA, if greater than 0, pca is used")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--feature-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'store', 'features'))
    parser.add_argument('--store-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'store', 'ensembles'))    
    main()