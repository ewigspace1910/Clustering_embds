import argparse
import glob
import os.path as osp
import random
import numpy as np
import torch
from .modules.cluster import get_clustor




def clustering(features_dict, keys, args):
    x = torch.cat([features_dict[f].unsqueeze(0) for f in keys], 0)
    clustor = get_clustor(name=args.algorithm, num_clusters=args.clusters, seed=args.seed, n_jobs=args.workers)
    db = clustor.fit(x)
    labels = db.labels_
    print(labels)

    labels_dict = {k:[labels[i]] for i, k in enumerate(keys)}
    del clustor, db, x
    return labels_dict


__src__ = []
__ensemble_dict = {}
def run(args):
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
                if k not in labels_dict.leys(): __ensemble_dict[k] += [-9] #random number, do not be confused
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
    final_label_dict = clustering(features_dict=__ensemble_dict, keys=gallery, args=args)
    
    #write2file
    with open("final_label_dict.o", 'w') as f:
        for k in final_label_dict: f.write("{}:{}\n".format(k, final_label_dict[k]))

    


def main():
    args = parser.parse_args()
    if args.algorithm == 'kmeans' and args.clusters == 0:
        assert False, "if using kmeans, set --clusters > 0"

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    global __src__
    __src__ = glob.glob(osp.join(args.features_dict, '*.pth'))
    assert len(__src__) > 0, "!!!Can't load *.pth,\n !!!Please fix feature_path or fill all feature.pth path into __src__" 
    run(args)
    #loc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clustering features")
    # data
    parser.add_argument('-a', '--algorithm', type=str, default='dbscan')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--minimum-sample', type=int, default=4, help="min sample in class")
    # cluster
    parser.add_argument('--flag-mulcluster', action='store_true', help="Using multicluster or not")
    # testing configs
    parser.add_argument('--clusters', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--feature-dir', type=str, metavar='PATH',default=osp.join(working_dir, 'store', 'features'))    
    main()