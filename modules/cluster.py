from sklearn.cluster import DBSCAN, Kmeans

names =  ['kmean', 'dbscan']

def get_clustor(name='dbscan', num_clusters=500, seed=1, n_jobs=2):
    if name in names:
        if name == 'dbscan':
            return DBSCAN(eps=0.3, min_samples=4)
        elif name == 'kmean':
            return Kmeans(n_clusters=num_clusters, random_state=seed, n_jobs=n_jobs,max_iter=400)
    else:
        print('Available clustering algorithms:', names)
        assert False, "incorrect name of algorithms"