from sklearn.cluster import DBSCAN, KMeans

names =  ['kmean', 'dbscan']

def get_clustor(name='dbscan', num_clusters=500, seed=1):
    if name in names:
        if name == 'dbscan':
            return DBSCAN(eps=0.3, min_samples=4)
        elif name == 'kmeans':
            return KMeans(n_clusters=num_clusters, random_state=seed,max_iter=400)
    else:
        print('Available clustering algorithms:', names)
        assert False, "incorrect name of algorithms"