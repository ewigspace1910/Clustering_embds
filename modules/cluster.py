from sklearn.cluster import DBSCAN, KMeans

names =  ['kmeans', 'dbscan']

def get_clustor(name='dbscan', num_clusters=500, seed=1, eps=0.6, min_samples=3):
    if name in names:
        if name == 'dbscan':
            return DBSCAN(eps=eps, min_samples=min_samples) #eps need modify empirically
        elif name == 'kmeans':
            return KMeans(n_clusters=num_clusters, random_state=seed,max_iter=300)
    else:
        assert False, "incorrect name of algorithms \n Available clustering algorithms: {}".format(names)