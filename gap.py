from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from python.COPDGene.utils.sample_wr import sample_wr

# import iris dataset
iris = datasets.load_iris()
data_raw = iris.data
labels_true = iris.target

# Normalization of the original dataset
data = scale(data_raw)

# extract reference distribution
data_ref = []
tp_row_id = sample_wr(range(data.shape[0]),data.shape[0])
for i in range(len(tp_row_id)):
    data_ref.append(list(data[tp_row_id[i],:]))
data_ref = np.array(data_ref)

n_clusters_range = range(2,11)
inertia = [0]*len(n_clusters_range)
inertia_ref = [0]*len(n_clusters_range)
score = [0]*len(n_clusters_range)

for i in range(len(n_clusters_range)):
    # Apply Kmeans on the original dataset
    estimator = KMeans(n_clusters=n_clusters_range[i],init='random',\
            n_init=10,n_jobs=-1)
    estimator.fit(data)
    inertia[i] = estimator.inertia_
    
    # Apply Kmeans on the reference dataset
    estimator_1 = KMeans(n_clusters=n_clusters_range[i],init='random',\
            n_init=10,n_jobs=-1)
    estimator_1.fit(data_ref)
    inertia_ref[i] = estimator_1.inertia_
    
    score[i] = inertia_ref[i]-inertia[i]

plt.plot(n_clusters_range,inertia,'b')
plt.plot(n_clusters_range,inertia_ref,'r')
plt.plot(n_clusters_range,score,'g')
plt.show()
