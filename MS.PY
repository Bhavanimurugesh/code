# Importing necessary library
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Loading the dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Displaying the column names
print(mcdonalds.columns)

# Displaying the dimensions of the dataset
print(mcdonalds.shape)

# Displaying the first 3 rows of the dataset
print(mcdonalds.head(3))

# Selecting the columns for PCA
MD_x = mcdonalds.iloc[:, 0:11]

# Converting "Yes" and "No" to 1 and 0
label_encoder = LabelEncoder()
MD_x = MD_x.apply(lambda col: label_encoder.fit_transform(col))

# Calculating the column means
print(round(MD_x.mean(), 2))

# Performing PCA
MD_pca = PCA()
MD_pca.fit(MD_x)

# Displaying the summary of PCA
print("Standard deviations (1, .., p=11):")
print(MD_pca.explained_variance_)
print("Proportion of Variance:")
print(MD_pca.explained_variance_ratio_)
print("Cumulative Proportion:")
print(MD_pca.explained_variance_ratio_.cumsum())
import numpy as np
from sklearn.cluster import KMeans

# Data
data = {
    'yummy': [0.477, -0.36, 0.30, -0.055, -0.308, 0.17, -0.28, 0.01, -0.572, 0.110, 0.045],
    'convenient': [0.155, -0.02, 0.06, 0.142, 0.278, -0.35, -0.06, -0.11, 0.018, 0.666, -0.542],
    'spicy': [0.006, -0.02, 0.04, -0.198, 0.071, -0.36, 0.71, 0.38, -0.400, 0.076, 0.142],
    'fattening': [-0.116, 0.03, 0.32, 0.354, -0.073, -0.41, -0.39, 0.59, 0.161, 0.005, 0.251],
    'greasy': [-0.304, 0.06, 0.80, -0.254, 0.361, 0.21, 0.04, -0.14, 0.003, -0.009, 0.002],
    'fast': [0.108, 0.09, 0.06, 0.097, 0.108, -0.59, -0.09, -0.63, -0.166, -0.240, 0.339],
    'cheap': [0.337, 0.61, 0.15, -0.119, -0.129, -0.10, -0.04, 0.14, -0.076, -0.428, -0.489],
    'tasty': [0.472, -0.31, 0.29, 0.003, -0.211, -0.08, 0.36, -0.07, 0.639, -0.079, 0.020],
    'expensive': [-0.329, -0.60, -0.02, -0.068, -0.003, -0.26, -0.07, 0.03, -0.067, -0.454, -0.490],
    'healthy': [0.214, -0.08, -0.19, -0.763, 0.288, -0.18, -0.35, 0.18, 0.186, 0.038, 0.158],
    'disgusting': [-0.375, 0.14, 0.09, -0.370, -0.729, -0.21, -0.03, -0.17, 0.072, 0.290, -0.041]
}

# Convert data to numpy array
X = np.array([data[key] for key in data])

# Set seed
np.random.seed(1234)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=8, n_init=10, random_state=0)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_
print(labels)
import numpy as np
import pandas as pd
from flexclust import bootFlexclust, slswFlexclust
from flexmix import stepFlexmix, getModel, flexmix
import matplotlib.pyplot as plt

np.random.seed(1234)

# Assuming MD.x is a pandas DataFrame
# Convert MD.x to numpy array if it's not already in that format
MD_x = MD.x.to_numpy() if isinstance(MD.x, pd.DataFrame) else MD.x

# Bootstrapping with flexclust
MD_b28 = bootFlexclust(MD_x, np.arange(2, 9), nrep=10, nboot=100)

# Plotting bootstrapped results
plt.plot(MD_b28)
plt.xlabel("number of segments")
plt.ylabel("adjusted Rand index")
plt.show()

# Histogram of kmeans clustering with 4 clusters
plt.hist(MD_km28["4"], data=MD_x, xlim=(0, 1))
plt.show()

# Using k=4 for flexclust
MD_k4 = MD_km28["4"]
MD_r4 = slswFlexclust(MD_x, MD_k4)

# Plotting flexclust results
plt.plot(MD_r4)
plt.ylim(0, 1)
plt.xlabel("segment number")
plt.ylabel("segment stability")
plt.show()

# Using flexmix for model-based clustering
MD_m28 = stepFlexmix(MD_x, model="FLXMCmvbinary", k=np.arange(2, 9), nrep=10, verbose=False)
print(MD_m28)

# Plotting information criteria for flexmix
plt.plot(MD_m28)
plt.ylabel("value of information criteria (AIC, BIC, ICL)")
plt.show()

# Getting model for k=4 from flexmix
MD_m4 = getModel(MD_m28, which="4")

# Comparing kmeans and mixture clusters
table = pd.crosstab(clusters(MD_k4), clusters(MD_m4))
print(table)

# Creating flexmix model using kmeans clusters
MD_m4a = flexmix(MD_x, cluster=clusters(MD_k4), model="FLXMCmvbinary")
table = pd.crosstab(clusters(MD_k4), clusters(MD_m4a))
print(table)

# Calculating log likelihood for flexmix models
loglik_m4a = MD_m4a.logLik()
loglik_m4 = MD_m4.logLik()
print(f"log Lik. MD_m4a: {loglik_m4a} (df={MD_m4a.df})")
print(f"log Lik. MD_m4: {loglik_m4} (df={MD_m4.df})")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# plot MD.ref2
plt.plot(MD.ref2)
plt.show()

# calculate hierarchical clustering
MD.vclust = linkage(pdist(MD.ref2))
# plot dendrogram
dendrogram(MD.vclust)
plt.show()

# plot MD.k4 with shading and specified order
plt.barchart(MD.k4, shade=True, which=np.flip(MD.vclust['order']))
plt.show()

# plot MD.k4 with specified principal components and options
plt.plot(MD.k4, project=MD.pca, data=MD.x, hull=False, simlines=False)
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.show()

# calculate projection axes for MD.pca
projAxes(MD.pca)

import pandas as pd
import matplotlib.pyplot as plt
from partykit import ctree

k4 = clusters(MD.k4)

# Mosaic plot for Like and segment number
like_segment_table = pd.crosstab(k4, mcdonalds['Like'])
plt.mosaic(like_segment_table, shade=True)
plt.xlabel('segment number')
plt.show()

# Mosaic plot for Gender and segment number
gender_segment_table = pd.crosstab(k4, mcdonalds['Gender'])
plt.mosaic(gender_segment_table, shade=True)
plt.show()

# Creating ctree
tree = ctree(
    (k4 == 3).astype('category') ~ mcdonalds['Like.n'] + mcdonalds['Age'] + mcdonalds['VisitFrequency'] + mcdonalds['Gender'],
    data=mcdonalds
)

# Plotting the tree
tree.plot()

# Calculating mean visit frequency for each segment
visit = mcdonalds.groupby(k4)['VisitFrequency'].mean()
visit

import numpy as np
import matplotlib.pyplot as plt

like = np.mean(mcdonalds['Like.n'].groupby(k4))
print(like)

female = np.mean((mcdonalds['Gender'] == "Female").astype(int).groupby(k4))
print(female)

plt.scatter(visit, like, s=10 * female, xlim=(2, 4.5), ylim=(-3, 3))
for i, txt in enumerate(range(1, 5)):
    plt.text(visit[i], like[i], txt)
plt.show()


