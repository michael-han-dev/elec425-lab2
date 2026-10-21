import scipy.io
from k_means import k_means

# Load data
data = scipy.io.loadmat("data.mat")["data"]

# Perform K-means clustering
# Supports visualization of up to 4 clusters
k_means(data, 3)