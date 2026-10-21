import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import cdist

def k_means(X, n_cluster):
    # X: data matrix (rows are data points, columns are features)
    # n_cluster: number of clusters
    
    if n_cluster > 4:
        print('You have set too many clusters.')
        print('Set the number of clusters to be 1-4.')
        print('The program and visualization allow for up to 4 clusters.')
        return None, None

    # Initialize the figure
    plt.figure(figsize=(8, 6))

    # Get the number of data points and features
    n_sample, n_feat = X.shape

    # Randomly initialize the starting cluster centers
    np.random.seed()

    up_bound = np.max(X, axis=0)
    lw_bound = np.min(X, axis=0)
    centres = lw_bound + (up_bound - lw_bound) * np.random.rand(n_cluster, n_feat)

    print('Start K-means clustering ... ')

    # Initialization:
    #In the begining, all data points are in cluster 1
    # The "old_membership" variable is an n_sample-by-1 matrix.
    # It saves the cluster id that each data point belongs to.
    old_membership = np.ones(n_sample, dtype=int)  # All points start in cluster 1

    # Display the initial cluster membership and centers
    show(X, old_membership, n_cluster, centres, 'Cluster centres initialized!')

    while True:
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % You need to add code here.
        % Calculate squared Euclidean distances
        % between every data point and every cluster centre.
        % Please put your results in an 
        % n_sample-by-n_cluster matrix named as "distance". 
        % You may find the Matlab function pdist2 to be useful here. 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        distance = cdist(X, centres, 'sqeuclidean')
        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % You need to add code here.
        % E step: Assign data points to closest clusters.
        % Specifically, for each data point, find closest 
        % cluster centre, and assign the data point
        % to that cluster.
        % Save your assignment to the variable "membership", 
        % which is an n_sample-by-1 vector and each row saves 
        % the cluster membership of a datapoint.
        % (See the description of "old_membership" as well.) 
        % You should use here the "distance" computed above.
        % You may find the function "min" useful here,
        % but feel free not to use it and write whatever code
        % you think can implement the above requirement.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        membership = np.argmin(distance, axis=1)

        # Show the result of the E step
        show(X, membership, n_cluster, centres, 'E step finished: Datapoints re-assigned!')

        '''
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % You need to add code here.
        % M step: Update cluster centres based on the new assignment.
        % You need to use "membership" here.
        % Save your new centres in variable "centres".
        % You may find the function "mean" to be useful here.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        '''
        for k in range(n_cluster):
            if np.any(membership == k):
                centres[k] = np.mean(X[membership == k], axis=0)

        # Show the result of the M step
        show(X, membership, n_cluster, centres, 'M step finished: Cluster centers updated!')

        # Stop if no more updates (i.e., membership hasn't changed)
        if np.all(membership == old_membership):
            show(X, membership, n_cluster, centres, 'Done!')
            plt.show()
            break

        old_membership = membership


def show(X, c_pred, n_cluster, centres, txt):
    symbol = ['ro', 'gp', 'bd', 'k^', 'r*']
    plt.clf()

    for i in range(n_cluster):
        marker = i % 5
        if i >= 4:
            print('Total number of clusters exceeds 4, some symbols in the plot are reused!')

        # Plot the data points for each cluster
        plt.plot(X[c_pred == i, 0], X[c_pred == i, 1], symbol[marker])
        
        # Plot the cluster centers
        plt.plot(centres[i, 0], centres[i, 1], symbol[marker][1], markerfacecolor=symbol[marker][0][0])

    plt.text(4.2, 5.4, txt, fontsize=12)
    plt.draw()

    # Pause for a bit to allow the plot to be displayed
    plt.pause(0.5)

    

if __name__=="__main__":
    # Load the data
    data = loadmat("data.mat")['data']

    
    #Perform K-means clustering.
    #The function provides visualization  to support up to 4 clusters.
    k_means(data, 3)








