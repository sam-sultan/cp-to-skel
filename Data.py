
import numpy as np
#import pcl
#import open3d as o3d
from scipy.cluster.vq import kmeans
#import pandas as pd
#import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


class Data:

    def __init__(self, file):
        self.Points = np.loadtxt(file) if isinstance(file, str) else file
        self.Points = np.unique(self.Points, axis=0) # remove duplicates


    """
    Center the points around a point or the mean
    if center is none then they are centered around the mean
    """
    def center(self, center=None):

        if center == None:

            xm, ym, zm = self.means()
            self.Points[:, 0] -= xm
            self.Points[:, 1] -= ym
            self.Points[:, 2] -= zm

        else:

            xmean, ymean, zmean = center
            xm, ym, zm = self.means()
            self.Points[:, 0] += (xmean - xm)
            self.Points[:, 1] += (ymean - ym)
            self.Points[:, 2] += (zmean - zm)

        return self.Points

    """
    Get the mean of the Points
    """
    def means(self):

        return np.mean(self.Points[:,0]), np.mean(self.Points[:,1]), np.mean(self.Points[:,2])


    """ Computes the co-variance matrix of the data. """
    def compute_cov_3d(self, Points):

        x = Points[:,0]
        y = Points[:,1]
        z = Points[:,2]

        # Compute the diagonal of covariance
        xx = np.sum( x * x, axis=0)
        yy = np.sum( y * y, axis=0)
        zz = np.sum( z * z, axis=0)

        # Compute the off-diagonal of covariance
        xy = np.sum( x * y, axis=0)
        xz = np.sum( x * z, axis=0)
        yz = np.sum( y * z, axis=0)

        return np.asarray( [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    """
        1. Compute the co-variance of the data matrix after mean adjustment, and then I computed its eigenvector.
        2. Multiply the eigenvector by the mean-adjusted data matrix to get the new data matrix.
    """
    def rotate(self, rotation_matrix=None):

        if rotation_matrix is None:

            # Step 1.
            Cov = self.compute_cov_3d(self.Points)
            EigVals, EigVecs = np.linalg.eig(Cov)

            # Step 2.
            self.Points = np.dot(self.Points,EigVecs)

        else:

            # Step 2.
            self.Points = np.dot(self.Points, rotation_matrix)

        return self.Points

    def export(self, output):

        np.savetxt(output, self.Points, delimiter="\t")
