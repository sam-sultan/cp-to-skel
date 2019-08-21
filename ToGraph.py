import numpy as np
from sklearn.neighbors import KDTree
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, euclidean
from Data import Data


"""

Converts Cloud points to Graph

"""

class ToGraph:

    def __init__(self, Data):
        self.Data = Data

    """
    Computes pairwise distances
    and returns the median
    """
    def medianPD(self, Points):
        """
        clusters = []
        for i in range(1, 50):
            centroids, distortion = kmeans(self.Points, i)
            clusters.append((i, distortion))

        df = pd.DataFrame(clusters, columns=["k", "distortion"])
        df.plot(kind="line", x="k", y="distortion")
        plt.show()
        """
        print("Computing the pdist...")
        dists = pdist(Points, metric="euclidean")
        print("Computing the median...")
        #df = pd.DataFrame(dists)
        #df.hist()
        #plt.show()
        median = np.median(dists)
        print("Done!")

        return median

    """
    Converts the cloud points to graph
    """
    def convert(self, distance, total=100000):

        # Points and get rid of duplicates
        Points = self.Data.Points

        # Kd tree
        self.tree = KDTree(Points, leaf_size=2, metric='euclidean')

        distance = distance #self.medianPD(Points)
        #print(self.medianPD(Points))
        #res = self.tree.kernel_density(Points, h=0.5)
        #print(res/sum(res))

        #print("Querying the nearest neighbors...")
        # get nearest points
        pts, distances = self.tree.query_radius( Points , r=distance, return_distance=True, sort_results=True)
        #print("Done!")

        # graph
        self.G = nx.Graph()

        #print("Building the Graph...")
        # loop through the closest points to each and build the graph
        for ptstack, diststack in zip(pts, distances):

            # make sure to grab the maximum allowed total neighbors
            t = ptstack.shape[0] if total > ptstack.shape[0] else total

            #print(ptstack, diststack)

            # add the current node
            self.G.add_node(ptstack[0])

            # loop through the nearest neighbors
            for pt_idx, dist in zip( ptstack[1:t], diststack[1:t] ):
                # add the node to connect to
                self.G.add_node(pt_idx)
                # add the edge with the distance as weight
                self.G.add_edge(ptstack[0], pt_idx, weight=dist)

        # graph
        #nx.draw(self.G, with_labels=True, font_weight='bold')
        #plt.show()
        #print("Done!")

        return self.G

    """
    Convert graph to a directed graph
    # dimension: 0 -> x
                 1 -> y
                 2 -> z
    # dir: False -> starting from +inf -> -inf
           True  -> starting from -inf -> +inf
    """
    def convertToDirectedG(self, dim, dir=False):

        # make sure the dimensions are with in limit
        assert 0 <= dim <= 2, "Please make sure the dim is between 0 and 2"

        # get ref to points
        Points = self.Data.Points

        # params
        self.dim = dim
        self.dir = dir

        # init the directed graph
        G = nx.DiGraph()

        # if positive
        if not dir:
            rootidx = np.argmax(Points, axis=0)
            rootidx = rootidx[dim]
        else:
            rootidx = np.argmin(Points, axis=0)
            rootidx = rootidx[dim]


        # compute shortest between this root node and the rest
        for id in self.G.nodes:
            if id != rootidx:
                try:
                    path = nx.astar_path(self.G, rootidx, id)
                    #print(rootidx, id, path)
                    cum_dist = 0.
                    for i in range(len(path)-1):
                        dist = self.G.edges[path[i+1], path[i]]['weight']
                        G.add_weighted_edges_from([(path[i + 1], path[i], dist )])
                        G.nodes[path[i]]['dist'] = cum_dist
                        #print((path[i], G.nodes[path[i]]['dist'] ))
                        cum_dist += dist

                    # set the cumulutative dist for the current node
                    G.nodes[id]['dist'] = cum_dist

                    #print("\n")

                except:
                    print("No path", rootidx, id)

        self.G = G
        """
        #print(Points[rootidx,:])
        #print([ (id, G.nodes[id]) for id in G.nodes ])
        """


        pos = {}
        labels = []
        for id in G.nodes:
            pos[id] = (Points[id][0], Points[id][1])
            if rootidx == id:
                labels.append("r")
            else:
                labels.append("y")

        # graph
        nx.draw(G, pos, node_color=labels, with_labels=True, font_weight='bold')
        plt.show()


    """
    Level set
    steps: how many pieces the architecture should be divided to
    """
    def levelSet(self, steps):
        pts = self.sortNodes()
        Points = self.Data.Points

        centroids = []
        longest = max(pts, key=lambda x:x[1])[1]
        step = longest/float(steps)
        s = 0.
        c = 0
        print(longest, step)
        distance = 0.
        while s <= longest:

            cp = []
            for i in range(c, len(pts)):
                pt = pts[i]
                if pt[1] > s:
                    break
                cp.append(pt[0])
                c+=1

            centroid = np.mean(Points[cp], axis=0)
            centroids.append(centroid)
            s += step

        print(centroids)
        # convert these generated figures
        data = Data(centroids)
        g = ToGraph(data)
        g.convert(step)
        g.convertToDirectedG(self.dim, self.dir)



    """
    Sort the nodes by distance to root
    """
    def sortNodes(self):

        """
        distances = []
        for id in self.G.nodes:
            distances.append(self.G.nodes[id]['dist'])
        print(distances)
        """

        data = [ (id, self.G.nodes[id]['dist']) for id in self.G.nodes ]
        data = sorted(data, key=lambda x:x[1])

        """
        df = pd.DataFrame( data, columns=["id", "dist_to_root"] )
        df["dist_to_root"].hist()
        plt.show()
        """

        return data
