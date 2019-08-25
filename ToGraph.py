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
    def convert(self, distance, total=100000, largest=False, show=False):

        print("Converting...")

        # Points and get rid of duplicates
        Points = self.Data.Points

        # Kd tree
        self.tree = KDTree(Points, leaf_size=2, metric='euclidean')

        print("Kd tree has been set!")

        #print(self.medianPD(Points))
        #res = self.tree.kernel_density(Points, h=0.5)
        #print(res/sum(res))

        #print("Querying the nearest neighbors...")
        # get nearest points
        # find the smallest distance that the query will return with at least 2 nearby neighbors
        self.distance = distance
        pts, distances = self.tree.query_radius( Points , r=distance, return_distance=True, sort_results=True)
        minPts = len(min(pts, key= lambda x:len(x)))
        while minPts < 2:
            self.distance += 0.5
            distance = self.distance
            pts, distances = self.tree.query_radius( Points , r=distance, return_distance=True, sort_results=True)
            minPts = len(min(pts, key= lambda x:len(x)))

        print("going with ", distance)

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
        if show:
            nx.draw(self.G, with_labels=True, font_weight='bold')
            plt.show()
        #print("Done!")

        # grab the graph with highest number of nodes
        if largest:
            self.G = max(nx.connected_component_subgraphs(self.G), key=len)


        return self.G

    """
    Convert graph to a directed graph
    # dimension: 0 -> x
                 1 -> y
                 2 -> z
    # dir: False -> starting from +inf -> -inf
           True  -> starting from -inf -> +inf
    """
    def convertToDirectedG(self, dim, dir=False, show=False):

        # make sure the dimensions are with in limit
        assert 0 <= dim <= 2, "Please make sure the dim is between 0 and 2"

        # get ref to points
        Points = self.Data.Points

        print("Converting to DiGraph...", "Nodes:", len(list(self.G.nodes)))

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

        # shortest path between every node and the source rootidx
        dists, paths = nx.single_source_dijkstra(self.G, rootidx, weight="weight")
        
        for n in dists.keys():
            path = paths[n]
            dist = dists[n]
            if len(path) > 1:
                G.add_weighted_edges_from([(path[len(path) - 1], path[len(path) - 2], dist )])
                G.nodes[path[len(path) - 1]]['dist'] = dist
            else:
                G.add_node(n)
                G.nodes[n]['dist'] = 0.

        self.G = G

        #print(Points[rootidx,:])
        #print([ (id, G.nodes[id]) for id in G.nodes ])

        if show:
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
            

        return G



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
        distance = 0.
        while s <= longest:

            cpIdx = []
            for i in range(c, len(pts)):
                pt = pts[i]
                if pt[1] > s:
                    break
                cpIdx.append(pt[0])
                c+=1

            if len(cpIdx) > 1:
                # convert these points to graph
                subData = Data(np.array(Points[cpIdx]))
                g = ToGraph(subData)
                H = g.convert(self.distance)
                Hs = list(nx.connected_component_subgraphs(H))
                #print( [ list(h.nodes) for h in Hs ] )
                for h in Hs:
                    points = subData.Points[list(h.nodes)]
                    centroid = np.mean(points, axis=0)
                    centroids.append(centroid)

            s += step

        centroids = np.array(centroids)


        # convert these generated figures
        data = Data(centroids)
        g = ToGraph(data)
        g.convert(0.5)
        self.G = g.convertToDirectedG(self.dim, self.dir)
        self.Data = data

        return self.G

    """
    Export the directed Graph
    """
    def export(self, fname):

        Points = self.Data.Points

        with open(fname, "w+") as f:
            f.write("# nodes, parent, x, y, z\n")
            for n in self.G.nodes:
                parents = list(self.G.successors(n))
                f.write(str(n)+", "+str( "" if len(parents) <= 0 else parents[0] )+","+ ",".join( list(map(str, Points[n])) ) + "\n")


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
