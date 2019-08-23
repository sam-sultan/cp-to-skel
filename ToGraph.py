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

        print("Converting...")

        # Points and get rid of duplicates
        Points = self.Data.Points

        # Kd tree
        self.tree = KDTree(Points, leaf_size=2, metric='euclidean')

        print("Kd tree has been set!")

        self.distance = distance #self.medianPD(Points)
        #print(self.medianPD(Points))
        #res = self.tree.kernel_density(Points, h=0.5)
        #print(res/sum(res))

        #print("Querying the nearest neighbors...")
        # get nearest points
        pts, distances = self.tree.query_radius( Points , r=distance, return_distance=True, sort_results=True)
        print("Done querying!")

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

        subs = nx.connected_component_subgraphs(self.G)
        # grab the graph with highest number of nodes
        self.G = sorted([ (f, len(list(f.nodes))) for f in subs ], reverse=True)[0][0]


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

        print("Converting to DiGraph...")

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
                    print("No path with distance", self.distance)
                    return self.distance

        self.G = G
        """
        #print(Points[rootidx,:])
        #print([ (id, G.nodes[id]) for id in G.nodes ])
        """

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


        # convert these generated figures
        data = Data(centroids)
        g = ToGraph(data)
        g.convert(step*1.2)
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
