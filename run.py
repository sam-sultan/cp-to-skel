from Data import Data
from ToGraph import ToGraph
import numpy as np
import networkx as nx


data = Data("example.xyz") # Data("example.xyz")
print(data.Points)
#data.export("test.xyz")
#data.center()
#data.rotate() # rotate with eignvecs
"""
# rotate with a matrix
data.rotate(np.array([ [2., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.] ]))
"""
dist = 0.5
res = None
while not isinstance(res, nx.DiGraph):
    graph = ToGraph(data)
    graph.convert(distance=dist)
    # across the z
    res = graph.convertToDirectedG(dim=1,     # x axis
                                     dir=True  # from highest to lowest
                                     )
    dist += 0.5

print(dist)
print("Done!")
#exit()
graph.levelSet(4)
graph.export("skel.csv")
