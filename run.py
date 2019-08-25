from Data import Data
from ToGraph import ToGraph
import numpy as np
import networkx as nx


data = Data("example.xyz") # Data("example.xyz")
print(data.Points, "\n")
#data.export("test.xyz")
#data.center()
#data.rotate() # rotate with eignvecs
"""
# rotate with a matrix
data.rotate(np.array([ [2., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.] ]))
"""
graph = ToGraph(data)
graph.convert(distance=0.5, largest=True, show=False)
# across the z
graph.convertToDirectedG(dim=1,     # x axis
                         dir=True,  # from highest to lowest
                         show=False)
print("Done!")
graph.levelSet(15) # <--------------- Problems here
graph.export("skel.csv")
