from Data import Data
from ToGraph import ToGraph
import numpy as np


data = Data("test_case3.xyz") # Data("example.xyz")
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
graph = ToGraph(data)
graph.convert(distance=1.)
# across the z
graph.convertToDirectedG(dim=0,     # x axis
                         dir=True  # from lowest to highest
                         )

graph.levelSet(4)
