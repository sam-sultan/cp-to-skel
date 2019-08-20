from Data import Data
from ToGraph import ToGraph
import numpy as np


data = Data("example.xyz")
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
graph.convert(distance=2.)
exit()
# across the z
graph.convertToDirectedG(dim=1,     # z axis
                         dir=False  # from highest to lowest
                         )
graph.sortNodes()
