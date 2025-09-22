from problem1 import LastMile as asignacion
from ortools.constraint_solver import pywrapcp
import numpy as np

def Routing_naive(block1, block2):
    I, J, c, S, k, a = block1
    capacidad, costo_fijo, costo_variable, coord_I, coord_J = block2
    model, x, y = asignacion(I, J, c, S, k, a)
    cost_1 = round(model.getObjVal(), 2)
    asignaciones = {i: [] for i in range(len(I))}

    for j in J:
        for i in I:
            if model.getVal(x[i, j]) == 1:
                asignaciones[i].append(j)
    
    for i in I:
        if len(asignaciones[i]) > 0:
            ''' -1 is the node i temporally'''
            coords = np.vstack([coord_J[asignaciones[i]], coord_I[i]])

            # define distance function (in this case euclidean)
            dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
            def dist(i, j):
                return int(dist_matrix[i, j])
            
            indexes = asignaciones[i] + [-1]
            output = run_routing(indexes, dist, capacidad, costo_fijo, costo_variable)
    return None

def run_routing(indexes, dist, capacidad, costo_fijo, costo_variable):
    return None


if __name__ == "__main__":
    coords = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    print(dist_matrix)
    print(coords[:, np.newaxis] - coords[np.newaxis, :])