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
    
    for i in I + [-1]:
        if len(asignaciones[i]) > 0:
            ''' -1 is the node i temporally'''
            data = {}
            data['indexes'] = asignaciones[i] + [-1]

            # define distance function (in this case euclidean)
            coords = np.vstack([coord_J[asignaciones[i]], coord_I[i]])
            data['distance_matrix'] = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
            
            data['demands'] = [1] * len(data['indexes'])
            data["vehicle_capacities"] = [capacidad]*len(J)
            data["num_vehicles"] = len(J) # no esta acotado
            data["depot"] = -1
            output = run_routing(data)
    return None

def run_routing(data):
    return None

if __name__ == "__main__":
    coords = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    print(dist_matrix)
    print(coords[:, np.newaxis] - coords[np.newaxis, :])