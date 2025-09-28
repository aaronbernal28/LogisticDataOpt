from problem1 import LastMile as asignacion
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

def Routing_naive(block1, block2, time_limit_per_routing=5):
    '''
    Procesa un input case y resuelve el routing para cada origen con paquetes asignados
    
    Parameters
    ----------
    block1 : tuple
        Tupla con (I, J, c, S, k, a) donde:
            I: array_like - Nodos de origen
            J: array_like - Paquetes  
            c: array_like - Costos por nodo
            S: float - Costo por SVC
            k: array_like - Capacidad maxima por nodo
            a: ndarray - Cobertura IxJ (nodo -> paquete)
    block2 : tuple
        Tupla con (capacidad, costo_fijo, costo_variable, coord_I, coord_J) donde:
            capacidad: int - Capacidad de vehiculos
            costo_fijo: float - Costo fijo por vehiculo
            costo_variable: float - Costo variable por vehiculo
            coord_I: ndarray - Coordenadas de nodos origen
            coord_J: ndarray - Coordenadas de paquetes
    time_limit_per_routing : int

    Returns
    -------
    information_routing : dict
        Diccionario con informacion de rutas por origen, paquetes y modelo
    '''
    I, J, _, _, _, _ = block1
    capacidad, costo_fijo, costo_variable, coord_I, coord_J = block2
    model1, x, y = asignacion(*block1)

    # Extraer asignaciones del modelo
    asignaciones = {i: [] for i in range(-1, len(I))}
    for j in J:
        for i in I:
            if model1.getVal(x[i, j]) == 1:
                asignaciones[i].append(j)
        if model1.getVal(y[j]) == 1:
            asignaciones[-1].append(j)

    information_routing = {
        'rutas_por_origen': {},
        'packages': J
    }

    # Procesar routing para cada origen con paquetes asignados
    for origen_id, paquetes_asignados in asignaciones.items():
        if not paquetes_asignados:
            continue
            
        # Mostrar informaciÃ³n del origen
        if origen_id == -1:
            print(f'\n--- SVC routing ---')
        else:
            print(f'\n--- Node_entrega {origen_id} routing ---')
        print(f'Packages: {paquetes_asignados}')
        
        num_paquetes = len(paquetes_asignados)
        data = {
            # Variables de los vehÃ­culos
            'num_vehicles': num_paquetes,
            'vehicle_capacities': [capacidad] * num_paquetes,
            'vehicle_costs_fixed': [costo_fijo] * num_paquetes,
            'vehicle_costs_variable': [costo_variable] * num_paquetes,
            
            'depot': num_paquetes,  # Index del origen en la distance_matrix
            'paquetes': paquetes_asignados
        }

        # Matriz de distancias: [paquetes, origen]
        coords = np.vstack([coord_J[paquetes_asignados], coord_I[origen_id]])
        data['distance_matrix'] = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
        
        # Demanda (paquetes=1, depot=0)
        data['demands'] = [1] * num_paquetes + [0]

        manager, routing, solution = run_routing(data, time_limit_per_routing)

        # Guardo todo para ser procesado aparte
        information_routing['rutas_por_origen'][origen_id] = [data, manager, routing, solution]
    
    information_routing['costo_total'] = total_cost(information_routing, model1)
    return information_routing

def run_routing(data, time_limit_per_routing=5):
    '''
    Resuelve el ruteo para un origen y un conjunto de paquetes usando OR-Tools
    
    Parameters
    ----------
    data : dict
        Diccionario con datos del problema de routing:
        - distance_matrix: matriz de distancias
        - num_vehicles: numero de vehiculos  
        - vehicle_capacities: capacidades de vehiculos
        - vehicle_costs_fixed: costos fijos por vehiculo
        - depot: indice del deposito/origen
        - demands: demandas por paquetes
        
    Returns
    -------
    manager : pywrapcp.RoutingIndexManager
        Gestor de indices de routing
    routing : pywrapcp.RoutingModel  
        Modelo de routing
    solution : pywrapcp.Assignment
        Solucion encontrada
        
    Notes
    -----
    Source: https://developers.google.com/optimization/routing/cvrp 
    '''
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), # number of nodes
        data['num_vehicles'],
        data['depot']
    )

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data["distance_matrix"][from_node][to_node]*100)  # Scale to avoid floating point

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # vehicles cant temporarily exceed their capacity
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Add fixed cost for using vehicles
    for vehicle_id in range(data['num_vehicles']):
        routing.SetFixedCostOfVehicle(
            int(data['vehicle_costs_fixed'][vehicle_id] * 100), # Scale to avoid floating point
            vehicle_id)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(time_limit_per_routing)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    
    return manager, routing, solution

def total_cost(information_routing, model1):
    '''
    Parameters
    ----------
    information_routing : 

    Returns
    -------
    costo_total : Float
    '''
    costo_model1 = model1.getObjVal()
    
    # Calcula el costo total del routing (model2)
    costo_model2 = 0
    for origen_id, (data, manager, routing, solution) in information_routing['rutas_por_origen'].items():
        if solution:
            costo_model2 += solution.ObjectiveValue()
    
    # Costo variable por paquete
    num_packages = len(information_routing['packages'])
    # Asumir costo_variable desde el primer data disponible
    costo_variable = 0
    if information_routing['rutas_por_origen']:
        first_data = next(iter(information_routing['rutas_por_origen'].values()))[0]
        costo_variable = first_data['vehicle_costs_variable'][0]
    
    # Costo total: costo(model1) + costo(model2)/100 + len(J) * costo_variable
    costo_total = costo_model1 + (costo_model2 / 100) + (num_packages * costo_variable)
    return costo_total

def routing_random(block1, block2, time_limit_per_routing, p, seed=28):
    '''
    Apaga algunos alcances de forma random y resuelve el routing
    
    Parameters
    ----------
    block1 : tuple
        Tupla con (I, J, c, S, k, a)
    block2 : tuple
        Tupla con (capacidad, costo_fijo, costo_variable, coord_I, coord_J)
    time_limit_per_routing : int
    p : float
        Entre 0, 1
    Returns
    -------
    information_routing : dict
        Diccionario con informacion de rutas por origen
    '''
    I, J, c, S, k, a = block1
    np.random.seed(seed)
    mask = np.random.binomial(n=1, p=p, size=a.shape)
    a = a * mask

    block1 = I, J, c, S, k, a
    return Routing_naive(block1, block2, time_limit_per_routing)

if __name__ == "__main__":
    coords = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    print(dist_matrix)
    print(coords[:, np.newaxis] - coords[np.newaxis, :])