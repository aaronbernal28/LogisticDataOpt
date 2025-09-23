from problem1 import LastMile as asignacion
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import numpy as np

def Routing_naive(block1, block2):
    I, J, c, S, k, a = block1
    capacidad, costo_fijo, costo_variable, coord_I, coord_J = block2
    model, x, y = asignacion(I, J, c, S, k, a)
    cost_1 = round(model.getObjVal(), 2)

    asignaciones = {i: [] for i in range(-1, len(I))}
    for j in J:
        for i in I:
            if model.getVal(x[i, j]) == 1:
                asignaciones[i].append(j)
        if model.getVal(y[j]) == 1:
            asignaciones[-1].append(j)
    
    total_cost_routing = 0
    for i in asignaciones.keys():
        if len(asignaciones[i]) > 0:
            if i == -1:
                print(f'\n--- SVC routing ---')
            else:
                print(f'\n--- Node_entrega {i} routing ---')
            print(f'Packages: {asignaciones[i]}')
            
            data = {}
            data["depot"] = len(asignaciones[i])  # depot is the last node in the distance matrix
            data['node_entrega'] = i  # Store delivery node ID for reference
            if i == -1:
                data['indexes'] = asignaciones[i] + ["SVC"]  # SVC as depot
            else:
                data['indexes'] = asignaciones[i] + [f"node_entrega_{i}"]  # Keep original indices

            coords = np.vstack([coord_J[asignaciones[i]], coord_I[i]])
            
            data['distance_matrix'] = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
            
            # define demand of each node (customers=1, depot=0)
            data['demands'] = [1] * len(asignaciones[i]) + [0]  # depot has 0 demand

            # define vehicle data
            data["num_vehicles"] = len(asignaciones[i])  # One vehicle per customer max
            data["vehicle_capacities"] = [capacidad] * len(asignaciones[i])
            data["vehicle_costs_fixed"] = [costo_fijo] * len(asignaciones[i])
            data["vehicle_costs_variable"] = [costo_variable] * len(asignaciones[i])
            
            print('Demands:', data['demands'])
            print('---------')

            total_cost_routing += run_routing(data)
    
    total_cost_routing = round(total_cost_routing + costo_variable * len(J), 2) # Add cost for delivering to each package
    print(f"\n=== SUMMARY ===")
    print(f"Assignment cost (from problem1): {cost_1}")
    print(f"Total routing cost: {total_cost_routing}")
    print(f"Total combined cost: {cost_1 + total_cost_routing}")
    
    return cost_1 + total_cost_routing

def run_routing(data):
    '''From https://developers.google.com/optimization/routing/cvrp'''
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
    search_parameters.time_limit.FromSeconds(10)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        total_cost = print_solution(data, manager, routing, solution)
        return total_cost
    else:
        print("No solution found!")
        return np.inf

def print_solution(data, manager, routing, solution):
    """Prints solution on console with original indices and cost breakdown."""
    print(f"Objective: {solution.ObjectiveValue() / 100}")  # Scale back
    
    total_distance = 0
    total_load = 0
    vehicles_used = 0
    
    for vehicle_id in range(data["num_vehicles"]):
        if not routing.IsVehicleUsed(solution, vehicle_id):
            continue
            
        vehicles_used += 1
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            
            # Map back to original index
            if node_index == data['depot']:
                if data['node_entrega'] == -1:
                    original_index = "SVC"
                else:
                    original_index = f"node_entrega_{data['node_entrega']}"
            else:
                original_index = data['indexes'][node_index]
                
            route_load += data["demands"][node_index]
            plan_output += f" {original_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
            
        # Handle final node (depot)
        final_node_index = manager.IndexToNode(index)
        if data['node_entrega'] == -1:
            final_original_index = "SVC"
        else:
            final_original_index = f"node_entrega_{data['node_entrega']}"
        
        plan_output += f" {final_original_index} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance / 100}m\n"  # Scale back
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    
    # Cost breakdown
    total_fixed_cost = vehicles_used * data['vehicle_costs_fixed'][0] if len(data['vehicle_costs_fixed']) > 0 else 0
    
    print(f"Total distance of all routes: {total_distance / 100}m")
    print(f"Total load of all routes: {total_load}")
    print(f"Vehicles used: {vehicles_used}")
    print(f"Fixed costs: {vehicles_used} vehicles x {data['vehicle_costs_fixed'][0]} = {total_fixed_cost}")
    
    return total_fixed_cost

if __name__ == "__main__":
    coords = np.array([[0, 0], [1, 2], [2, 1], [3, 3]])
    dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=-1)
    print(dist_matrix)
    print(coords[:, np.newaxis] - coords[np.newaxis, :])