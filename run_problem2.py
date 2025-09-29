import argparse
from run_problem1 import read_block as read_block_asignacion
from models.problem2 import Routing_naive as model2_routing, routing_random, re_routing
from utils.save import save_routing
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def read_block_routing(file, N, M):
    '''
    Lee un bloque para el problema de routing desde el archivo dado
    
    Parameters
    ----------
    file : file object
        Archivo abierto para lectura
    N : int
        Numero de nodos de origen
    M : int
        Numero de paquetes
        
    Returns
    -------
    capacidad : int
        Capacidad de los vehiculos
    costo_fijo : float
        Costo fijo por vehiculo
    costo_variable : float
        Costo variable por vehiculo
    coord_I : ndarray
        Coordenadas de nodos origen (N+1 x 2), SVC es el nodo -1
    coord_J : ndarray
        Coordenadas de paquetes (M x 2)
    '''
    capacidad = list(map(int, file.readline().split()))[0]
    costo_fijo, costo_variable = list(map(float, file.readline().split()))

    coord_I = np.zeros((N+1, 2))
    coord_J = np.zeros((M, 2))
    for coord, n in zip([coord_I, coord_J], [N+1, M]):
        for _ in range(n):
            id, x, y = list(map(float, file.readline().split()))
            coord[int(id), :] = [x, y]

    return capacidad, costo_fijo, costo_variable, coord_I, coord_J

def get_output(filename, output_name, random_coberture_boxplot=False):
    '''
    Procesa todos los casos del archivo de entrada y genera los resultados

    Parameters
    ----------
    filename : str
        Nombre del archivo de entrada
    output_name : str
        Nombre del archivo de salida
    random_coberture_boxplot : bool
        Genera un boxplot alterando la cobertura original de forma random
    '''
    RANDOM_P_STEPS = 3
    NUM_COMPLETE_RANDOM_ITERATIONS = 10
    NUM_REROUTING_ITERATIONS = 20
    TIME_LIMIT_PER_ROUTING = 0.1

    with open(filename, 'r') as file:
        case_idx = 1

        # read each block
        while True:
            block1 = read_block_asignacion(file)

            if block1 is None:
                break

            I, J, _, _, _, _ = block1
            block2 = read_block_routing(file, len(I), len(J))

            # run original model
            best_solution = model2_routing(block1, block2, TIME_LIMIT_PER_ROUTING)
            original_cost = best_solution['costo_total']

            # run model redistribuye los paquetes para que queden lo mas cerca a (mod capacidad) 
            for i in range(NUM_REROUTING_ITERATIONS):
                problem2_output = re_routing(block1, block2, best_solution, TIME_LIMIT_PER_ROUTING, i)
                if problem2_output['costo_total'] < best_solution['costo_total']:
                    best_solution = problem2_output

            # run model perturbando la cobertura original de forma random
            results = []
            for p in np.linspace(0.77, 0.81, RANDOM_P_STEPS):
                for seed in range(NUM_COMPLETE_RANDOM_ITERATIONS):
                    problem2_output = routing_random(block1, block2, TIME_LIMIT_PER_ROUTING, p, seed=seed)
                    cost = problem2_output['costo_total']
                    results.append((round(p,2), cost))
                    
                    if cost < best_solution['costo_total']:
                        best_solution = problem2_output

            if random_coberture_boxplot:
                df = pd.DataFrame(results, columns=['p','cost'])
                plt.figure(figsize=(10,6))
                sns.boxplot(x='p', y='cost', data=df, width=0.5)
                sns.stripplot(x='p', y='cost', data=df, jitter=True, alpha=0.8)
                plt.axhline(original_cost, color='red', linestyle='dotted', label='Original Cost')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"images/{output_name.split('/')[1].split('.')[0]}_boxplot.png")
                plt.close()

            # run model hardcodeando la capacity
            problem2_output = model2_routing(block1, block2, TIME_LIMIT_PER_ROUTING, hardcodear_capacity=True)
            if problem2_output['costo_total'] < best_solution['costo_total']:
                best_solution = problem2_output

            print(f"Procesado case {case_idx}")
            print(f"Costo original: {original_cost}")
            print(f"Costo de la mejor solución encontrada: {best_solution['costo_total']}")

            save_routing(best_solution, output_name, case_idx)
            case_idx += 1

def main():
    parser = argparse.ArgumentParser(description="Run Problem2 Model")
    parser.add_argument('filename')
    parser.add_argument('output_name')
    args = parser.parse_args()

    get_output(args.filename, output_name = args.output_name)

if __name__ == "__main__":
    main()