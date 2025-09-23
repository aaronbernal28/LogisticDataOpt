import argparse
from run_problem1 import read_block as read_block_asignacion
from models.problem2 import Routing_naive as model2_routing
from utils.save import save_routing
import numpy as np

def read_block_routing(file, N, M):
    ''' Reads a block for the routing problem from the given file. SVC is the node -1'''
    capacidad = list(map(int, file.readline().split()))[0]
    costo_fijo, costo_variable = list(map(float, file.readline().split()))

    coord_I = np.zeros((N+1, 2))
    coord_J = np.zeros((M, 2))
    for coord, n in zip([coord_I, coord_J], [N+1, M]):
        for _ in range(n):
            id, x, y = list(map(float, file.readline().split()))
            coord[int(id), :] = [x, y]

    return capacidad, costo_fijo, costo_variable, coord_I, coord_J

def get_output(filename, output_name):
    with open(filename, 'r') as file:
        case_idx = 1

        # read each block
        while True:
            block1 = read_block_asignacion(file)

            if block1 is None:
                break

            I, J, _, _, _, _ = block1
            block2 = read_block_routing(file, len(I), len(J))

            print(f"Processing case {case_idx}")

            # run models
            problem2_output = model2_routing(block1, block2)

            # save
            save_routing(problem2_output, output_name, case_idx)
            case_idx += 1

def main():
    parser = argparse.ArgumentParser(description="Run Problem2 Model")
    parser.add_argument('filename')
    parser.add_argument('output_name')
    args = parser.parse_args()

    get_output(args.filename, output_name = args.output_name)

if __name__ == "__main__":
    main()