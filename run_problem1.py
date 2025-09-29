from models.problem1 import asignacion
from utils.plot import _plotLastMile
from utils.save import saveLastMile
import argparse
import numpy as np

def read_block(file):
    ''' Lee un case del objeto file
    Parameters
    ----------
    file : ?
        Referencia a un file

    Returns
    -------
    I : array_like
        Nodos
    J : array_like
        Paquetes
    c : array_like
        Costos por nodo
    S : Float
        Costo por SVC
    k : array_like
        Capacidad maxima por nodo
    a : ndarray
        Cobertura (nodo -> paquete)

    Notes
    -----
    readline() reads a single line from the file
    strip() removes leading and trailing whitespace
    split() splits the string into a list where each word is a list item
    '''
    N, M = list(map(int, file.readline().split()))

    if N == M == 0:
        # end of input, break while
        return None

    S = list(map(float, file.readline().split()))[0]

    I = list(range(N))
    J = list(range(M))

    k = []
    c = []
        
    for _ in I:
        capacidad, costo = list(map(float, file.readline().split()))
        k.append(int(capacidad))
        c.append(costo)
        
    a = np.array([[0]*M for _ in range(N)])

    for i in I:
        input = list(map(int, file.readline().split()))
        for j in input[1:]:
            a[i, j] = 1

    return I, J, c, S, k, a

def get_output(filename, output_name):
    ''' Procesa cada case por separado y los guarda en output_name
    Parameters
    ----------
    filename : str
        Nombre del archivo de entreda
    output_name : str
        Nombre del archivo de salida
    '''
    with open(filename, 'r') as file:
        case_idx = 1

        # read each block
        while True:
            block = read_block(file)
            if block is None:
                break

            print(f"Processing case {case_idx}")

            # run model
            I, J, c, S, k, a = block
            model, x, y = asignacion(I, J, c, S, k, a)

            # save
            saveLastMile(model, x, y, output_name, case_idx, I, J)
            case_idx += 1

def main():
    parser = argparse.ArgumentParser(description="Run Problem1 Model")
    parser.add_argument('filename')
    parser.add_argument('output_name')
    args = parser.parse_args()

    get_output(args.filename, output_name = args.output_name)

if __name__ == "__main__":
    main()