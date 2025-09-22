from models.problem1 import LastMile
from utils.plot import _plotLastMile
from utils.save import _saveLastMile
import argparse

def read_block(file):
    '''
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
        
    a = [[0]*M for _ in range(N)]

    for i in I:
        input = list(map(int, file.readline().split()))
        for j in input[1:]:
            a[i][j] = 1

    return I, J, c, S, k, a

def get_output(filename, output_name):
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
            model, x, y = LastMile(I, J, c, S, k, a)

            # save
            _saveLastMile(model, x, y, output_name, case_idx, I, J)
            case_idx += 1

def main():
    parser = argparse.ArgumentParser(description="Run Problem1 Model")
    parser.add_argument('filename')
    parser.add_argument('output_name')
    args = parser.parse_args()

    get_output(args.filename, output_name = args.output_name)

if __name__ == "__main__":
    main()