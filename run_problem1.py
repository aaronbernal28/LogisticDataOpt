from models.problem1 import LastMile
import argparse

def read_data(filename):
    # ?
    with open(filename, 'r') as file:
        '''
        readline() reads a single line from the file
        strip() removes leading and trailing whitespace
        split() splits the string into a list where each word is a list item
        '''
        N, M = list(map(int, file.readline().strip().split()))
        
    return I, J, c, S, k, a

def main():
    parser = argparse.ArgumentParser(description="Run Problem1 Model")
    parser.add_argument('filename.in')
    args = parser.parse_args()

    I, J, c, S, k, a = read_data(args.filename)

    model, x, y = LastMile(I, J, c, S, k, a)

    if model.getStatus() == "optimal":
        print("Optimal solution found:")
        for i in I:
            for j in J:
                if model.getVal(x[i, j]) > 0.5:
                    print(f"Package {j} delivered from node {i}")
        for j in J:
            if model.getVal(y[j]) > 0.5:
                print(f"Package {j} delivered from Service Center")
    else:
        print("No optimal solution found.")

if __name__ == "__main__":
    main()