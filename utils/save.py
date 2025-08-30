from pyscipopt import Model, quicksum

def _saveLastMile(model, x, y, filename, case_idx, I, J):
    '''
    model : pySCIPopt LastMile model object
    x : decision variables for nodes to packages
    y : decision variables for SC
    filename : output file name
    case_idx : index of the actual case
    '''
    mode = 'a'
    if case_idx == 1:
        mode = 'w'
    
    with open(filename, mode) as f:
        print('Caso ' + str(case_idx), file=f)
        print(round(model.getObjVal(), 2), file=f)

        for j in J:
            for i in I:
                if model.getVal(x[i, j]) == 1:
                    print(j, i, file=f)

            if model.getVal(y[j]) == 1:
                print(j, -1, file=f)