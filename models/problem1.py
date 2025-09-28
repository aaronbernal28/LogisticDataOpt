from pyscipopt import Model, quicksum

def LastMile(I, J, c, S, k, a):
    '''
    Parameters
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
        Cobertura IxJ (nodo -> paquete)

    Returns
    -------
    model : pySCIPopt model object
    x : dict pySCIPopt
    y : dict pySCIPopt
    '''
    M = len(J)
    
    # initialice model
    model = Model("Problema_1")

    # define variables
    x = {}
    for i in I:
        for j in J:
            x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")
    
    y = {}
    for j in J:
        y[j] = model.addVar(vtype="B", name=f"y_{j}")

    # objective function
    model.setObjective(
        S * quicksum(y[j] for j in J) +
        quicksum(c[i] * quicksum(x[i, j] for j in J) for i in I),
        sense="minimize"
    )

    # constrains

    ## cobertura
    for i in I:
        for j in J:
            model.addCons(x[i, j] <= a[i, j])

    ## capacidad
    for i in I:
        model.addCons(quicksum(x[i,j] for j in J) <= k[i])

    ## cons 3
    for j in J:
        model.addCons(quicksum(x[i,j] for i in I) + y[j] == 1)
    
    ## cons 4
    model.addCons(quicksum(quicksum(x[i,j] for i in I) + y[j] for j in J) == M)

    model.optimize()

    return model, x, y