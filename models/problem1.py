from pyscipopt import Model, quicksum

def LastMile(I, J, c, S, k, a):
    '''
    Input
    -------
    I: Conjunto de todos los nodos de entrega disponibles.
    J: Conjunto de todos los paquetes que deben ser entregados.

    c[i]: Costo fijo por entregar un paquete desde el nodo i.
    S: Costo promedio por entregar un paquete desde el Service Center.
    k[i]: Capacidad máxima (en número de paquetes) del nodo i.
    a[i,j]: es 1 si y solo si el nodo i puede entregar el paquete j

    Returns
    -------
    pySCIPopt model object
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
            model.addCons(x[i, j] <= a[i][j])

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