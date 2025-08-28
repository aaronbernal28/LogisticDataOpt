import pyscipopt as scip

def run_model(I, J, c, S, k, a):
    '''
    I: Conjunto de todos los nodos de entrega disponibles.
    J: Conjunto de todos los paquetes que deben ser entregados.

    c[i]: Costo fijo por entregar un paquete desde el nodo i.
    S: Costo promedio por entregar un paquete desde el Service Center.
    k[i]: Capacidad máxima (en número de paquetes) del nodo i.
    a[i,j]: es 1 si y solo si el nodo i puede entregar el paquete j
    '''
    # initialice model
    model = scip.Model("Problema_1")

    # define variables
    x = {}
    for i in I:
        for j in J:
            x[i, j] = model.addVar(vtype="BINARY", name=f"x_{i}_{j}")
    y = {}
    for j in J:
        y[j] = model.addVar(vtype="BINARY", name=f"y_{j}")

    # objective function
    model.setObjective(
        S * sum(y[j] for j in J) +
        sum(c[i] * sum(x[i, j] for j in J) for i in I),
        sense="minimize"
    )

    # constrains

    
    

