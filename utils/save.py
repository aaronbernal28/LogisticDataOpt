from pyscipopt import Model, quicksum

def saveLastMile(model, x, y, filename, case_idx, I, J):
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
        print('', file=f)

def save_routing(information_routing, filename, case_idx):
    '''
    Estructura de un Bloque de Salida
        Linea1: Costo global total minimo. #costo(model1) + #costo(model2) / 100 + #len(J) * costo_variable
        Linea2: K (Numerodeorigenesconrutas).
        K bloques sig.:Describen las rutas por origen.
    Formato de Rutas por Origen
        Linea1: id origen num vehiculos
        Lineas sig.: Una linea por vehiculo con los id paquete de su ruta.
    '''

    mode = 'a'
    if case_idx == 1:
        mode = 'w'
    
    with open(filename, mode) as f:
        print('Caso ' + str(case_idx), file=f)
        
        # Calcular costo total global
        model1 = information_routing['model1']
        costo_model1 = model1.getObjVal()
        
        # Calcular costo total del routing (model2)
        costo_model2 = 0
        for origen_id, (data, manager, routing, solution) in information_routing['rutas_por_origen'].items():
            if solution:
                costo_model2 += solution.ObjectiveValue()
        
        # Costo variable por paquete
        num_packages = len(information_routing['packages'])
        # Asumir costo_variable desde el primer data disponible
        costo_variable = 0
        if information_routing['rutas_por_origen']:
            first_data = next(iter(information_routing['rutas_por_origen'].values()))[0]
            costo_variable = first_data['vehicle_costs_variable'][0]
        
        # Costo total: costo(model1) + costo(model2)/100 + len(J) * costo_variable
        costo_total = costo_model1 + (costo_model2 / 100) + (num_packages * costo_variable)
        
        print(f'{costo_total:.2f}', file=f)
        
        # Número de orígenes con rutas
        K = len(information_routing['rutas_por_origen'])
        print(K, file=f)
        
        # Describir rutas por origen
        for origen_id, (data, manager, routing, solution) in information_routing['rutas_por_origen'].items():
            if solution:
                # Extraer rutas de la solución
                rutas_vehiculos = []
                
                for vehicle_id in range(data['num_vehicles']):
                    ruta = []
                    index = routing.Start(vehicle_id)
                    
                    while not routing.IsEnd(index):
                        node_index = manager.IndexToNode(index)
                        # Si no es el depot, agregar el paquete a la ruta
                        if node_index != data['depot']:
                            paquete_id = data['paquetes'][node_index]
                            ruta.append(paquete_id)
                        index = solution.Value(routing.NextVar(index))
                    
                    # Solo incluir vehículos que tienen paquetes asignados
                    if ruta:
                        rutas_vehiculos.append(ruta)
                
                # Escribir información del origen
                num_vehiculos_usados = len(rutas_vehiculos)
                print(f'{origen_id} {num_vehiculos_usados}', file=f)
                
                # Escribir rutas de cada vehículo
                for ruta in rutas_vehiculos:
                    print(' '.join(map(str, ruta)), file=f)
        
        print('', file=f)