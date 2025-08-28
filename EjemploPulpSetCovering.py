import pulp

# --- 1. Definición de Datos del Problema ---

# Conjunto de SVCs que necesitan ser cubiertos
svcs = [f"SVC_{i}" for i in range(1, 11)]

# Conjunto de ubicaciones candidatas para los depositos
depositos_candidatos = ["Deposito_A", "Deposito_B", "Deposito_C", "Deposito_D", "Deposito_E"]

# Definimos qué SVCs puede cubrir cada deposito.
# Esta es la matriz 'a_ij' del modelo teórico.
# Un diccionario es una forma muy clara de representar esta relación.
cobertura = {
    "Deposito_A": ["SVC_1", "SVC_3", "SVC_10"],
    "Deposito_B": ["SVC_2", "SVC_4", "SVC_6"],
    "Deposito_C": ["SVC_1", "SVC_5", "SVC_7", "SVC_9"],
    "Deposito_D": ["SVC_2", "SVC_8", "SVC_10"],
    "Deposito_E": ["SVC_3", "SVC_5", "SVC_6", "SVC_9"]
}

# --- 2. Creación del Modelo ---

# Se crea el objeto del problema, especificando que es de minimización
modelo = pulp.LpProblem("Ubicacion_Depositos_SetCover", pulp.LpMinimize)

# --- 3. Definición de las Variables de Decisión ---

# x_j: 1 si se decide abrir el deposito j, 0 si no.
abrir = pulp.LpVariable.dicts("Abrir_Deposito", depositos_candidatos, cat='Binary')

# --- 4. Definición de la Función Objetivo ---

# El objetivo es minimizar el número total de depositos abiertos.
# Esto es simplemente la suma de todas las variables de decisión 'abrir'.
modelo += pulp.lpSum(abrir[j] for j in depositos_candidatos), "Numero_Total_de_Depositos"

# --- 5. Definición de las Restricciones ---

# Restricción de Cobertura: Para cada SVC, al menos un deposito que lo
# cubra debe ser seleccionado.
for i in svcs:
    # Para el SVC 'i', sumamos las variables de los depositos 'j' que lo cubren
    expresion_cobertura = pulp.lpSum(abrir[j] for j in depositos_candidatos if i in cobertura[j])
    
    # La suma debe ser mayor o igual a 1
    modelo += expresion_cobertura >= 1, f"Cobertura_de_{i}"

# --- 6. Solución del Problema ---

# Se invoca al solver para que encuentre la solución óptima
modelo.solve()

# --- 7. Visualización de los Resultados ---

print(f"Estado de la solución: {pulp.LpStatus[modelo.status]}")
print("-" * 30)
# El valor del objetivo es el número mínimo de depositos
min_depositos = int(pulp.value(modelo.objective))
print(f"Número mínimo de depositos a abrir: {min_depositos}")
print("-" * 30)
print("Depositos seleccionados:")

# Imprimimos solo los depositos cuya variable de decisión sea 1
for j in depositos_candidatos:
    if pulp.value(abrir[j]) == 1:
        print(f"  -> {j} (cubre a: {', '.join(cobertura[j])})")