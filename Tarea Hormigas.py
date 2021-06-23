##################################
# Sistema de hormigas PVA
##################################

import numpy as np
import matplotlib.pyplot as plt
import timeit

#Se incializa el contador de tiempo
start = timeit.default_timer()

#Se extraen los datos del archivo proporcionado
archivo = open('CoordenadasCiudades.txt')
coordenadasCiudades = []

for linea in archivo:
    coordenadas = linea.replace(',','').replace('[','').replace(']','').split(' ')
    coordenadasCiudades.append([float(coordenadas[0]) , float(coordenadas[1])])


##################################
# Parámetros del algoritmo
nCiudades = len(coordenadasCiudades) #Se obtiene el número de ciudades a recorrer
nHormigas = 20  #Se definie el número de hormigas del sistema
alpha = 1.0     #Se define el coeficiente de atracción de la feromona
beta = 5.0      #Se define el coeficiente de preferencia a caminos más cercanos
rho = 0.1       #Se define la tasa de vaporización de las feromonas

# Funciones
########################################################
def DistanciaEntreCiudades(nodo1, nodo2, coordCiudades):
    """
    Cálculo de la longitud euclídea entre dos ciudades
    ---
    Entradas:nodo1, nodo2: enteros, índices de la ciudades a calcular; arreglo de coordenadas de las ciudades
    """
    deltaX = coordCiudades[int(nodo2)][0]-coordCiudades[int(nodo1)][0]
    deltaY= coordCiudades[int(nodo2)][1]-coordCiudades[int(nodo1)][1]
    longitudTrayectoria=np.sqrt(deltaX**2+deltaY**2)    
    
    return longitudTrayectoria

#########################################################
def ObtenerLongitudCaminoVecinosMásCercanos(coordenadasCiudades):
    """
    Esta función obtiene la longitud del camino de los vecinos más cercanos
    para las coordenadas de las ciudades utilizadas con una ciudad inicial seleccionada
    aleatoriamente. Esta corresponde a una longitud estimada que solo toma en cuenta los 
    nodos más cercanos, lo que no necesariemente es el camino más corto
    ---
    Entrada: arreglo de coordenadas de las ciudades
    Salida: La función retorna el valor de la longitud del camino de los vecinos más cercanos
    """
    #Parámetros iniciales
    longitudCaminoMásCercano = 0
    nCiudades = len(coordenadasCiudades)
    ciudadInicial = np.random.randint(nCiudades) #Se selecciona la ciudad inicial aleatoriamente
    
    ciudadActual = ciudadInicial
    ciudadesVisitadas = [ciudadInicial]
    
    for iIteración in range(nCiudades-1):
        #Se inicializa la distancia al vecino más cercano con un valor muy grande al igual que en el ciclo principal del SH
        longitudCaminoVecinoMásCercano = 1e4
        
        #Se itera sobre todas las ciudades
        for kCiudad in range(nCiudades):
            #Se calcula la distancia entre la ciudad actual y la kCiudad
            longitudCamino = DistanciaEntreCiudades(ciudadActual, kCiudad, coordenadasCiudades)
            
            #Se acualiza el valor de la distancia al vecino más cercano y se actualiza la ciudad más cercana si se cumplen las condiciones
            if (longitudCamino < longitudCaminoVecinoMásCercano) and (longitudCamino > 0) and (kCiudad not in ciudadesVisitadas):
                longitudCaminoVecinoMásCercano = longitudCamino
                ciudadMásCercana = kCiudad
        
        #Se guardan los resultados        
        longitudCaminoMásCercano += longitudCaminoVecinoMásCercano
        ciudadActual = ciudadMásCercana
        ciudadesVisitadas.append(ciudadMásCercana)
        
    return longitudCaminoMásCercano


#########################################################
def InicializarNivelFeromonas(nCiudades, tau_0):
    """
    Inicialización de los niveles de feromonas de la simulación
    Salidas: arreglo de nCiudades x nCiudades con tau_0 como valor
    inicial de feromona para cada arista
    """
    nivelFeromonas = np.zeros((nCiudades , nCiudades)) + tau_0
    return nivelFeromonas

#########################################################
def ObtenerVisibilidad(coordCiudades):
    """
    Usa el arreglo de coordenadas de las ciudades para obtener las distancias
    entre todas las ciudades. La visibilidad corresponde al parámetro eta del
    algoritmo.
    ---
    Salida arregloVisibilidad: matriz simétrica con las distancias entre todas
    las ciudades
    """
    #Se obtiene el número de ciudades y se incializa la matriz visibilidad
    nCiudades = len(coordCiudades)
    arregloVisibilidad = np.zeros((nCiudades , nCiudades))
    
    #Se calculan las distancias  entre todas las ciudades
    for iNodo in range(nCiudades):
        for jNodo in range(nCiudades):
            if iNodo != jNodo:
                longitudCamino = DistanciaEntreCiudades(jNodo , iNodo , coordCiudades)
                arregloVisibilidad[iNodo , jNodo] = 1./longitudCamino
                arregloVisibilidad[jNodo , iNodo] = 1./longitudCamino
                
    return arregloVisibilidad

#########################################################
def GenerarCamino(nivelFeromonas, visibilidad, alpha, beta):
    '''
    Genera la lista de nodos recorridos por la k-ésima hormiga
    Entradas
    ----------
    nivelFeromonas : Corresponde a la matriz del nivel de feromonas.
    visibilidad : Corresponde a la matriz de visibilidad de cada nodo.
    alpha : coeficiente de atracción de la feromona.
    beta : coeficiente de preferencia a caminos más cercanos.
    Returns
    -------
    caminoGenerado : Retorna el camino recorrido por la hormiga correspondiente
    '''
    #Se obtiene el número de ciudades
    nCiudades , _ = nivelFeromonas.shape
    
    #Se selecciona la ciudad inicial aleatoriamente
    nodoInicial = np.random.randint(nCiudades)
    listaTabú = [nodoInicial] #Se añade esta a la lista Tabú
    
    iNodo = nodoInicial
    for jIteracion in range(nCiudades-1):
        matrizProbabilidad = np.zeros((nCiudades-jIteracion , 2)) #Se incializa la matriz de probabilidades que tendrá la probabilidad y el nodo correspondiente
                                                    
        contador = 0
        for jNodo in range(nCiudades):
            if jNodo not in listaTabú:
                contador += 1
                probNodo = ObtenerProbabilidad(nivelFeromonas , visibilidad, alpha, beta, listaTabú, iNodo, jNodo) #Se obtiene la probabilidad de pasar al nodo correspondiente
                matrizProbabilidad[contador , 0] = probNodo #Se guarda en la matriz de probabilidad
                matrizProbabilidad[contador , 1] = jNodo
                
        iNodo = ObtenerNodo(matrizProbabilidad) #Se obtiene el siguiente nodo según la probabilidad
        listaTabú.append(iNodo) #Se guarda en la lista tabú
        
    listaTabú.append(nodoInicial) #Se añade el nodo inicial al final del camino
    caminoGenerado = listaTabú
    
    return caminoGenerado

##########################################################

def ObtenerProbabilidad(nivelFeromonas, visibilidad, alpha, beta,
                        listaTabú, iNodo, jNodo):
    """
    probabilidad condicional p(Cij|S)
    Retorna la probabilidad de ir del nodo i al nodo j
    """
    nCiudades , _ =nivelFeromonas.shape
    sumaTemp = 0.0
    
    #Se calcula suma correspondiente al denominador de la probabilidad condicional
    for lNodo in range(nCiudades):
        if lNodo not in listaTabú:
            sumaTemp += (nivelFeromonas[int(lNodo)][int(jNodo)])**alpha * (visibilidad[int(lNodo)][int(jNodo)])**beta
    
    #Se calcula la probabilidad correspondiente
    probabilidadNodo = (nivelFeromonas[int(iNodo)][int(jNodo)]**alpha*visibilidad[int(iNodo)][int(jNodo)]**beta)/sumaTemp
    
    
    return probabilidadNodo


############################################################
def ObtenerNodo(matrizProb):
    """
    Retorna el próximo nodo en el camino de manera estocástica según la probabilidad correspondiente
    Salida: índice del próximo nodo
    """
    nNodos , _ =matrizProb.shape
    listaProb = []
    #Se obtiene la lista de probabilidades
    for i in range(nNodos):
        listaProb.append(matrizProb[i , 0])
        
    #Se ordena la lista de manera descendiente
    listaProb2 = listaProb.copy()
    listaProb2.sort(reverse=True)
    listaOrdenada = listaProb2
    
    #Se recorre la lista ordenada
    probProximo = 0
    for iProb in range(nNodos):
        #Se establece la condición estocástica
        if listaOrdenada[iProb] > np.random.rand():
            probProximo += listaOrdenada[iProb]
            break
        else:
            pass
    #Si no se escoge niguno de esta manera, se pasa al nodo de mayor probabilidad
    if probProximo == 0:
        probProximo += np.max(listaProb)
    else:
        pass
    #Se obtiene el próximo nodo
    for i in range(nNodos):
        if (matrizProb[i , 0]) == (probProximo):
            próximoNodo = matrizProb[i , 1]
        else:
            pass
    
    return próximoNodo


############################################################
def ObtenerLongitudCamino(camino, coordCiudades):
    """
    Se calcula la longitud del camino correspondiente según las coordenadas de las ciudades
    Retorna la longitud del camino
    """
    #Se obtiene el número de nodos
    nNodos = len(camino)
    #Se inicializa la longitud del camino
    longitudCamino = 0.0
    
    #Se realiza el cálculo de la longitud para el camino utilizando la función de la distancia euclídea
    for pNodo in range(nNodos-1):
        nodoActual = camino[pNodo]
        proxNodo = camino[pNodo+1]
        longTemp = DistanciaEntreCiudades(nodoActual, proxNodo, coordCiudades)
        longitudCamino += longTemp
    
    return longitudCamino


############################################################
def CálculoDeltaTau(colecciónCaminos, colecciónLongitudCaminos):
    '''
    Calcula la matriz deltaTau que permite actualizar el nivel de feromonas
    Parameters
    ----------
    colecciónCaminos : corresponde a los caminos recorridos por cada hormiga
    colecciónLongitudCaminos : corresponde a las longitudes de los caminos recorridos por cada hormiga
    Returns
    -------
    deltaTau : Corresponde a la matriz deltaTau
    '''

    #Se obtiene el número de nodos, que en este caso no se toma el nodo final, pues es el mismo que el incial
    nNodos = len(colecciónCaminos[0])-1
    nCaminos = len(colecciónCaminos)
    
    #Se inicializa la matriz donde se almacenan los valores deltaTau para cada hormiga
    deltaTauTemp = np.zeros((nNodos , nNodos , nCaminos))
    
    #Se inicializa la matriz deltaTau
    deltaTau = np.zeros((nNodos , nNodos))
    
    #Se realiza el cálculo del deltaTau para cada hormiga
    for kHormiga in range(nCaminos):
        for i in range(nNodos-1):
            for j in range(i , nNodos):
                iNodo = colecciónCaminos[kHormiga][i]
                jNodo = colecciónCaminos[kHormiga][j]
                #Se obtiene la longitud del camino
                kLongCamino = colecciónLongitudCaminos[kHormiga]
                deltaTauTemp[int(iNodo), int(jNodo) , int(kHormiga)] = 1./kLongCamino #Se define el valor del deltaTau
    
    #Se almacenan los valores en la matriz deltaTau         
    for k in range(nCaminos):
        deltaTau += deltaTauTemp[:,:,k]

    return deltaTau


##############################################################
def ActualizarNivelFeromonas(nivelFeromonas, deltaNivelFeromonas, rho):
    '''
    Actualiza el arreglo nivel feromonas con los valores del arreglo deltaNivelferomonas
    Retorna el nivel de feromonas actualizado
    Parameters
    ----------
    nivelFeromonas : Corresponde al nivel de feromonas de la iteración en la que se encuentra
    deltaNivelFeromonas : Corresponde a la matriz deltaTau para cambiar el nivel de feromonas
    rho : Corresponde a la tasa de vaporización

    Returns
    -------
    nivelFeromonasActualizado : Corresponde a la matriz del nivel de feromonas actualizada
    '''

    nivelFeromonasActualizado = (1-rho)*nivelFeromonas+deltaNivelFeromonas
    
    return nivelFeromonasActualizado

# Ciclo Principal

#Se obtiene una longitud estimada que solo toma en cuenta las ciudades más cercanas
lEstimada = ObtenerLongitudCaminoVecinosMásCercanos(coordenadasCiudades)

print("La longitud estimada incial es de: " , lEstimada)

# Parámetros inciales del ciclo

# Se define la variable que va a almacenar la longitud mínima encontrada por iteración
longitudMínima = 1e4
longitudMínimaDeseada = 123 #Se establece una condición mínima
tau_0 = nHormigas/lEstimada #Se establece el valor de tau_0 según la longitud estimada

iIteración = 0 #Se inicializa la variable que almacena el número de iteraciones
nivelFeromonas = InicializarNivelFeromonas(nCiudades, tau_0) #Se incializa el nivel de feromonas
visibilidad = ObtenerVisibilidad(coordenadasCiudades) #Se establece la matriz de visibilidad

#Se establece el ciclo while para que pare cuando se alcance la longitud mínima deseada o bien cuando se alcancen 250 iteraciones
while (longitudMínima > longitudMínimaDeseada) and (iIteración < 250):
    iIteración += 1

    # Se generan los caminos que recorren las hormigas
    
    #Se incializan las lista de caminos recorridos y de longitudes de estos caminos
    colecciónCaminos = []
    colecciónLongitudCaminos = []
    
    #Se itera sobre todas las hormigas
    for kHormiga in range(nHormigas):
        # Se obtienen el camino recorrido
        camino = GenerarCamino(nivelFeromonas, visibilidad, alpha, beta)

        # Se calcula la longitud del camino correspondiente
        longitudCamino = ObtenerLongitudCamino(camino, coordenadasCiudades)
        
        #Se imprime la iteración, hormiga y longitud si la longitud obtenida es menor que la longitud mínima establecida
        if longitudCamino < longitudMínima:
            longitudMínima = longitudCamino #Se actualiza el valor de la longitud mínima
            print('Iteración {}, hormiga {}: longitud del camino más corto = {}'.format(iIteración, kHormiga, longitudMínima))
            trayectoMásCorto = camino #Se almacena el trayecto más corto
            
            
        #Se almacenan los caminos
        colecciónCaminos.append(camino)

        #Se almacena la longitud de cada camino
        colecciónLongitudCaminos.append(longitudCamino)
    

    # Fin del ciclo sobre las hormigas

    ###############################################
    # Actualización de los niveles de feromonas

    # Se calcula el cambio del nivel de feromonas
    deltaNivelFeromonas = CálculoDeltaTau(colecciónCaminos, colecciónLongitudCaminos)
    

    # Se actualiza el nivel de feromonas
    nivelFeromonas = ActualizarNivelFeromonas(nivelFeromonas, deltaNivelFeromonas, rho)
    
# Se guarda el camino más corto como un arreglo
arregloTrayectoMásCorto = np.array([ int(x) for x in trayectoMásCorto ])

#Se guarda en el archivo de texto caminoMásCorto_SH
archivoResultado = open("caminoMásCorto_SH.txt", "w")
np.savetxt(archivoResultado, arregloTrayectoMásCorto, fmt ='%1.3i' , newline=',\n')
    
archivoResultado.close()

#Se imprime la longitud del camino más corto
print('La longitud del camino más corto encontrado es de: ', longitudMínima)

#########################Gráfica#########################

ejeX = []
ejeY = []
for iPunto in range (len (trayectoMásCorto)):
    punto = trayectoMásCorto[iPunto]
    puntoX = coordenadasCiudades[int(punto)][0]
    puntoY = coordenadasCiudades[int(punto)][1]
    ejeX.append (puntoX)
    ejeY.append (puntoY)

fig, ax = plt.subplots (dpi = 120)  
ax.set_title ("Trayectoria más corta VA")

ax.set_xlabel ('x (m)')
ax.set_ylabel ('y (m)')
ax.plot (ejeX, ejeY)

plt.plot(ejeX[0::] , ejeY[0::] , marker = 'o' , color = 'red')
plt.plot(ejeX[0] , ejeY[0] , marker = 'o' , color = 'lime')

plt.show()
#########################################################

stop = timeit.default_timer()
# Se imprime el tiempo de ejecución
print('Tiempo de ejecución: ', stop - start , 's')