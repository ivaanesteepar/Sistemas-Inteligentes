from copy import deepcopy
from typing import Tuple, List
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


class TableroLinja:
    
    # Constructor de la clase
    def __init__(self, matriz):
        self.setMatriz(matriz)
        self.fila = len(matriz)
        self.columna = len(matriz[0])
        self.jugador = 1 # Empieza la partida el jugador 1 
        self.pieza = ""
        self.contadorMovimientos = 2
        self.filaControl = 0
        self.turnoExtra = False
    
    
    
    def __eq__(self, other):
        return self.matriz == other.matriz
    
    
    
    # Método que contiene las variables actualizadas al realizar los movimientos
    def initActualizado(self, other):
        self.fila = other.fila
        self.columna = other.columna
        self.jugador = other.jugador
        self.pieza = other.pieza
        self.contadorMovimientos = other.contadorMovimientos
        self.filaControl = other.filaControl
        self.turnoExtra = other.turnoExtra
        
        
        
    # Método que devuelve la matriz actualizada
    def setMatriz(self, matriz):
        self.matriz = deepcopy(matriz)
        self.fila = len(matriz)
        self.columna = len(matriz[0])
        
        
        
    # Método que devuelve una copia de la matriz 
    def getMatriz(self):
        return deepcopy(self.matriz)
        
        
        
    # Método que coloca una pieza en una celda
    def colocarPieza(self, fila, columna):
        if 0 <= fila < len(self.matriz) and 0 <= columna < len(self.matriz[0]): # si la pieza está dentro de los límites de la matriz
            if self.estaVacia(fila, columna): # si la celda de destino está vacía, se coloca la pieza correspondiente
                if self.jugador == 1:
                    self.matriz[fila][columna] = "R"
                elif self.jugador == 2:
                    self.matriz[fila][columna] = "N"
                else:
                    print("Error: Jugador no válido.")
            else:
                print("Error: La celda no está vacía.")
            
            
            
    # Método que borra una pieza de una celda
    def borrarPieza(self, fila, columna):
        if 0 <= fila < len(self.matriz) and 0 <= columna < len(self.matriz[0]): # si la pieza está dentro de los limites de la matriz
            self.matriz[fila][columna] = "V" # se reemplaza la letra por una V (vacío)
        else:
            print("Fila y columna fuera de rango")
            
            
            
    # Método que comprueba si una celda está vacía
    def estaVacia(self,fila,columna):
        return self.matriz[fila][columna] == "V"
        

        
    # Método que calcula la función de coste de cada jugador, además de la diferencia entre ellas
    def utilidad(self, jugador: int) -> int:
        puntosR = 0
        puntosN = 0
        puntos_valores = [5, 3, 2, 1, -1, -2, -3, -5] # valores que tiene cada fila de la matriz (de la 0 a la 7)

        for i in range(len(self.matriz)):
            for j in range(self.columna):
                if self.matriz[i][j] == 'R':
                    puntosR += puntos_valores[i]
                elif self.matriz[i][j] == 'N':
                    puntosN -= puntos_valores[i]

        if jugador == 1:
            return puntosR # Devuelve la puntuación de las piezas R
        elif jugador == 2:
            return puntosN # Devuelve la puntuación de las piezas N
        return puntosR-puntosN
    
    
    
    # Método que imprime el tablero con sus figuras correspondientes
    def imprimirTablero(self):
        matriz_np = np.array(self.matriz)

        # Crear una figura y ejes
        fig, ax = plt.subplots()

        # Configurar el tamaño de la figura para que coincida con el tamaño de la matriz
        fig.set_size_inches(np.array(matriz_np.shape[::-1]) / 2)

        # Iterar sobre la matriz y dibujar círculos o espacios según las letras
        for i in range(matriz_np.shape[0]):
            for j in range(matriz_np.shape[1]):
                # Dibujar líneas para separar las celdas
                ax.add_line(plt.Line2D([j, j], [i, i + 1], color='black'))
                ax.add_line(plt.Line2D([j, j + 1], [i, i], color='black'))

                if matriz_np[i, j] == 'R':
                    circle = plt.Circle((j + 0.5, matriz_np.shape[0] - i - 0.5), 0.4, color='red', fill=True)
                    ax.add_patch(circle)
                elif matriz_np[i, j] == 'N':
                    circle = plt.Circle((j + 0.5, matriz_np.shape[0] - i - 0.5), 0.4, color='black', fill=True)
                    ax.add_patch(circle)
                elif matriz_np[i, j] == 'V':
                    # Imprimir un espacio vacío en lugar de un círculo
                    pass

        # Dibujar líneas para separar las celdas en el borde derecho e inferior
        for i in range(matriz_np.shape[0]):
            ax.add_line(plt.Line2D([matriz_np.shape[1], matriz_np.shape[1]], [i, i + 1], color='black'))
        for j in range(matriz_np.shape[1]):
            ax.add_line(plt.Line2D([j, j + 1], [matriz_np.shape[0], matriz_np.shape[0]], color='blue'))

        # Configurar ejes y mostrar la figura
        ax.set_xlim(0, matriz_np.shape[1])
        ax.set_ylim(0, matriz_np.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axis('off')

        plt.show()
    
    
    
    # Método que cambia el turno del jugador
    def cambiarTurno(self):
        if self.jugador == 1:
            self.jugador = 2
        else:
            self.jugador = 1
            
            
    
    # Método auxiliar que calcula el número de filas que tiene que saltar una pieza
    def numeroMovimientos(self, row: int) -> int:
        cont = 0
        if 0 <= row < len(self.matriz): # si la fila está dentro de la matriz
            for i in range(len(self.matriz[row])): # se recorren las celdas de la fila
                if 0 <= i < len(self.matriz[row]): # si la celda está dentro de la fila
                    if (self.obtenerCelda(row, i) != 'V'): # si hay alguna celda ocupada, se aumenta el contador
                        cont += 1
        
        if row == 0 or row == 7: #Si la fila es alguno de los 2 extremos, en el segundo movimiento solo se moverá una fila
            cont = 1

        return cont
            
    
    
    # Método auxiliar que asigna piezas a los extremos si su fila de destino está fuera del tablero
    def controlDimMatriz(self, row:int):
        if row < 0:
            row = 0
        elif row > 7:
            row = 7
        return row
        
    
    
    # Método que se encarga de calcular la fila de destino y devolver una lista bidimensional con la celda de origen y la celda de destino
    # R se mueve hacia arriba en la matriz
    # N se mueve hacia abajo en la matriz
    def obtenerMovimiento(self, row: int, col: int, jugador: int, numMov:int) -> List[List]:
        lista = []

        if numMov == 1: # si es el primer movimiento, solo se puede avanzar una fila
            if jugador == 1:
                filaNueva = row - 1
            else:
                filaNueva = row + 1
            self.filaControl = self.numeroMovimientos(filaNueva) # se almacena el numero de saltos que tiene que dar la pieza actual

        elif numMov == 2: # si es el segundo movimiento, se avanzará tantas filas como piezas hubiesen en la fila de destino del primer movimiento
            if jugador == 1:
                filaNueva = row - self.filaControl 
            else:
                filaNueva = row + self.filaControl
                
                
        # Si el número de saltos que da la pieza está fuera de los límites del tablero, para no salirse de la matriz al colocar la pieza, se asigna la fila 0 o 7
        if filaNueva < 0 or filaNueva > 7: 
            filaNueva = self.controlDimMatriz(filaNueva)
        
        lista.append([row, col]) # se mete en la lista la fila y columna de origen
        lista.append([filaNueva]) # se mete en la lista la fila de destino ya que la columna la obtengo en otro método

        return lista
        
        
    
    # Método que obtiene la celda (R, N o V)    
    def obtenerCelda(self, row: int, col: int):
        if 0 <= row < len(self.matriz) and 0 <= col < len(self.matriz[0]):
            return self.matriz[row][col]
        else:
            print("La celda está fuera del tablero")
    
    
    
    # Método auxiliar que retorna el movimiento actual (se utiliza en el Minimax)
    def movimientoActual(self):
        if self.contadorMovimientos == 2: # si me quedan 2 movs, estoy en mi 1er mov
            return 1
        else: # si me queda 1 mov, estoy en mi 2ndo mov
            return 2
    
    
    
    # Método que, al realizar un movimiento, comprueba que es válido (se usa en el Minimax)
    def validarMovimiento(self, row: int, col: int, nRow: int, nCol: int):
        pieza = self.obtenerCelda(row,col) # obtengo la pieza de la celda actual
        
        if pieza == 'R' and self.jugador == 1: # si es el jugador 1 y pieza R es válido
            jugador_valido = True
        elif pieza == 'N' and self.jugador == 2: # si es el jugador 2 y pieza N es válido
            jugador_valido = True
        else:
            jugador_valido = False # si no es alguna de las anteriores, no es válido


        if jugador_valido == True: # si el jugador es válido
            if self.contadorMovimientos == 2: # si el contador de movimientos es 2, estoy en el primer movimiento
                numMov = 1
            else:
                numMov = 2

            jugada = self.obtenerMovimiento(row, col, self.jugador, numMov) # se obtiene el movimiento

            if len(jugada) != 2: # Si la lista jugada no contiene una de sus sublistas, se sale de la función
                print("No se encontró una jugada válida para el movimiento inicial.")
                return

            filaDest = jugada[1][0] # se obtiene la fila de destino
            
            if filaDest == nRow: # si la fila de destino es igual a la introducida por pantalla
                if 0 <= filaDest < len(self.matriz) and 0 <= nCol < len(self.matriz[0]): # si la pieza está en los limites del tablero
                    if self.obtenerCelda(row,col) == pieza: # si la celda de origen (R o N) es igual al color correspondiente del jugador actual (R o N)
                        if self.obtenerCelda(row, col) != 'V': # si la celda de origen no esta vacía
                            if self.obtenerCelda(filaDest, nCol) == 'V': # si la celda de destino está vacía
                                return True
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False

            
    
    # Método que se encarga de realizar el movimiento
    def hacerMovimiento(self, row: int, col: int, nRow: int, nCol: int):
        pieza = self.obtenerCelda(row,col) # obtengo la pieza de la celda actual
        
        if pieza == 'R' and self.jugador == 1: # si es el jugador 1 y pieza R es válido
            jugador_valido = True
        elif pieza == 'N' and self.jugador == 2: # si es el jugador 2 y pieza N es válido
            jugador_valido = True
        else:
            jugador_valido = False # si no es alguna de las anteriores, no es válido


        if jugador_valido == True: # si el jugador es válido
            if self.contadorMovimientos == 2:  # si el contador es de 2 movimientos, estoy en el primer movimiento
                numMov = 1
            else:
                numMov = 2

            jugada = self.obtenerMovimiento(row, col, self.jugador, numMov) # se obtiene el movimiento

            if len(jugada) != 2: # Si la lista jugada no contiene una de sus sublistas, se sale de la función
                print("No se encontró una jugada válida para el movimiento inicial.")
                return

            filaDest = jugada[1][0] # se obtiene la fila de destino
            
            if filaDest == nRow: # si la fila de destino es igual a la introducida por pantalla
                if 0 <= filaDest < len(self.matriz) and 0 <= nCol < len(self.matriz[0]): # si la pieza está en los limites del tablero
                    if self.obtenerCelda(row,col) == pieza: # si la celda de origen (R o N) es igual al color correspondiente del jugador actual (R o N)
                        if self.obtenerCelda(row, col) != 'V': # si la celda de origen no esta vacía
                            if self.obtenerCelda(filaDest, nCol) == 'V': # si la celda de destino está vacía
                                
                                # se realiza el movimiento y se comprueba si hay turno extra
                                self.borrarPieza(row, col)
                                self.contadorMovimientos -= 1 # Se disminuye el contador antes de comprobar si hay turno extra
                                self.realizarTurnoExtra(filaDest)
                                self.colocarPieza(filaDest, nCol)                                

                                # se cambia de turno
                                if self.contadorMovimientos == 0:
                                    self.contadorMovimientos = 2
                                    if self.turnoExtra == False: # si no hay turno extra, se cambia de jugador
                                        self.cambiarTurno()
                            else:
                                return print("La celda de destino no está vacía.")
                        else:
                            return print("La celda escogida no tiene ninguna pieza.")
                else:
                    return print("La celda de destino está fuera del tablero.")
            else:
                return print("La fila de destino es incorrecta.")
        else:
            return print("Error: La pieza seleccionada no coincide con el jugador actual.")

    
    
    # Método que comprueba si hay un turno extra
    # Si la fila a la que accede la pieza en el primer movimiento está vacía, el jugador finalizará inmediatamente su turno sin realizar el movimiento secundario
    # Si la fila a la que accede la pieza en el segundo movimiento está vacía, el jugador ganará un turno
    def realizarTurnoExtra(self, filaDestino: int):
        if self.contadorMovimientos == 0 and self.turnoExtra:  # si estoy en mi 2º movimiento y tengo un turno extra
            self.turnoExtra = False  # se reinicia la flag del turno extra
        else:
            if self.contadorMovimientos == 1 and all(celda == 'V' for celda in self.matriz[filaDestino]):  # si estoy en mi 1º movimiento y me muevo a una fila vacía
                self.contadorMovimientos = 0  # se pierde el turno
                self.turnoExtra = False
            elif self.contadorMovimientos == 0 and all(celda == 'V' for celda in self.matriz[filaDestino]):  # si estoy en mi 2º movimiento y me muevo a una fila vacía
                if not self.turnoExtra:  # si no tenía un turno extra
                    self.turnoExtra = True
                    self.contadorMovimientos = 2  # se gana un turno
                else:
                    self.turnoExtra = False  # si ya tuvo un turno extra, no puede tener otro más
    
    
    
    # Método que valida y devuelve la columna de destino
    def validarColumna(self, filaDestino: int) -> int:
        # Verifica la primera columna donde se encuentra una celda vacía ('V') en la fila de destino
        for fila in range(filaDestino, len(self.matriz)):
            for col in range(len(self.matriz[fila])):
                if self.matriz[fila][col] == 'V':
                    return col
        
    
    
    # Método que comprueba que la partida ha sido finalizada
    def partidaAcabada(self) -> bool:
        # Contar las piezas N en la mitad inferior del tablero (codigo adaptado a pyplot)
        count_N_inferior = sum(np.count_nonzero(self.matriz[i] == 'N') for i in range(self.fila // 2, self.fila))

        # Contar las piezas R en la mitad superior del tablero (codigo adaptado a pyplot)
        count_R_superior = sum(np.count_nonzero(self.matriz[i] == 'R') for i in range(self.fila // 2))

        # Verificar que haya exactamente 12 piezas N en la mitad inferior y 12 piezas R en la mitad superior
        if count_N_inferior == 12 and count_R_superior == 12:
            print("¡La partida ha acabado!")
            self.ganadorPartida()
            return True
        else:
            return False
    
    
    
    # Método que devuelve el ganador de la partida
    def ganadorPartida(self):
        utilidad_R = self.utilidad(1)
        utilidad_N = self.utilidad(2)
        
        print(f"Puntuación de las piezas rojas: {utilidad_R}") # Muestra la puntuación del jugador 1
        print(f"Puntuación de las piezas negras: {utilidad_N}") # Muestra la puntuación del jugador 2

        if utilidad_R > utilidad_N:
            print ("¡Jugador 1 (R) es el ganador!")
        elif utilidad_R < utilidad_N:
            print ("¡Jugador 2 (N) es el ganador!")
        else:
            print ("¡Es un empate!")
    

def minimax(state: TableroLinja, currentLevel: int, maxLevel: int, jugador: int, alpha: int, beta: int, stop: bool) -> Tuple[TableroLinja, int, bool]:
    maxHijos = 6 # se define el numero de hijos
    primerMovimiento = []
    segundoMovimiento = []
    stop = False
    stopDigging = False

    if currentLevel == maxLevel:
        stop = True
        stopDigging = True
        coste = state.utilidad(jugador)
        return state, coste, stop

    utility = state.utilidad(jugador) # llamada a la función de coste para calcular el coste de cada hijo

    # Simulación del primer movimiento (se saca el primer movimiento de los hijos)
    for row in range(8):
        for col in range(6):
            # Se sacarán hijos hasta que la longitud de la lista llegue al numero definido
            if len(primerMovimiento) == maxHijos:
                break # se sale de los bucles for si ya se han sacado todos los primeros movimientos de los hijos
                
            successorBoard = TableroLinja(state.getMatriz()) # se crea una instancia de la matriz actual
            successorBoard.initActualizado(state) # se actualiza el estado de la matriz
            
            # Se obtiene una lista con la celda de origen y la fila de destino
            jugada = successorBoard.obtenerMovimiento(row, col, successorBoard.jugador, successorBoard.movimientoActual())
            
            if len(jugada) != 2: # comprobamos que la fila de destino no esté ocupada
                continue # si lo está, pasamos a la siguiente iteración
            
            nRow = jugada[1][0] # se obtiene la fila de destino
            nCol = successorBoard.validarColumna(nRow) # se obtiene la columna de destino
            
            # Si no encuentra ninguna celda vacía en la fila de destino, no almacena el hijo en la lista
            if successorBoard.validarMovimiento(row, col, nRow, nCol):
                successorBoard.hacerMovimiento(row, col, nRow, nCol)
                primerMovimiento.append(successorBoard)
                

    # Simulación del segundo movimiento (se saca el segundo movimiento de los hijos)
    for i in range(len(primerMovimiento)): # se escogen los hijos almacenados en la lista del primer movimiento
        for row in range(8):
            for col in range(6):
                # Se sacarán hijos hasta que la longitud de la lista llegue al numero definido
                if len(segundoMovimiento) == maxHijos:
                    break # se sale de los bucles for si ya se han sacado todos los segundos movimientos de los hijos
                    
                successorBoard2 = TableroLinja(primerMovimiento[i].getMatriz()) # se crea una instancia de la matriz a partir de los hijos del primer movimiento
                successorBoard2.initActualizado(primerMovimiento[i]) # se actualiza el estado de la matriz

                # Se obtiene una lista con la celda de origen y la fila de destino
                jugada2 = successorBoard2.obtenerMovimiento(row, col, successorBoard2.jugador, successorBoard2.movimientoActual())
                
                if len(jugada) != 2: # comprobamos que la fila de destino no esté ocupada
                    continue # si lo está, pasamos a la siguiente iteración
                
                nRow2 = jugada2[1][0] # se obtiene la fila de destino
                nCol2 = successorBoard2.validarColumna(nRow2) # se obtiene la columna de destino

                # Si no encuentra ninguna celda vacía en la fila de destino, no almacena el hijo en la lista
                if successorBoard2.validarMovimiento(row, col, nRow2, nCol2):
                    successorBoard2.hacerMovimiento(row, col, nRow2, nCol2)
                    segundoMovimiento.append(successorBoard2)

    # Si no se ha podido sacar ningun hijo, se termina el método               
    if len(segundoMovimiento) == 0:
        stopDigging = True
        stop = True
        coste = state.utilidad(jugador)
        return state, coste, stop

    bestMatrix = None

    if jugador == 2: # se simula el movimiento del jugador automático
        maxValue = -math.inf

        for i in range(0, len(segundoMovimiento)):
            mat = TableroLinja(segundoMovimiento[i].getMatriz()) # se genera un sucesor de la matriz con los hijos del segundo movimiento
            mat.initActualizado(segundoMovimiento[i]) # se actualiza el estado de la matriz
            matrizS, utility, stop = minimax(mat, currentLevel + 1, maxLevel, 1, alpha, beta, stop) # llamada recursiva

            best = utility

            if best > maxValue: # se irá almacenando en la variable maxValue el hijo con la mayor función de coste
                maxValue = best
                bestMatrix = TableroLinja(segundoMovimiento[i].getMatriz())
                bestMatrix.initActualizado(segundoMovimiento[i])

            alpha = max(alpha, best)
            if best >= beta: # si hay una poda beta, se sale del método
                return matrizS, best, stop

    else: # se simula el movimiento del jugador físico
        minValue = math.inf

        for i in range(0, len(segundoMovimiento)):
            mat = TableroLinja(segundoMovimiento[i].getMatriz())
            mat.initActualizado(segundoMovimiento[i])
            matrizS, utility, stop = minimax(mat, currentLevel + 1, maxLevel, 2, alpha, beta, stop) # llamada recursiva

            if utility < minValue: # se irá almacenando en la variable minValue el hijo con la menor función de coste
                minValue = utility
                bestMatrix = TableroLinja(segundoMovimiento[i].getMatriz())
                bestMatrix.initActualizado(segundoMovimiento[i])

            beta = min(beta, utility)
            if utility <= alpha: # si hay una poda alpha, se sale del metodo
                return matrizS, utility, stop

    return bestMatrix, maxValue if jugador == 2 else minValue, stop
  


# Este método realiza una búsqueda en el árbol de juego utilizando el algoritmo Minimax para seleccionar la mejor matriz de juego
def performActionMinMax(state:TableroLinja, jugador:int):
    depth = 2
    tmp = TableroLinja(state.getMatriz())
    tmp.initActualizado(state)
    stop = False
    currentLevel = 0
    itera = 0

    while not stop and itera <= depth:
        tmpMatrizB = TableroLinja(tmp.getMatriz())  # se crea una instancia de la matriz
        tmpMatrizB.initActualizado(tmp) # se obtiene la matriz actualizada
        (matrizoptima, valoroptimo, stop) = minimax(tmpMatrizB, currentLevel, depth, jugador, -math.inf, math.inf, stop)
        if itera == 0 and stop:
            state.cambiarTurno()
        itera += 1

    return matrizoptima



# Este obtiene y devuelve el mejor movimiento encontrado por Minimax
def AIAction(state:TableroLinja, jugador:int):
    
    global AIReadyToMove

    matriz = TableroLinja(state.getMatriz())
    matriz.initActualizado(state)
    
    # Llamada a performActionMinMax para obtener la mejor acción
    mejor_movimiento = performActionMinMax(matriz, jugador)
    
    AIReadyToMove = False
    
    # Devuelve el mejor movimiento encontrado por la IA
    return mejor_movimiento


matriz = np.array([["N", "N", "N", "N", "N", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "V", "V", "V", "V", "N"],
          ["R", "R", "R", "R", "R", "R"]])



def obtenerEntradaUsuario():
    while True:
        try:
            entrada = input("Introduce tu jugada (fila, columna, nuevaFila, nuevaColumna): ")
            valores = list(map(int, entrada.split()))
            if len(valores) == 4:
                return valores
            else:
                print("Error: Ingresa exactamente cuatro valores separados por un espacio")
        except ValueError:
            print("Error: Ingresa valores numéricos")
            

# Inicialización del tablero
tab = TableroLinja(matriz)

while True:
    print("1. Jugar")
    print("2. Salir")
    
    try:
        opcion_inicio = int(input("Ingresa tu elección (1 o 2): "))
        if opcion_inicio == 1:
            break  # Sale del bucle y comienza el juego
        elif opcion_inicio == 2:
            print("¡Gracias por jugar! Hasta luego.")
            sys.exit()  # Termina la ejecución del programa
        else:
            print("Error: Ingresa 1 para jugar o 2 para salir.")
    except ValueError:
        print("Error: Ingresa un valor numérico (1 o 2).")

# Se imprime el tablero inicial
print(tab.imprimirTablero())

flag = False

while not flag: # Mientras no se haya terminado la partida, se puede seguir jugando
    
    print("\n######################################")
    print(f"El contador de movimientos es: {tab.contadorMovimientos}")
    print("######################################")
    
    if tab.turnoExtra:
        print("¡Obtuviste un turno extra!")

    # Turno del jugador humano (jugador 1)
    if tab.jugador == 1:
        print("\n👤 Jugador 1")
        valores = obtenerEntradaUsuario()
        fila, col, nFila, nCol = valores
        tab.hacerMovimiento(fila, col, nFila, nCol)

    # Turno del jugador automático (jugador 2)
    else:
        print("\n🤖 Jugador 2 (IA)")
        movimientoAutomatico = AIAction(tab, 2)
        tab.setMatriz(movimientoAutomatico.getMatriz())
        tab.initActualizado(movimientoAutomatico)

    # Imprime el tablero después del movimiento
    print(tab.imprimirTablero())

    # Verifica si la partida ha terminado
    if tab.partidaAcabada():
        flag = True
