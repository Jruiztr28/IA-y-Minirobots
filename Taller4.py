import math
import operator
import random
import numpy as np
from deap import algorithms, base, creator, gp, tools

# Semilla para que los resultados se puedan repetir. Cámbiala o quítala si
# quieres ver resultados distintos en cada ejecución.
random.seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------
# Tipos de DEAP (se crean UNA sola vez para todo el archivo)
#   - FitnessMax: queremos MAXIMIZAR la aptitud (más alto = mejor).
#   - Individual: un individuo es un árbol de programa (PrimitiveTree).
# ------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


# ==========================================================================
#  EJERCICIO 1: CIRCUITO LÓGICO - DECODIFICADOR DE 7 SEGMENTOS
# ==========================================================================
"""
PROBLEMA
--------
Un display de 7 segmentos muestra los dígitos 0-9. Se nombran los segmentos
con letras:

         aaa
        f   b
        f   b
         ggg
        e   c
        e   c
         ddd

La entrada es el dígito en binario (BCD) con 4 bits: A B C D
(A es el bit más significativo). Por ejemplo, el 5 es A=0, B=1, C=0, D=1.
La salida son 7 valores (a, b, c, d, e, f, g): 1 = segmento encendido.

Hay que diseñar, con PG, la expresión lógica de CADA segmento.
Por eso haremos 7 corridas de PG: una por segmento.

CONJUNTO DE TERMINALES (las hojas del árbol)
--------------------------------------------
    A, B, C, D  -> los 4 bits de entrada (valores 0 o 1)

CONJUNTO DE FUNCIONES (los nodos del árbol)
-------------------------------------------
    AND(x, y)  -> 1 si x e y son 1
    OR(x, y)   -> 1 si x o y es 1
    XOR(x, y)  -> 1 si x e y son distintos
    NOT(x)     -> invierte el bit
    (Con AND, OR y NOT ya se puede construir cualquier circuito lógico.
     XOR se agrega porque ayuda a encontrar soluciones más cortas.)

    Propiedad de CLAUSURA (que pide la guía): todas las funciones reciben
    y devuelven 0 o 1, así que cualquier función encaja con cualquier otra.

FUNCIÓN DE APTITUD
------------------
    aptitud = (número de dígitos en los que el segmento sale correcto)
              - 0.005 * (tamaño del árbol)

    - El primer término va de 0 a 10 (hay 10 dígitos: 0 a 9).
      Aptitud máxima = 10 -> el circuito es perfecto.
    - El segundo término es una "penalización por tamaño" muy pequeña:
      entre dos circuitos igual de correctos gana el más simple
      (menos compuertas). Sin esto los árboles crecen sin control.

    Nota: las combinaciones 10 a 15 (1010 ... 1111) no son dígitos, así que
    son "no importa" (don't care) y no se evalúan. Eso le da libertad a la
    PG para encontrar circuitos más pequeños.
"""

# Tabla de verdad del display (cátodo común). Cada fila = un dígito,
# cada letra de la cadena = segmento a, b, c, d, e, f, g.
NOMBRES_SEGMENTOS = ["a", "b", "c", "d", "e", "f", "g"]
TABLA_7_SEGMENTOS = {
    0: "1111110",
    1: "0110000",
    2: "1101101",
    3: "1111001",
    4: "0110011",
    5: "1011011",
    6: "1011111",
    7: "1110000",
    8: "1111111",
    9: "1111011",
}


def bits_de_digito(digito):
    """Convierte un dígito (0-9) en sus 4 bits (A, B, C, D). Ej: 5 -> (0,1,0,1)."""
    return tuple(int(b) for b in format(digito, "04b"))


# --- Funciones primitivas lógicas (trabajan con enteros 0 y 1) -----------
def f_and(x, y):
    return x & y


def f_or(x, y):
    return x | y


def f_xor(x, y):
    return x ^ y


def f_not(x):
    return 1 - x


def crear_pset_logico():
    """Crea el conjunto de funciones y terminales para el circuito lógico."""
    pset = gp.PrimitiveSet("LOGICA", 4)  # 4 entradas
    pset.renameArguments(ARG0="A", ARG1="B", ARG2="C", ARG3="D")
    pset.addPrimitive(f_and, 2, name="AND")
    pset.addPrimitive(f_or, 2, name="OR")
    pset.addPrimitive(f_xor, 2, name="XOR")
    pset.addPrimitive(f_not, 1, name="NOT")
    return pset


def evaluar_segmento(individuo, indice_segmento, toolbox):
    """
    Aptitud de un circuito para UN segmento.
    Cuenta en cuántos dígitos acierta y resta una pequeña penalización por tamaño.
    """
    circuito = toolbox.compile(expr=individuo)  # árbol -> función de Python
    aciertos = 0
    for digito, salida_correcta in TABLA_7_SEGMENTOS.items():
        A, B, C, D = bits_de_digito(digito)
        esperado = int(salida_correcta[indice_segmento])
        if circuito(A, B, C, D) == esperado:
            aciertos += 1
    return (aciertos - 0.005 * len(individuo),)


def crear_toolbox_logico(pset, indice_segmento):
    """Configura los operadores de DEAP para el problema lógico."""
    tb = base.Toolbox()

    # Cómo se crean los árboles iniciales (mitad "completos", mitad "irregulares")
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)

    # Aptitud
    tb.register("evaluate", evaluar_segmento,
                indice_segmento=indice_segmento, toolbox=tb)

    # Operadores genéticos
    tb.register("select", tools.selTournament, tournsize=3)   # selección por torneo
    tb.register("mate", gp.cxOnePoint)                        # cruce: intercambia subárboles
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)  # mutación: cambia un subárbol

    # Límite de altura: evita que los árboles crezcan demasiado (bloat)
    limite = gp.staticLimit(key=operator.attrgetter("height"), max_value=8)
    tb.decorate("mate", limite)
    tb.decorate("mutate", limite)
    return tb


def evolucionar_segmento(indice_segmento, tamano_poblacion=300,
                         generaciones=60, prob_cruce=0.7, prob_mutacion=0.1,
                         intentos=5):
    """
    Corre la PG para un segmento. Si no consigue un circuito perfecto (10/10),
    reintenta con otra población inicial (hasta 'intentos' veces).
    Devuelve el mejor árbol encontrado.
    """
    pset = crear_pset_logico()
    mejor_global = None

    for intento in range(1, intentos + 1):
        tb = crear_toolbox_logico(pset, indice_segmento)
        poblacion = tb.population(n=tamano_poblacion)
        salon_fama = tools.HallOfFame(1)  # guarda el mejor de todas las generaciones (elitismo)

        algorithms.eaSimple(poblacion, tb, cxpb=prob_cruce, mutpb=prob_mutacion,
                            ngen=generaciones, halloffame=salon_fama, verbose=False)

        mejor = salon_fama[0]
        if mejor_global is None or mejor.fitness.values[0] > mejor_global.fitness.values[0]:
            mejor_global = mejor

        # Aptitud >= 9 significa 10 aciertos (menos la pequeña penalización)
        if mejor_global.fitness.values[0] >= 9.0:
            break
    return mejor_global


def simplificar_logica(individuo):
    """
    Intenta simplificar la expresión con SymPy para que sea más fácil de leer.
    Si SymPy no está instalado, devuelve el árbol tal cual.
    """
    try:
        from sympy import And, Not, Or, Xor, symbols
        from sympy.logic import simplify_logic

        A, B, C, D = symbols("A B C D")
        espacio = {"A": A, "B": B, "C": C, "D": D,
                   "AND": And, "OR": Or, "XOR": Xor, "NOT": Not}
        expresion = eval(str(individuo), {"__builtins__": {}}, espacio)
        return str(simplify_logic(expresion))
    except Exception:
        return str(individuo)


def dibujar_digito_ascii(segmentos):
    """Devuelve 3 líneas de texto que dibujan un dígito de 7 segmentos."""
    a, b, c, d, e, f, g = segmentos
    linea1 = " " + ("_" if a else " ") + " "
    linea2 = ("|" if f else " ") + ("_" if g else " ") + ("|" if b else " ")
    linea3 = ("|" if e else " ") + ("_" if d else " ") + ("|" if c else " ")
    return [linea1, linea2, linea3]


def ejercicio_1():
    print("=" * 70)
    print("EJERCICIO 1: Decodificador de 7 segmentos con Programación Genética")
    print("=" * 70)

    circuitos = []  # aquí guardamos el mejor árbol de cada segmento
    for i, nombre in enumerate(NOMBRES_SEGMENTOS):
        mejor = evolucionar_segmento(i)
        circuitos.append(mejor)
        aciertos = round(mejor.fitness.values[0] + 0.005 * len(mejor))
        print(f"\nSegmento {nombre}: {aciertos}/10 dígitos correctos")
        print(f"  Árbol de PG : {mejor}")
        print(f"  Simplificado: {simplificar_logica(mejor)}")

    # --- Verificación final: usamos los 7 circuitos juntos y dibujamos ------
    pset = crear_pset_logico()
    funciones = [gp.compile(c, pset) for c in circuitos]

    print("\n" + "-" * 70)
    print("VERIFICACIÓN: salida de los 7 circuitos evolucionados para cada dígito")
    print("-" * 70)
    todo_correcto = True
    dibujos = []
    for digito in range(10):
        A, B, C, D = bits_de_digito(digito)
        salida = [int(f(A, B, C, D)) for f in funciones]
        esperado = [int(x) for x in TABLA_7_SEGMENTOS[digito]]
        ok = salida == esperado
        todo_correcto = todo_correcto and ok
        print(f"Dígito {digito} (ABCD={A}{B}{C}{D}) -> abcdefg = "
              f"{''.join(map(str, salida))}   esperado = {TABLA_7_SEGMENTOS[digito]}   "
              f"{'OK' if ok else 'ERROR'}")
        dibujos.append(dibujar_digito_ascii(salida))

    print("\nDibujo de los dígitos generados por los circuitos evolucionados:\n")
    for fila in range(3):
        print("  ".join(d[fila] for d in dibujos))

    if todo_correcto:
        print("\nResultado: la PG encontró un decodificador perfecto para los 10 dígitos.")
    else:
        print("\nResultado: algún segmento no quedó perfecto. "
              "Prueba aumentar generaciones o tamaño de población.")


# ==========================================================================
#  EJERCICIO 4: PROBLEMA PROPIO -> DESCUBRIR LA LEY DEL PÉNDULO
# ==========================================================================
"""
PROBLEMA (inventado por nosotros, relacionado con mecatrónica/física)
---------------------------------------------------------------
Tenemos un péndulo simple en un laboratorio. Medimos el PERIODO T (segundos
que tarda en ir y volver) para péndulos de distinta LONGITUD L (metros).
Las mediciones tienen un poquito de ruido, como pasa con cualquier sensor.

Pregunta: ¿podemos descubrir la fórmula T = f(L) solo con los datos, sin
conocer la física? Esto es REGRESIÓN SIMBÓLICA (sección 4.9 de la guía):
no se conoce la forma de la función, la PG debe encontrar la fórmula y sus
coeficientes.

(La respuesta real de la física es T = 2*pi*sqrt(L/g), con g = 9.81 m/s².
 Es decir, T ≈ 2.006 * sqrt(L). Usaremos esto solo para COMPARAR al final;
 la PG no lo sabe.)

CONJUNTO DE TERMINALES
----------------------
    L               -> la longitud del péndulo (variable de entrada)
    constantes      -> números aleatorios entre 0.1 y 5 (constantes efímeras)

CONJUNTO DE FUNCIONES
---------------------
    add(x, y)  -> suma
    sub(x, y)  -> resta
    mul(x, y)  -> multiplicación
    div(x, y)  -> división "protegida": si el divisor es casi 0, devuelve 1
    raiz(x)    -> raíz cuadrada "protegida": usa el valor absoluto de x
    (Las versiones "protegidas" garantizan la CLAUSURA: ninguna operación
     puede fallar, sin importar los valores que reciba.)

FUNCIÓN DE APTITUD (la misma de la guía, sección 4.9)
-----------------------------------------------------
    aptitud = 1 / (0.1 + suma de |T_real - T_calculado|)

    - Si el error es cero, la aptitud es 10 (el máximo posible).
    - El 0.1 evita dividir por cero.
    - Se le resta una penalización pequeña por tamaño (0.0005 * nodos)
      para preferir fórmulas cortas.
"""

GRAVEDAD = 9.81  # m/s^2 (solo se usa para GENERAR los datos de ejemplo)


# --- Funciones primitivas protegidas -------------------------------------
def suma(x, y):
    return x + y


def resta(x, y):
    return x - y


def producto(x, y):
    return x * y


def division_protegida(x, y):
    """División que no falla: si y es casi 0 devuelve 1."""
    if abs(y) < 1e-6:
        return 1.0
    return x / y


def raiz_protegida(x):
    """Raíz cuadrada que no falla con números negativos."""
    return math.sqrt(abs(x))


def generar_datos_pendulo(n_puntos=50, ruido=0.01):
    """
    Simula las mediciones del laboratorio:
    longitudes al azar entre 0.1 m y 2 m, y el periodo teórico + ruido.
    Devuelve dos listas: longitudes y periodos.
    """
    longitudes = np.random.uniform(0.1, 2.0, n_puntos)
    periodos = 2 * math.pi * np.sqrt(longitudes / GRAVEDAD)
    periodos = periodos + np.random.normal(0, ruido, n_puntos)  # ruido del sensor
    return longitudes, periodos


def crear_pset_pendulo():
    """Crea el conjunto de funciones y terminales para la regresión simbólica."""
    pset = gp.PrimitiveSet("PENDULO", 1)  # 1 entrada: la longitud
    pset.renameArguments(ARG0="L")
    pset.addPrimitive(suma, 2, name="add")
    pset.addPrimitive(resta, 2, name="sub")
    pset.addPrimitive(producto, 2, name="mul")
    pset.addPrimitive(division_protegida, 2, name="div")
    pset.addPrimitive(raiz_protegida, 1, name="raiz")
    # Constante efímera: cada vez que aparece en un árbol toma un valor aleatorio
    pset.addEphemeralConstant("const", lambda: round(random.uniform(0.1, 5.0), 2))
    return pset


def evaluar_pendulo(individuo, toolbox, longitudes, periodos):
    """Aptitud = 1 / (0.1 + suma de errores absolutos), menos penalización por tamaño."""
    formula = toolbox.compile(expr=individuo)
    error_total = 0.0
    for L, T_real in zip(longitudes, periodos):
        try:
            T_calculado = formula(L)
            error_total += abs(T_real - T_calculado)
        except (OverflowError, ValueError, ZeroDivisionError):
            return (0.0,)  # árbol que falla: aptitud mínima
    if not math.isfinite(error_total):  # evita inf o nan
        return (0.0,)
    aptitud = 1.0 / (0.1 + error_total)
    return (aptitud - 0.0005 * len(individuo),)


def ejercicio_4():
    print("\n" + "=" * 70)
    print("EJERCICIO 4: Descubrir la ley del péndulo con Programación Genética")
    print("=" * 70)

    # 1) Datos: 35 puntos para entrenar y 15 para probar (datos que la PG nunca ve)
    L_todos, T_todos = generar_datos_pendulo(n_puntos=50, ruido=0.01)
    L_train, T_train = L_todos[:35], T_todos[:35]
    L_test, T_test = L_todos[35:], T_todos[35:]

    # 2) Configuración de DEAP
    pset = crear_pset_pendulo()
    tb = base.Toolbox()
    tb.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    tb.register("individual", tools.initIterate, creator.Individual, tb.expr)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("compile", gp.compile, pset=pset)
    tb.register("evaluate", evaluar_pendulo, toolbox=tb,
                longitudes=L_train, periodos=T_train)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("mate", gp.cxOnePoint)
    tb.register("expr_mut", gp.genFull, min_=0, max_=2)
    tb.register("mutate", gp.mutUniform, expr=tb.expr_mut, pset=pset)
    limite = gp.staticLimit(key=operator.attrgetter("height"), max_value=8)
    tb.decorate("mate", limite)
    tb.decorate("mutate", limite)

    # 3) Parámetros de la corrida (los mismos valores de la guía en cruce y mutación)
    TAMANO_POBLACION = 300
    GENERACIONES = 80
    PROB_CRUCE = 0.7
    PROB_MUTACION = 0.1

    poblacion = tb.population(n=TAMANO_POBLACION)
    salon_fama = tools.HallOfFame(1)  # elitismo: siempre recordamos al mejor

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
    estadisticas.register("promedio", np.mean)
    estadisticas.register("maximo", np.max)

    # 4) ¡Evolución!
    print("\nEvolucionando... (se muestra el progreso cada 10 generaciones)")
    _, bitacora = algorithms.eaSimple(poblacion, tb, cxpb=PROB_CRUCE,
                                      mutpb=PROB_MUTACION, ngen=GENERACIONES,
                                      stats=estadisticas, halloffame=salon_fama,
                                      verbose=False)
    for registro in bitacora:
        if registro["gen"] % 10 == 0:
            print(f"  generación {registro['gen']:3d} | aptitud promedio = "
                  f"{registro['promedio']:.3f} | mejor aptitud = {registro['maximo']:.3f}")

    # 5) Resultado
    mejor = salon_fama[0]
    formula = gp.compile(mejor, pset)
    print(f"\nMejor árbol encontrado:\n  {mejor}")

    try:
        import sympy
        L_sym = sympy.symbols("L", positive=True)
        espacio = {"L": L_sym, "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
                   "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
                   "raiz": lambda a: sympy.sqrt(sympy.Abs(a))}
        simple = sympy.simplify(eval(str(mejor), {"__builtins__": {}}, espacio))
        print(f"Fórmula simplificada:\n  T(L) = {simple}")
    except Exception:
        pass

    print(f"\nFórmula real de la física: T = 2*pi*sqrt(L/g) ≈ "
          f"{2 * math.pi / math.sqrt(GRAVEDAD):.3f} * sqrt(L)")

    # 6) Qué tan bueno es el modelo (sobre datos de entrenamiento y de prueba)
    def error_cuadratico_medio(Ls, Ts):
        predicciones = np.array([formula(L) for L in Ls])
        return float(np.mean((Ts - predicciones) ** 2))

    def r_cuadrado(Ls, Ts):
        predicciones = np.array([formula(L) for L in Ls])
        ss_res = np.sum((Ts - predicciones) ** 2)
        ss_tot = np.sum((Ts - np.mean(Ts)) ** 2)
        return float(1 - ss_res / ss_tot)

    print(f"\nECM entrenamiento = {error_cuadratico_medio(L_train, T_train):.6f}   "
          f"R² = {r_cuadrado(L_train, T_train):.4f}")
    print(f"ECM prueba        = {error_cuadratico_medio(L_test, T_test):.6f}   "
          f"R² = {r_cuadrado(L_test, T_test):.4f}")
    print("(R² cercano a 1 significa que la fórmula explica muy bien los datos.)")

    # 7) Gráfica: datos con ruido vs. fórmula encontrada vs. fórmula real
    try:
        import matplotlib.pyplot as plt

        L_curva = np.linspace(0.1, 2.0, 200)
        plt.figure(figsize=(7, 4.5))
        plt.scatter(L_train, T_train, label="Datos de entrenamiento", alpha=0.7)
        plt.scatter(L_test, T_test, label="Datos de prueba", marker="x", color="red")
        plt.plot(L_curva, [formula(L) for L in L_curva],
                 label="Fórmula encontrada por PG", linewidth=2)
        plt.plot(L_curva, 2 * np.pi * np.sqrt(L_curva / GRAVEDAD), "--",
                 label="Fórmula real (física)", color="black")
        plt.xlabel("Longitud L (m)")
        plt.ylabel("Periodo T (s)")
        plt.title("Regresión simbólica con PG: ley del péndulo")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("pendulo_pg.png", dpi=120)
        print("\nGráfica guardada como 'pendulo_pg.png'")
        plt.show()
    except Exception as error:
        print(f"(No se pudo graficar: {error})")


# ==========================================================================
#  PROGRAMA PRINCIPAL: ejecuta los dos ejercicios
# ==========================================================================
if __name__ == "__main__":
    ejercicio_1()
    ejercicio_4()