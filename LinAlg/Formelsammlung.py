import numpy as np
import sympy as sp



def normal_vector(v):
    """Errechnet den normalen Vektor"""
    v = np.array(v, dtype=float)

    if np.allclose(v, 0):
        raise ValueError("Der Nullvektor hat keine Normalenvektoren.")

    n = np.zeros_like(v)

    # Suche erste Nicht-Null-Komponente
    idx = np.nonzero(v)[0][0]

    # Setze eine beliebige Komponente (1)
    n[idx] = -v[-1]

    # Setze letzte Komponente so, dass v·n = 0
    if idx != len(v) - 1:
        n[-1] = v[idx]
    else:
        # Falls erster Nicht-Null-Eintrag die letzte Komponente war
        n[-1] = 1
        n[idx - 1] = -v[-1]

    return n

def vektorProdukt(a,b):
    """Das Vektorprodukt ist auch gleich die Fläche zwischen den beiden Vektoren"""
    return np.cross(a,b) #Vekorprodukt
    
def skalarProdukt(a,b):
    """Errechnet das Skalar- oder Matrizenprodukt"""
    return a@b #Skalar- & Matrizenprodukt
    
def normVektor(a):
    """Normiert einen Vektor"""
    return np.linalg.norm(a) #Norm

def gauss_elimination_with_ratios(A):
    """
    Führt Gauss-Elimination auf einer Matrix A durch und gibt zurück:
      - U: obere Dreiecksform der Matrix
      - ratios: Liste der verwendeten Verhältnisse (Multiplikatoren)
    
    A: Liste von Listen (Matrix), z.B. erweiterte Matrix eines LGS
    
    ratios ist eine Liste von Tupeln:
      (pivot_zeile, ziel_zeile, faktor)
    was der Operation entspricht:
      Zeile[ziel] := Zeile[ziel] - faktor * Zeile[pivot]
    """
    # Kopie der Matrix, damit das Original nicht verändert wird
    A = [row[:] for row in A]
    n = len(A)       # Anzahl Zeilen
    m = len(A[0])    # Anzahl Spalten

    ratios = []      # hier speichern wir die Verhältnisse

    for k in range(min(n, m)):  # Pivot-Spalte k
        # Pivot-Suche (falls Element A[k][k] = 0 ist)
        pivot_row = k
        while pivot_row < n and abs(A[pivot_row][k]) < 1e-12:
            pivot_row += 1

        # Falls eine ganze Spalte Null ist, weiter zur nächsten Spalte
        if pivot_row == n:
            continue

        # Falls Pivot-Zeile nicht k ist: Zeilen tauschen
        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            # Zeilentausch könnte man auch loggen, wenn gewünscht

        pivot = A[k][k]

        # Elimination unterhalb des Pivots
        for i in range(k + 1, n):
            if abs(A[i][k]) < 1e-12:
                continue  # nichts zu eliminieren

            factor = A[i][k] / pivot  # Verhältnis m_ik
            ratios.append((k, i, factor))

            # Zeile i := Zeile i - factor * Zeile k
            for j in range(k, m):
                A[i][j] -= factor * A[k][j]

    print("Obere Dreiecksform U:")
    for row in A:
        print(row)
    
    print("\nVerhältnisse (pivot_zeile, ziel_zeile, faktor):")
    for p, z, f in ratios:
        print(f"Z{z} := Z{z} - ({f}) * Z{p}")
        
    return A, ratios

def linear_dependence_relation(vectors):
    """
    Prüft lineare Abhängigkeit und gibt das Abhängigkeitsverhältnis zurück,
    falls die Vektoren abhängig sind.
    
    vectors: Liste von Listen oder NumPy-Arrays
    
    Rückgabe:
        - dependent (bool)
        - relation (Liste von Koeffizienten c_i, falls abhängig)
    """
    # Matrix M (Spalten = Vektoren)
    M = sp.Matrix(vectors).T
    
    # Nullraum bestimmen
    nullspace = M.nullspace()

    if len(nullspace) == 0:
        # Keine Abhängigkeit
        n = M.shape[1]
        return False, [0]*n
    
    # Erste Basislösung aus dem Nullraum entnehmen
    rel = nullspace[0]
    rel = [sp.simplify(c) for c in rel]

    return True, rel

import math

def distance_point_plane(point, plane):
    """
    Berechnet den Abstand eines Punktes von einer Ebene.

    point: (x, y, z)
    plane: (a, b, c, d) für ax + by + cz + d = 0

    Rückgabe: Abstand als float
    """
    x, y, z = point
    a, b, c, d = plane

    numerator = abs(a*x + b*y + c*z + d)
    denominator = math.sqrt(a*a + b*b + c*c)

    if denominator == 0:
        raise ValueError("Ungültigen Ebenen-Normalenvektor (a,b,c) = (0,0,0)")

    return numerator / denominator


def plane_from_param(p0, v1, v2):
    """
    Bestimmt aus der Parameterform einer Ebene:
        x = p0 + s * v1 + t * v2
    die Koordinatenform und die Hesse-Normalform.

    Eingaben:
        p0 : iterable der Länge 3 (Stützvektor)
        v1 : iterable der Länge 3 (Richtungsvektor 1)
        v2 : iterable der Länge 3 (Richtungsvektor 2)

    Rückgabe:
        coord_form: (a, b, c, d) für ax + by + cz + d = 0
        hesse_form: (n0, d_h), wobei
                    n0 = normierter Normalenvektor (np.array der Länge 3)
                    d_h = Abstand der Ebene vom Ursprung (mit Vorzeichen)
    """
    p0 = np.array(p0, dtype=float)
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # Normalenvektor mittels Kreuzprodukt
    n = np.cross(v1, v2)

    if np.allclose(n, 0):
        raise ValueError("Die Richtungsvektoren v1 und v2 sind linear abhängig – keine eindeutige Ebene.")

    a, b, c = n

    # d so bestimmen, dass die Ebene durch p0 geht:
    # n · x - n · p0 = 0  ⇒ ax + by + cz + d = 0 mit d = -n·p0
    d = -np.dot(n, p0)

    coord_form = (a, b, c, d)

    # Hesse-Normalform: n0 · x - d_h = 0 mit ||n0|| = 1
    norm_n = np.linalg.norm(n)
    n0 = n / norm_n
    d_h = np.dot(n0, p0)  # = (n·p0)/||n||

    hesse_form = (n0, d_h)

    return coord_form, hesse_form


def hesse_und_koordinatenform_punkte(A, B):
    """
    Erzeugt:
      - Koordinatenform
      - Hesse-Normalform 
      - Normalenvektor + Betrag
      - Abstand der Ebene zum Ursprung

    Ebene gegeben durch:
        (X - A) · B = 0
    """

    A = np.asarray(A, float)
    B = np.asarray(B, float)

    # Normalenvektor
    n = B
    betrag_n = np.linalg.norm(n)

    # Skalarprodukt n · A
    c = np.dot(n, A)

    # Koordinatenform: n_x x + n_y y + n_z z = c
    koordinatenform = f"{n[0]:.4f}·x + {n[1]:.4f}·y + {n[2]:.4f}·z = {c:.4f}"

    # Hesse-Normalform im gewünschten Format:
    # h = (((x,y,z)-A)·B - c) / |n|
    hesse_form = (
        f"h = [((x, y, z) - ({A[0]}, {A[1]}, {A[2]})) · "
        f"({n[0]}, {n[1]}, {n[2]}) - {c:.4f}] / {betrag_n:.4f}"
    )

    # Abstand zum Ursprung (Betrag bei Hesse-Form)
    abstand_ursprung = abs(c / betrag_n)

    return {
        "Normalenvektor": n,
        "Betrag_norm": betrag_n,
        "Koordinatenform": koordinatenform,
        "Hesse_Normalform": hesse_form,
        "Abstand_Ursprung": abstand_ursprung
    }

def point_in_plane_coord(point, plane, tol=1e-9):
    """
    Prüft, ob ein Punkt in einer Ebene in Koordinatenform liegt:
        ax + by + cz + d = 0
    """
    x, y, z = point
    a, b, c, d = plane
    
    value = a*x + b*y + c*z + d
    return np.isclose(value, 0.0, atol=tol)

def parameterform(A, B, C):
    """
    Erstellt die Parameterform einer Ebene aus drei Punkten A, B, C.
    Gibt Stützpunkt, Richtungsvektoren und vollständige Parametergleichung zurück.
    """

    A = np.asarray(A, float)
    B = np.asarray(B, float)
    C = np.asarray(C, float)

    # Richtungsvektoren
    v1 = B - A
    v2 = C - A

    # Prüfen: Sind die Richtungsvektoren linear abhängig?
    if np.allclose(np.cross(v1, v2), 0):
        return "Die drei Punkte liegen auf einer Geraden – keine eindeutige Ebene möglich."

    # Ausgabe als formatierten Text
    output = "=== Ebene durch drei Punkte ===\n"
    output += f"Punkt A = {A}\n"
    output += f"Punkt B = {B}\n"
    output += f"Punkt C = {C}\n\n"
    output += f"Stützpunkt: A = {A}\n"
    output += f"Richtungsvektor v₁ = B - A = {v1}\n"
    output += f"Richtungsvektor v₂ = C - A = {v2}\n\n"

    # Parameterdarstellung
    output += "Parameterform der Ebene:\n"
    output += f"E:  X = {A} + r·{v1} + s·{v2}\n"
    output += "mit r, s ∈ ℝ\n"
    
    return output

def distance_point_line(P, P0, d):
    """
    Berechnet den Abstand eines Punktes P zu einer Geraden, gegeben durch:
        x = P0 + t * d

    P  : Punkt (iterable Länge 3)
    P0 : Stützpunkt der Geraden (iterable Länge 3)
    d  : Richtungsvektor der Geraden (iterable Länge 3)

    Rückgabe:
        Abstand als float
    """
    P  = np.array(P, float)
    P0 = np.array(P0, float)
    d  = np.array(d, float)

    if np.allclose(d, 0):
        raise ValueError("Der Richtungsvektor d darf nicht der Nullvektor sein.")

    diff = P - P0
    cross = np.cross(diff, d)

    return np.linalg.norm(cross) / np.linalg.norm(d)

def punkt_in_ebene_spat(P, A, v1, v2, tol=1e-8):
    """
    Prüft, ob ein Punkt P in der Ebene X = A + r*v1 + s*v2 liegt
    nach der Bedingung: (v1 × v2) · (P - A) = 0
    """

    P = np.asarray(P, float)
    A = np.asarray(A, float)
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)

    # Normalenvektor über Kreuzprodukt
    d = np.cross(v1, v2)

    # Vektor vom Aufpunkt zum Prüfpunkt
    p = P - A

    # Skalarprodukt
    sprod = np.dot(d, p)

    # Punkt liegt in der Ebene, wenn d·p = 0
    liegt = abs(sprod) < tol

    return liegt, sprod, d, p

def abstand_punkt_ebene(P, A, v1, v2):
    """
    Berechnet den Abstand eines Punktes P von einer Ebene:
    X = A + r*v1 + s*v2

    Vorgegeben:
    n = v1 × v2
    w = P - A
    s = (w·n) / |n|
    Abstand = |s|
    """

    P = np.asarray(P, float)
    A = np.asarray(A, float)
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)

    # Normalenvektor der Ebene
    n = np.cross(v1, v2)

    # Verbindungsvektor P - A
    w = P - A

    # Skalaranteil s
    s = np.dot(w, n) / np.linalg.norm(n)

    # Abstand ist der Betrag von s
    abstand = abs(s)

    return abstand, s, n, w

def ebene_aus_koordinatenform(a, b, c, d):
    """
    Eingabe:
        Koordinatenform ax + by + cz + d = 0

    Ausgabe:
        - Normalenvektor n
        - Abstand der Ebene zum Ursprung
        - Ein Aufpunkt A der Ebene
        - Zwei Richtungsvektoren v1, v2
        - Parameterform: X = A + r*v1 + s*v2
    """

    # 1. Normalenvektor
    n = np.array([a, b, c], dtype=float)

    # 2. Abstand zum Ursprung
    norm_n = np.linalg.norm(n)
    abstand = abs(d) / norm_n

    # 3. Einen Punkt A auf der Ebene finden:
    #    Setze möglichst einfache Variablen 0
    if a != 0:
        A = np.array([-d/a, 0, 0])
    elif b != 0:
        A = np.array([0, -d/b, 0])
    else:
        A = np.array([0, 0, -d/c])

    # 4. Richtungsvektoren der Ebene finden:
    #    Beliebige Vektoren auswählen, die orthogonal zum Normalenvektor sind.
    #    Methode: zwei lineare unabhängige Lösungen von n · v = 0
    
    # erster Richtungsvektor (v1)
    if a != 0 or b != 0:
        v1 = np.array([b, -a, 0])   # garantiert orthogonal zu (a,b,c)
    else:
        v1 = np.array([1, 0, 0])

    # zweiter Richtungsvektor (v2) orthogonal zu n und nicht parallel zu v1
    v2 = np.cross(n, v1)

    return {
        "Normalenvektor": n,
        "Abstand_Ursprung": abstand,
        "Aufpunkt": A,
        "Richtungsvektor1": v1,
        "Richtungsvektor2": v2,
        "Parameterform": f"X = {A} + r·{v1} + s·{v2}"
    }

def nullraum(*gleichungen):
    """
    Berechnet den Nullraum (Kernel) eines linearen Gleichungssystems.
    
    Eingabe:
        gleichungen = beliebig viele Gleichungen, jeweils als Liste der Koeffizienten.
                      z.B. [1, 2, -1, 3] für 1·x1 + 2·x2 - 1·x3 + 3·x4 = 0

    Rückgabe:
        - Nullraum als Basisvektoren
        - RREF
        - Rang
        - Dimension des Nullraums
    """

    # Matrix aus den Gleichungen
    M = sp.Matrix(gleichungen)

    # Nullraum berechnen
    nullraum_basis = M.nullspace()

    return {
        "Matrix": M,
        "RREF": M.rref(),
        "Rang": M.rank(),
        "Nullraum-Dimension": len(nullraum_basis),
        "Nullraum-Basis": nullraum_basis
    }

def lgs_analyse(A, b):
    """
    Analysiert ein lineares Gleichungssystem Ax = b.
    Gibt Pivot-Variablen, freie Variablen, spezielle Lösung, allgemeine Lösung und die Lösungsmenge aus.
    """
    A = sp.Matrix(A)
    b = sp.Matrix(b)
    M = A.row_join(b)

    # RREF
    rref_M, pivots = M.rref()

    n_var = A.shape[1]
    pivot_vars = list(pivots)
    free_vars = [i for i in range(n_var) if i not in pivot_vars]

    # Sympy löst Ax = b → (spezielle Lösung, Parameterlösungen)
    try:
        spezielle_loesung, parameter_vektoren = A.gauss_jordan_solve(b)
        inconsistent = False
    except ValueError:
        # Widerspruch → keine Lösung
        return {
            "RREF": rref_M,
            "Pivot-Variablen": pivot_vars,
            "Freie Variablen": free_vars,
            "Lösungsmenge": "∅ (keine Lösung)",
            "Allgemeine Lösung": "Keine",
        }

    # Lösungsmenge formatieren
    if len(parameter_vektoren) == 0:
        # Eindeutige Lösung
        loesungsmenge = f"L = {{ {spezielle_loesung} }}"
    else:
        # Unendlich viele Lösungen
        param_str = ""
        for i, v in enumerate(parameter_vektoren):
            param_str += f" + λ{i+1}·{v}"
        loesungsmenge = f"L = {{ {spezielle_loesung}{param_str} }}"

    return {
        "RREF": rref_M,
        "Pivot-Variablen": pivot_vars,
        "Freie Variablen": free_vars,
        "Spezielle Lösung": spezielle_loesung,
        "Parameter-Vektoren": parameter_vektoren,
        "Allgemeine Lösung": spezielle_loesung if len(parameter_vektoren)==0 else f"{spezielle_loesung} + span{parameter_vektoren}",
        "Lösungsmenge": loesungsmenge
    }

def lgs_analyse_v2(A, b, vektoren, namen=None):
    """
    Analysiert ein LGS A x = b.
    
    A : 2D-Liste oder np.array (m x n)
    b : Liste oder np.array (m)
    vektoren : Liste von Vektoren (Listen oder np.arrays), die überprüft werden sollen
    namen : optionale Namen für die Vektoren
    
    Gibt ein Dictionary zurück mit:
        - 'homogen': Liste der homogenen Lösungen
        - 'partikulaer': Liste der partikulären Lösungen
        - 'allgemein': Symbolische allgemeine Lösung mit λ
        - 'beweis': A @ (xp + λ*xh) == b
    """
    
    A = np.array(A)
    b = np.array(b)
    
    if namen is None:
        namen = [f"v{i+1}" for i in range(len(vektoren))]
    
    homogen = []
    partikulaer = []
    
    # Prüfen homogen / partikulär
    for v, name in zip(vektoren, namen):
        v_np = np.array(v)
        if np.allclose(A @ v_np, np.zeros(A.shape[0])):
            homogen.append((name, v_np))
        elif np.allclose(A @ v_np, b):
            partikulaer.append((name, v_np))
    
    # Wenn mindestens eine partikuläre und eine homogene Lösung existiert
    if len(partikulaer) == 0 or len(homogen) == 0:
        xp = None
        xh = None
        allgemein = None
        beweis = None
    else:
        xp = sp.Matrix(partikulaer[0][1])
        xh = sp.Matrix(homogen[0][1])
        lam = sp.symbols('lambda')
        allgemein = xp + lam*xh
        beweis = sp.Matrix(A) * allgemein  # A*(xp + λ xh)
    
    return {
        "homogen": homogen,
        "partikulaer": partikulaer,
        "allgemein": allgemein,
        "beweis": beweis
    }

def lgs_loesen_sortiert(A_list, b_list):
    print("DEBUG — Funktion läuft...")

    # Sympy Matrix erstellen
    A = sp.Matrix(A_list)
    b = sp.Matrix(b_list)
    n = A.shape[1]

    # Variablen erzeugen
    vars = sp.symbols(f'x1:{n+1}')

    # Erweiterte Matrix
    M = A.row_join(b)

    # RREF
    R, pivots = M.rref()

    # Pivot- und freie Variablen
    pivot_vars = [vars[i] for i in pivots]
    free_vars = [vars[i] for i in range(n) if i not in pivots]

    # Spezielle Lösung extrahieren
    x_particular = [0]*n
    for row_i, col_i in enumerate(pivots):
        x_particular[col_i] = R[row_i, -1]

    # Nullraum-Basis bestimmen
    N = A.nullspace()

    # allgemeine Lösung aufbauen
    lambdas = sp.symbols(f'λ1:{len(N)+1}')
    general_solution = sp.Matrix(x_particular)
    
    for i, v in enumerate(N):
        general_solution += lambdas[i] * v

    # Lösungsmenge schön darstellen
    if len(N) == 0:
        L = f"L = {{ {sp.Matrix(x_particular)} }}"
    else:
        parts = [f"{sp.Matrix(x_particular)}"]
        for i, v in enumerate(N):
            parts.append(f"+ λ{i+1}·{sp.Matrix(v)}")
        L = "L = { " + " ".join(parts) + " }"

    return {
        "Pivotvariablen": pivot_vars,
        "Freie Variablen": free_vars,
        "Spezielle Lösung": sp.Matrix(x_particular),
        "Richtungsvektoren": N,
        "Allgemeine Lösung": general_solution,
        "Lösungsmenge": L
    }

def schnittmenge_drei_ebenen(A, b):
    """
    Berechnet die Schnittmenge von drei Ebenen im 3D-Raum.
    Ausgabe:
      - Eindeutiger Schnittpunkt
      - Schnittgerade in Parameterform (übersichtlich)
      - oder keine Lösung
    """
    erweiterte_matrix = np.hstack((A, b.reshape(-1,1)))
    print("Erweiterte Matrix [A|b]:\n", erweiterte_matrix)
    
    rang_A = np.linalg.matrix_rank(A)
    rang_erw = np.linalg.matrix_rank(erweiterte_matrix)
    print("Rang der Koeffizientenmatrix A:", rang_A)
    print("Rang der erweiterten Matrix [A|b]:", rang_erw)
    
    if rang_A == rang_erw == 3:
        loesung = np.linalg.solve(A, b)
        print("Typ: Eindeutiger Schnittpunkt")
        print("Lösung:", loesung)
    
    elif rang_A == rang_erw < 3:
        nullraum = null_space(A)
        richtungsvektor = nullraum[:,0]
        
        # Richtungsvektor auf einfache ganze Zahlen skalieren
        # Schritt 1: auf Brüche approximieren
        brueche = [Fraction(x).limit_denominator() for x in richtungsvektor]
        # Schritt 2: kleinsten gemeinsamen Nenner finden
        nennern = [f.denominator for f in brueche]
        lcm_nenner = np.lcm.reduce(nennern)
        richtungsvektor_skaliert = np.array([float(f * lcm_nenner) for f in brueche])
        
        punkt = np.linalg.pinv(A) @ b
        
        # Ausgabe in Parameterform
        print("Typ: Schnittgerade")
        print(f"Geradengleichung: X = {punkt} + t*{richtungsvektor_skaliert}, mit t ∈ ℝ")
    
    else:
        print("Typ: Keine Lösung")

def lgs_allgemein(A, b, parameter=None):
    """
    Allgemeine Funktion zur Lösung eines LGS, auch mit Parameter.
    
    Parameter:
    ----------
    A : sympy.Matrix -> Koeffizientenmatrix
    b : sympy.Matrix -> rechte Seite (kann Parameter enthalten)
    parameter : sympy.Symbol, optional -> Parameter z.B. m
    
    Funktion:
    ----------
    - Prüft, ob LGS konsistent ist
    - Gibt allgemeine Lösung in Parameterform aus
    - Berücksichtigt Parameter m, falls angegeben
    """

    n_vars = A.shape[1]
    vars = sp.symbols('x1:'+str(n_vars+1))

    # Gleichungen erzeugen
    eqs = [sp.Eq(sum(A.row(i)[j]*vars[j] for j in range(n_vars)), b[i]) for i in range(A.rows)]

    # Wenn Parameter angegeben, zuerst versuchen zu erkennen, ob LGS für jedes m lösbar ist
    if parameter is not None:
        # Lösung der ersten n-1 Gleichungen nach n-1 Variablen
        lösung_vorne = sp.solve(eqs[:-1], vars[:-1], dict=True)
        
        if not lösung_vorne:
            print("❌ Konnte die ersten Gleichungen nicht lösen")
            return
        
        # Letzte Gleichung einsetzen
        letzte_eq = eqs[-1].subs(lösung_vorne[0])
        frei_vars = list(letzte_eq.free_symbols - {parameter})
        
        # Fall 1: LGS abhängig von freier Variable → für jedes m konsistent
        if frei_vars:
            print(f"✔ LGS ist konsistent für jedes {parameter}")
            z = frei_vars[0]  # Wir nehmen die freie Variable als Parameter
            # Lösung ausdrücken in Abhängigkeit von z
            lösung_all = sp.solve(eqs, vars, dict=True)
            print("\nAllgemeine Lösung (Parameterform):")
            for sol in lösung_all:
                for v in vars:
                    sp.pprint(sol[v])
            return
        
        # Fall 2: Letzte Gleichung löst Parameter
        para_loesung = sp.solve(letzte_eq, parameter)
        if not para_loesung:
            print(f"❌ Das LGS ist für keinen Wert von {parameter} konsistent")
            return
        param_value = para_loesung[0]
        print(f"✔ LGS ist konsistent für {parameter} = {param_value}")
        # Gesamtlösung
        b_eval = b.subs(parameter, param_value)
        lösung = sp.linsolve((A, b_eval), vars)
        print("\nAllgemeine Lösung (Parameterform):")
        sp.pprint(lösung)
        return

    # Wenn kein Parameter, normales LGS lösen
    lösung = sp.linsolve((A, b), vars)
    rang_A = A.rank()
    rang_Ab = A.row_join(b).rank()
    print("Rang(A) =", rang_A)
    print("Rang(A|b) =", rang_Ab)
    print("\nAllgemeine Lösung (Parameterform):")
    sp.pprint(lösung)

def linearkombination_matrixprodukt(faktoren, vektoren):
    """
    Berechnet eine Linearkombination von Vektoren als Matrixprodukt
    und gibt Matrix A, Faktorvektor l und Ergebnis aus.
    
    Parameter:
    ----------
    faktoren : Liste oder np.array -> Skalarfaktoren [a1, a2, ...]
    vektoren : Liste von Listen oder np.array -> Vektoren [[v11,v12,...], [v21,v22,...], ...]
    
    Ausgabe:
    -------
    Zeigt Matrix A (Vektoren als Spalten), Faktorvektor l, und Ergebnis
    """

    # Matrix A erstellen (Vektoren als Spalten)
    A = np.column_stack(vektoren)  # NxM, jede Spalte = ein Vektor
    l = np.array(faktoren).reshape(-1,1)  # Faktorvektor als Spalte

    # Matrixprodukt
    result = A @ l

    # Ausgabe
    print("Matrix A (Vektoren als Spalten):")
    print(A)
    print("\nFaktorvektor l:")
    print(l)
    print("\nMatrixprodukt A * l = Ergebnisvektor:")
    print(result.flatten())
    
    return result.flatten()

def loesungstyp_vektoren(C, b, vektoren, namen=None):
    """
    Prüft für jeden Vektor, ob er:
    - partikuläre Lösung (Aufpunkt)
    - homogene Lösung (Richtungsvektor)
    - keine Lösung
    """
    if namen is None:
        namen = [f"v{i+1}" for i in range(len(vektoren))]
    
    C = sp.Matrix(C)
    b = sp.Matrix(b)
    
    for i, vec in enumerate(vektoren):
        v = sp.Matrix(vec)
        prod = C*v
        print(f"{namen[i]}:")
        sp.pprint(prod)
        if prod == b:
            print("→ partikuläre Lösung (Aufpunkt)\n")
        elif prod == sp.zeros(*b.shape):
            print("→ homogene Lösung (Richtungsvektor)\n")
        else:
            print("→ keine Lösung\n")

def linearitaet_pruefen(M, v, w, lam):
    """
    Prüft, ob die Abbildung L(x) = M*x linear ist.
    
    Parameter:
    ----------
    M : sympy.Matrix -> Abbildungsmatrix
    v, w : sympy.Matrix -> Testvektoren
    lam : Skalar -> Testwert
    
    Ausgabe:
    -------
    - L(lambda*v) = lambda*L(v) ?
    - L(v + w) = L(v) + L(w) ?
    """
    # Matrixabbildung
    L = lambda x: M * x
    
    # Homogenität prüfen
    lhs_hom = L(lam * v)
    rhs_hom = lam * L(v)
    print("=== Homogenität prüfen ===")
    print("L(lambda * v) =")
    sp.pprint(lhs_hom)
    print("lambda * L(v) =")
    sp.pprint(rhs_hom)
    print("Erfüllt?", lhs_hom == rhs_hom, "\n")
    
    # Additivität prüfen
    lhs_add = L(v + w)
    rhs_add = L(v) + L(w)
    print("=== Additivität prüfen ===")
    print("L(v + w) =")
    sp.pprint(lhs_add)
    print("L(v) + L(w) =")
    sp.pprint(rhs_add)
    print("Erfüllt?", lhs_add == rhs_add, "\n")
    
    # Gesamtergebnis
    if lhs_hom == rhs_hom and lhs_add == rhs_add:
        print("✅ Die Abbildung ist linear.")
        return True
    else:
        print("❌ Die Abbildung ist nicht linear.")
        return False

def linearitaet_abbildung(L):
    """
    Prüft, ob eine Abbildung L: R^2 -> R^2 linear ist und gibt ggf. die
    Matrixdarstellung in der Standardbasis aus.
    
    Parameter:
    ----------
    L : Funktion, die sympy.Matrix([x1, x2]) -> sympy.Matrix([y1, y2]) liefert
    
    Beispiel:
    L = lambda v: sp.Matrix([sp.cos(phi)*v[0], sp.sin(phi)*v[1]])
    """
    # Symbolische Variablen
    x1, x2 = sp.symbols('x1 x2')
    p1, p2 = sp.symbols('p1 p2')
    lam = sp.symbols('lambda')

    v = sp.Matrix([x1, x2])
    w = sp.Matrix([p1, p2])
    
    # Homogenität prüfen: L(lambda*v) == lambda*L(v)
    lhs_hom = L(lam*v)
    rhs_hom = lam*L(v)
    homogen = sp.simplify(lhs_hom - rhs_hom) == sp.zeros(2,1)
    
    # Additivität prüfen: L(v + w) == L(v) + L(w)
    lhs_add = L(v + w)
    rhs_add = L(v) + L(w)
    additiv = sp.simplify(lhs_add - rhs_add) == sp.zeros(2,1)
    
    # Ergebnis
    if homogen and additiv:
        print("✅ Die Abbildung ist linear.")
        # Matrixdarstellung in Standardbasis
        e1 = sp.Matrix([1,0])
        e2 = sp.Matrix([0,1])
        M = sp.Matrix.hstack(L(e1), L(e2))
        print("Matrixdarstellung in Standardbasis:")
        sp.pprint(M)
        return True, M
    else:
        print("❌ Die Abbildung ist nicht linear.")
        return False, None

def linearitaet_und_matrix(L):
    """
    Prüft, ob eine Abbildung L:R^2->R^2 linear ist.
    Falls linear, gibt Matrixdarstellung in Standardbasis aus.
    
    Parameter:
    ----------
    L : Funktion, die sympy.Matrix([x1,x2]) -> sympy.Matrix([y1,y2]) liefert
    
    Ausgabe:
    -------
    - Linearität prüfen (Homogenität & Additivität)
    - Matrixdarstellung in Standardbasis (e1, e2)
    """
    # Symbolische Variablen
    x1, x2 = sp.symbols('x1 x2')
    p1, p2 = sp.symbols('p1 p2')
    lam = sp.symbols('lambda')
    
    v = sp.Matrix([x1, x2])
    w = sp.Matrix([p1, p2])
    
    # Homogenität
    lhs_hom = L(lam*v)
    rhs_hom = lam*L(v)
    
    # Additivität
    lhs_add = L(v + w)
    rhs_add = L(v) + L(w)
    
    print("=== Homogenität prüfen ===")
    sp.pprint(lhs_hom)
    print("== lambda * L(v) ==")
    sp.pprint(rhs_hom)
    print("Erfüllt?", sp.simplify(lhs_hom - rhs_hom) == sp.zeros(2,1), "\n")
    
    print("=== Additivität prüfen ===")
    sp.pprint(lhs_add)
    print("== L(v) + L(w) ==")
    sp.pprint(rhs_add)
    print("Erfüllt?", sp.simplify(lhs_add - rhs_add) == sp.zeros(2,1), "\n")
    
    # Gesamtergebnis
    linear = sp.simplify(lhs_hom - rhs_hom) == sp.zeros(2,1) and sp.simplify(lhs_add - rhs_add) == sp.zeros(2,1)
    
    if linear:
        print("✅ Die Abbildung ist linear.\n")
        # Standardbasis
        e1 = sp.Matrix([1,0])
        e2 = sp.Matrix([0,1])
        print("Standardbasis e1, e2:")
        sp.pprint(e1)
        sp.pprint(e2)
        
        # Matrixdarstellung
        M = sp.Matrix.hstack(L(e1), L(e2))
        print("\nMatrixdarstellung in Standardbasis (M):")
        sp.pprint(M)
        return True, M
    else:
        print("❌ Die Abbildung ist nicht linear.")
        return False, None

def linearitaet_R2(L):
    """
    Prüft, ob eine Abbildung L:R^2->R^2 linear ist und liefert die Matrixdarstellung.
    """
    x, y = sp.symbols('x y')
    lam = sp.symbols('lambda')

    e1 = sp.Matrix([1,0])
    e2 = sp.Matrix([0,1])
    
    # Homogenität: L(lambda*v) = lambda*L(v)
    linear_hom = True
    for v in [e1, e2]:
        lhs = L(lam*v)
        rhs = lam*L(v)
        for i in range(lhs.shape[0]):
            if sp.simplify(lhs[i] - rhs[i]) != 0:
                linear_hom = False
                break
    
    # Additivität: L(v1+v2) = L(v1)+L(v2)
    lhs = L(e1+e2)
    rhs = L(e1)+L(e2)
    linear_add = True
    for i in range(lhs.shape[0]):
        if sp.simplify(lhs[i]-rhs[i]) != 0:
            linear_add = False
            break
    
    linear = linear_hom and linear_add
    
    if not linear:
        print("❌ Die Abbildung ist nicht linear.")
        return False, None
    
    print("✅ Die Abbildung ist linear.")
    
    # Matrix in Standardbasis
    M = sp.Matrix.hstack(L(e1), L(e2))
    print("Matrixdarstellung in Standardbasis:")
    sp.pprint(M)
    
    return True, M

import sympy as sp

def linearitaet_und_matrix(L, basis, name="Abbildung"):
    """
    Prüft, ob eine Abbildung L linear ist und berechnet die Matrixdarstellung
    in einer gegebenen Basis.
    
    L : Funktion
        Lambda-Funktion, die einen Basisvektor/Funktion/Polynom transformiert
    basis : Liste
        Basisvektoren/Funktionen/Polynome als sympy.Matrix oder sympy.Expr
    name : str
        Name der Abbildung für die Ausgabe
    """
    
    n = len(basis)
    lam = sp.symbols('lambda')
    linear = True
    
    # Hilfsfunktion: Ergebnis in Spaltenmatrix n×1 umwandeln
    def to_column(v):
        if isinstance(v, sp.Matrix):
            return v
        elif isinstance(v, sp.Expr):
            # Polynom oder Funktion: Koeffizienten in Basis bestimmen
            coeffs = []
            for b in basis:
                try:
                    coeffs.append(sp.simplify(v.coeff(b)))
                except:
                    # Falls coeff nicht verfügbar, einfach v als 0D-Matrix
                    coeffs.append(sp.simplify(v) if v != 0 else 0)
            return sp.Matrix(coeffs)
        else:
            # Skalar -> 1x1 Matrix
            return sp.Matrix([v])
    
    # Linearität prüfen
    for i in range(n):
        # Homogenität
        lhs = to_column(L(lam*basis[i]))
        rhs = lam*to_column(L(basis[i]))
        if not all(sp.simplify(lhs[k]-rhs[k])==0 for k in range(lhs.shape[0])):
            linear = False
            break
    if linear:
        # Additivität
        for i in range(n):
            for j in range(i+1, n):
                lhs = to_column(L(basis[i]+basis[j]))
                rhs = to_column(L(basis[i])+L(basis[j]))
                if not all(sp.simplify(lhs[k]-rhs[k])==0 for k in range(lhs.shape[0])):
                    linear = False
                    break
    
    if not linear:
        print(f"❌ {name} ist nicht linear.")
        return False, None
    
    print(f"✅ {name} ist linear.\n")
    
    # Matrixdarstellung in Basis
    cols = [to_column(L(v)) for v in basis]
    M = sp.Matrix.hstack(*cols)
    
    print(f"Matrixdarstellung von {name} in der gegebenen Basis:")
    sp.pprint(M)
    
    return True, M

def linearitaet_und_matrix_universal(L, basis, name="Abbildung"):
    n = len(basis)
    lam = sp.symbols('lambda')
    linear = True
    
    def to_column(v):
        if isinstance(v, sp.Matrix):
            return v
        elif isinstance(v, sp.Expr):
            coeffs = []
            for b in basis:
                try:
                    coeffs.append(sp.simplify(v.coeff(b)))
                except:
                    coeffs.append(sp.simplify(v) if v != 0 else 0)
            return sp.Matrix(coeffs)
        else:
            return sp.Matrix([v])
    
    # Linearität prüfen
    for i in range(n):
        lhs = to_column(L(lam*basis[i]))
        rhs = lam*to_column(L(basis[i]))
        if not all(sp.simplify(lhs[k]-rhs[k])==0 for k in range(lhs.shape[0])):
            linear = False
            break
    if linear:
        for i in range(n):
            for j in range(i+1, n):
                lhs = to_column(L(basis[i]+basis[j]))
                rhs = to_column(L(basis[i])+L(basis[j]))
                if not all(sp.simplify(lhs[k]-rhs[k])==0 for k in range(lhs.shape[0])):
                    linear = False
                    break
    if not linear:
        print(f"❌ {name} ist nicht linear.\n")
        return False, None
    
    print(f"✅ {name} ist linear.\n")
    cols = [to_column(L(v)) for v in basis]
    M = sp.Matrix.hstack(*cols)
    print(f"Matrixdarstellung von {name} in der gegebenen Basis:")
    sp.pprint(M)
    print("\n")
    return True, M

# ------------------------
# Funktion für Projektion auf Gerade
# ------------------------
def projektion_auf_gerade(v, a, R=None):
    a = sp.Matrix(a)
    v = sp.Matrix(v)
    if R is None:
        proj = (v.dot(a)/a.dot(a))*a
    else:
        R = sp.Matrix(R)
        proj = R + ((v-R).dot(a)/a.dot(a))*a
    return proj

def pruefe_linearitaet_projektion(a, R=None):
    v1 = sp.Matrix([1,0])
    v2 = sp.Matrix([0,1])
    lam = sp.symbols('lambda')
    
    if R is None:
        homo1 = projektion_auf_gerade(lam*v1, a, R) == lam*projektion_auf_gerade(v1, a, R)
        homo2 = projektion_auf_gerade(lam*v2, a, R) == lam*projektion_auf_gerade(v2, a, R)
        add = projektion_auf_gerade(v1+v2, a, R) == projektion_auf_gerade(v1,a,R)+projektion_auf_gerade(v2,a,R)
    else:
        homo1 = homo2 = add = False
    linear = homo1 and homo2 and add
    return linear
