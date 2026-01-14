import numpy as np
import sympy as sp


def solve_gauss_sympy(A_liste, b_liste):
    # Listen in SymPy-Matrizen umwandeln
    A = sp.Matrix(A_liste)
    b = sp.Matrix(b_liste)
    
    # Prüfen ob Dimensionen passen
    if A.rows != A.cols or A.rows != b.rows:
        return "Nur quadratische Matrizen n x n erlaubt."

    # Augmentierte Matrix (A|b)
    M = A.row_join(b)
    print("Gleichunssytem:")
    sp.pprint(M)
    print("")

    # In reduzierte Stufenform bringen (Gauss-Jordan)
    M_rref, pivots = M.rref()
    
    # Die letzte Spalte der resultierenden Matrix ist der Lösungsvektor
    return M_rref

def angle_between_sympy(u, v, degrees=True):
    u = sp.Matrix(u)
    v = sp.Matrix(v)

    if u.norm() == 0 or v.norm() == 0:
        raise ValueError("Winkel mit Nullvektor ist nicht definiert")

    cos_theta = (u.dot(v)) / (u.norm() * v.norm())
    angle = sp.acos(cos_theta)

    return sp.deg(angle) if degrees else angle



def parameter_to_points(p, v):
    p = sp.Matrix(p)
    v = sp.Matrix(v)

    if p.shape != v.shape:
        raise ValueError("p und v müssen gleiche Dimension haben")

    return p, p + v


def points_to_parameter(P1, P2):
    P1 = sp.Matrix(P1)
    P2 = sp.Matrix(P2)

    if P1.shape != P2.shape:
        raise ValueError("Punkte müssen gleiche Dimension haben")

    v = P2 - P1
    if v.norm() == 0:
        raise ValueError("Punkte müssen verschieden sein")

    return P1, v

def normal_to_coordinate(n, p0):
    n = sp.Matrix(n)
    p0 = sp.Matrix(p0)

    if n.shape != p0.shape:
        raise ValueError("n und p0 müssen gleiche Dimension haben")

    c = -n.dot(p0)
    return n, sp.simplify(c)

def coordinate_to_normal(n, c):
    n = sp.Matrix(n)

    if n.norm() == 0:
        raise ValueError("Ungültiger Normalvektor")

    # Stützpunkt p0 bestimmen
    p0 = sp.zeros(len(n), 1)
    for i in range(len(n)):
        if n[i] != 0:
            p0[i] = -c / n[i]
            break

    return n, p0


def points_to_normal_2d(P1, P2):
    P1 = sp.Matrix(P1)
    P2 = sp.Matrix(P2)

    if len(P1) != 2:
        raise ValueError("Nur in R² definiert")

    v = P2 - P1
    n = sp.Matrix([v[1], -v[0]])

    return n, P1

def parameter_to_normal_2d(p, v):
    p = sp.Matrix(p)
    v = sp.Matrix(v)

    if len(p) != 2:
        raise ValueError("Nur in R² definiert")

    n = sp.Matrix([v[1], -v[0]])
    return n, p

import sympy as sp

def parameter_to_coordinate(stütz, richtungs_vektoren):
    """
    stütz: Liste oder sp.Matrix der Länge n
    richtungs_vektoren: Liste von n-1 Listen/Vektoren der Länge n
    """
    n = len(stütz)
    stütz = sp.Matrix(stütz)
    U = sp.Matrix(richtungs_vektoren) # Matrix der Richtungsvektoren ((n-1) x n)
    
    # 1. Normalenvektor n finden: Löse U * n = 0
    # Wir suchen den Kern (Nullraum) der Richtungsmatrix
    nullraum = U.nullspace()
    if not nullraum:
        return "Die Vektoren spannen keine Hyperebene auf (linear abhängig?)"
    
    n_vektor = nullraum[0]
    
    # 2. Koordinatenform: n_vektor . (X - stütz) = 0
    variablen = sp.symbols(f'x1:{n+1}')
    X = sp.Matrix(variablen)
    
    koordinaten_eq = n_vektor.dot(X - stütz)
    return sp.simplify(koordinaten_eq), " = 0"



def coordinate_to_parameter(koeffizienten, d):
    """
    koeffizienten: Liste [a1, a2, ..., an]
    d: Ergebnis der Gleichung (rechte Seite)
    """
    n = len(koeffizienten)
    vars = sp.symbols(f'x1:{n+1}')
    # Gleichung: a1*x1 + ... + an*xn - d = 0
    eq = sum(c * v for c, v in zip(koeffizienten, vars)) - d
    
    # Lösung nach der ersten Variable mit Koeffizient != 0
    sol = sp.solve(eq, vars, dict=True)[0]
    freie_vars = [v for v in vars if v not in sol]
    
    # Stützvektor (alle freien Variablen = 0)
    stütz = sp.Matrix([sol.get(v, 0) for v in vars])
    
    # Richtungsvektoren
    richtungen = []
    for f_var in freie_vars:
        # Vektor durch Ableitung nach der freien Variable
        r = sp.Matrix([sp.diff(sol.get(v, v), f_var) for v in vars])
        
        # "Verschönern": Brüche entfernen durch Multiplikation mit dem Hauptnenner
        nenner = [sp.denom(val) for val in r]
        lcm_val = sp.lcm(nenner)
        richtungen.append(r * lcm_val)
        
    return stütz, richtungen

def schnittmenge_ebenen(koeffizienten_liste, ergebnisse_liste):
    """
    koeffizienten_liste: Liste von Listen (die 'a'-Werte jeder Ebene)
    ergebnisse_liste: Liste der Ergebnisse (die 'd'-Werte jeder Ebene)
    """
    # Anzahl der Dimensionen bestimmen
    n = len(koeffizienten_liste[0])
    vars = sp.symbols(f'x1:{n+1}')
    
    # Matrizen aufstellen: A*x = b
    A = sp.Matrix(koeffizienten_liste)
    b = sp.Matrix(ergebnisse_liste)
    
    # Das LGS lösen
    loesung = sp.solve_linear_system(A.row_join(b), *vars)
    
    if loesung is None:
        return "Die Schnittmenge ist leer (kein Schnittpunkt)."
    
    if not loesung:
        return "Die Ebenen sind identisch oder der gesamte Raum ist die Lösung."

    # Freie Variablen identifizieren (Parameter für die Schnittmenge)
    freie_vars = [v for v in vars if v not in loesung]
    
    if not freie_vars:
        # Fall 1: Ein eindeutiger Punkt
        punkt = sp.Matrix([loesung[v] for v in vars])
        return f"Eindeutiger Schnittpunkt: {punkt}"
    else:
        # Fall 2: Schnittmenge ist eine Gerade oder Ebene (Parameterform)
        parameter = sp.symbols(f'r1:{len(freie_vars)+1}')
        ersetzung = {f_var: parameter[i] for i, f_var in enumerate(freie_vars)}
        
        stütz = sp.Matrix([loesung.get(v, v).subs({fv: 0 for fv in freie_vars}) for v in vars])
        
        richtungs_vektoren = []
        ausdruck = sp.Matrix([loesung.get(v, v).subs(ersetzung) for v in vars])
        for p in parameter:
            richtungs_vektoren.append(ausdruck.diff(p))
            
        return {
            "Typ": f"Unterraum der Dimension {len(freie_vars)}",
            "Stützvektor": stütz,
            "Richtungsvektoren": richtungs_vektoren
        }


def finde_pivot_aus_gleichungen(koeffizienten_matrix, konstanten):
    """
    koeffizienten_matrix: Liste von Listen (die linken Seiten der Gleichungen)
    konstanten: Liste der Werte auf der rechten Seite
    """
    # Matrix in SymPy Format umwandeln
    A = sp.Matrix(koeffizienten_matrix)
    b = sp.Matrix(konstanten)
    
    # Augmentierte Matrix (A|b) erstellen
    M = A.row_join(b)
    
    # RREF berechnen: liefert (Reduzierte Matrix, Pivot-Spalten-Indizes)
    m_rref, pivot_indices = M.rref()
    
    # Wichtig: Falls die letzte Spalte (die Konstanten) ein Pivot ist, 
    # ist das System unlösbar (Widerspruch). Wir filtern das:
    anzahl_variablen = A.cols
    echte_pivots = [i for i in pivot_indices if i < anzahl_variablen]
    
    # Variablen-Namen zuordnen
    vars = [f"x{i+1}" for i in range(anzahl_variablen)]
    pivot_vars = [vars[i] for i in echte_pivots]
    freie_vars = [vars[i] for i in range(anzahl_variablen) if i not in echte_pivots]
    
    return {
        "pivot_variablen": pivot_vars,
        "freie_variablen": freie_vars,
        "rref_matrix": m_rref,
        "loesbar": not (anzahl_variablen in pivot_indices)
    }

def spatprodukt_sympy(a, b, c):
    """Berechnet das Spatprodukt für 3D Vektoren."""
    vec_a = sp.Matrix(a)
    vec_b = sp.Matrix(b)
    vec_c = sp.Matrix(c)
    
    # Kreuzprodukt von a und b, dann Skalarprodukt mit c
    return vec_a.cross(vec_b).dot(vec_c)

def determinante(vektoren):
    """Verallgemeinertes Spatprodukt (Volumen) im n-dimensionalen Raum."""
    # Erstellt eine n x n Matrix aus den n Vektoren
    M = sp.Matrix(vektoren)
    return M.det()

def analysiere_matrix(A_liste, printout = False):
    A = sp.Matrix(A_liste)
    n = A.rows
    m = A.cols
    
    # 1. Determinante (nur bei quadratischen Matrizen)
    det = None
    if n == m:
        det = A.det()
        status = "Regulär (invertierbar)" if det != 0 else "Singulär (nicht invertierbar)"
    else:
        status = "Nicht quadratisch (keine Determinante definiert)"

    # 2. Rang bestimmen
    rang = A.rank()
    
    # 3. Kern (Nullraum / Homogene Lösung) finden
    # Der Kern enthält die Vektoren, die beschreiben, wie der Raum "kollabiert"
    kern = A.nullspace()
    if printout:
        print(f"Analyse der Matrix {(n,m)}:")
        print(f"Status: {status}")
        print(f"Rang: {rang} von maximal {(n,m)[0]}")
        if determinante is not None:
            print(f"Determinante: {determinante}")
        
        if kern:
            print("\nDie Matrix ist singulär. Basis des Kerns (homogene Lösungen):")
            for v in kern:
                sp.pprint(v)
        else:
            print("\nDie Matrix ist regulär. Nur der Nullvektor löst das homogene System.")
            
    return {
        "status": status,
        "determinante": det,
        "rang": rang,
        "dimension": (n, m),
        "ist_singulaer": det == 0 if n == m else None,
        "kern_basis": kern
    }

import sympy as sp

def check_mapping_properties(A_liste):
    A = sp.Matrix(A_liste)
    n_rows, n_cols = A.shape  # n_rows = Dimension des Zielraums, n_cols = Dimension des Definitionsraums
    rang = A.rank()
    
    # Injektiv: Kern ist leer / Voller Spaltenrang
    # Das bedeutet: Jeder Vektor wird auf ein eindeutiges Ziel abgebildet.
    is_injective = (rang == n_cols)
    
    # Surjektiv: Bild ist der gesamte Zielraum / Voller Zeilenrang
    # Das bedeutet: Jedes Ziel kann erreicht werden.
    is_surjective = (rang == n_rows)
    
    # Bijektiv: Sowohl injektiv als auch surjektiv
    # Nur bei quadratischen Matrizen mit vollem Rang möglich.
    is_bijective = is_injective and is_surjective
    
    return {
        "Dimension": f"R^{n_cols} -> R^{n_rows}",
        "Rang": rang,
        "Injektiv (eindeutig)": is_injective,
        "Surjektiv (deckend)": is_surjective,
        "Bijektiv (perfekt)": is_bijective,
        "Kern_Dimension": n_cols - rang
    }

def check_orthogonalität(v1, v2):
    # Numerisch mit NumPy
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    skalarprodukt = np.dot(vec1, vec2)
    
    # Symbolisch mit SymPy (für exakte Brüche/Wurzeln)
    v1_s = sp.Matrix(v1)
    v2_s = sp.Matrix(v2)
    is_ortho = v1_s.dot(v2_s) == 0
    
    return {
        "Skalarprodukt": skalarprodukt,
        "Orthogonal":is_ortho}


