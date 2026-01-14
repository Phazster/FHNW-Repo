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






