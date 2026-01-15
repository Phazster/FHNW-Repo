import numpy as np
import sympy as sp


def solve_gauss_sympy(A_liste, b_liste):
    """
    Löst ein lineares Gleichungssystem (LGS) der Form Ax = b mittels Gauss-Jordan-Elimination.

    Diese Funktion wandelt Eingabelisten in SymPy-Matrizen um, erstellt eine 
    augmentierte Matrix (Erweiterte Koeffizientenmatrix) und überführt diese in die 
    reduzierte Zeilenstufenform (RREF).

    Args:
        A_liste (list of lists): Die Koeffizientenmatrix A (n x n). 
            Beispiel: [[2, 1], [1, -3]]
        b_liste (list): Die rechte Seite des Gleichungssystems (Ergebnisvektor b). 
            Beispiel: [5, 7]

    Returns:
        sympy.matrices.dense.MutableDenseMatrix: Die Matrix in reduzierter Zeilenstufenform (RREF).
            - Bei einer eindeutigen Lösung entspricht die letzte Spalte dem Lösungsvektor x.
            - Die Matrix enthält auch die Identitätsmatrix im linken Teil (falls regulär).
        str: Eine Fehlermeldung, falls die Matrix nicht quadratisch ist oder die 
            Dimensionen von A und b nicht übereinstimmen.

    Details:
        1. Dimensionstest: Stellt sicher, dass A quadratisch ist und b die gleiche Zeilenanzahl hat.
        2. Augmentierung: Verknüpft A und b horizontal zur Matrix (A|b).
        3. RREF: Nutzt das Gauss-Jordan-Verfahren. In der resultierenden Matrix kann direkt 
           abgelesen werden:
           - Eindeutige Lösung: Letzte Spalte zeigt die Werte für x1, x2, ... xn.
           - Unendlich viele Lösungen: Zeigt Abhängigkeiten (Nullzeilen möglich).
           - Keine Lösung: Widerspruch in der letzten Zeile (z.B. [0, 0, ..., 0 | 1]).
    """
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
    """
    Berechnet den Winkel zwischen zwei n-dimensionalen Vektoren u und v.

    Die Funktion nutzt das Skalarprodukt und die Normen der Vektoren basierend auf der 
    Formel: cos(theta) = (u · v) / (||u|| * ||v||).

    Args:
        u (list, tuple oder sp.Matrix): Der erste Vektor.
        v (list, tuple oder sp.Matrix): Der zweite Vektor.
        degrees (bool, optional): Wenn True, wird das Ergebnis in Grad (0-180°) 
            zurückgegeben. Wenn False, erfolgt die Ausgabe im Bogenmaß (Radiant, 0-pi). 
            Standard ist True.

    Returns:
        sp.Expr: Der berechnete Winkel als SymPy-Ausdruck. Dies erlaubt eine 
            exakte Darstellung (z.B. pi/2) oder eine numerische Auswertung mit .evalf().

    Raises:
        ValueError: Wenn einer der beiden Vektoren der Nullvektor ist, da die Norm 
            Null eine Division durch Null verursachen würde und der Winkel 
            geometrisch nicht definiert ist.

    Anmerkung:
        In n Dimensionen ist der Winkel immer der "kleinere" eingeschlossene Winkel 
        zwischen den beiden Vektoren im durch sie aufgespannten 2D-Unterraum.
    """
    u = sp.Matrix(u)
    v = sp.Matrix(v)

    if u.norm() == 0 or v.norm() == 0:
        raise ValueError("Winkel mit Nullvektor ist nicht definiert")

    cos_theta = (u.dot(v)) / (u.norm() * v.norm())
    angle = sp.acos(cos_theta)

    return sp.deg(angle) if degrees else angle



def parameter_to_points(p, v):
    """
    Wandelt die Parameterdarstellung einer Geraden in zwei konkrete Punkte um.

    Eine Gerade im n-dimensionalen Raum ist oft durch x = p + t * v definiert. 
    Diese Funktion berechnet den Stützpunkt (t=0) und einen weiteren Punkt (t=1).

    Args:
        p (list, tuple oder sp.Matrix): Der Stützvektor (Aufpunkt) der Geraden.
        v (list, tuple oder sp.Matrix): Der Richtungsvektor der Geraden.

    Returns:
        tuple (sp.Matrix, sp.Matrix): Ein Tupel bestehend aus:
            - Dem Stützpunkt P1 (entspricht dem Eingabevektor p).
            - Einem zweiten Punkt P2, der durch Addition von Richtungsvektor 
              und Stützvektor entsteht (p + v).

    Raises:
        ValueError: Wenn die Dimensionen (Länge) von p und v nicht übereinstimmen, 
            da eine Addition im n-dimensionalen Raum nur bei gleicher Dimension möglich ist.

    Beispiel:
        >>> p = [1, 2], v = [3, 0]
        >>> parameter_to_points(p, v)
        (Matrix([[1], [2]]), Matrix([[4], [2]]))
    """
    p = sp.Matrix(p)
    v = sp.Matrix(v)

    if p.shape != v.shape:
        raise ValueError("p und v müssen gleiche Dimension haben")

    return p, p + v


def points_to_parameter(P1, P2):
    """
    Erzeugt die Parameterdarstellung einer Geraden aus zwei gegebenen Punkten.

    Berechnet den Richtungsvektor v als Differenz zwischen zwei Punkten P1 und P2,
    sodass die Gerade durch g: x = P1 + t * (P2 - P1) beschrieben werden kann.

    Args:
        P1 (list, tuple oder sp.Matrix): Der erste Punkt (wird als Stützvektor verwendet).
        P2 (list, tuple oder sp.Matrix): Der zweite Punkt zur Bestimmung der Richtung.

    Returns:
        tuple (sp.Matrix, sp.Matrix): Ein Tupel bestehend aus:
            - Dem Stützvektor p (entspricht P1).
            - Dem Richtungsvektor v (berechnet aus P2 - P1).

    Raises:
        ValueError: 
            - Wenn P1 und P2 unterschiedliche Dimensionen haben (z.B. R^2 und R^3).
            - Wenn P1 und P2 identisch sind (v = 0), da daraus keine eindeutige 
              Gerade definiert werden kann.

    Beispiel:
        >>> P1 = [1, 2], P2 = [4, 2]
        >>> points_to_parameter(P1, P2)
        (Matrix([[1], [2]]), Matrix([[3], [0]]))
    """
    P1 = sp.Matrix(P1)
    P2 = sp.Matrix(P2)

    if P1.shape != P2.shape:
        raise ValueError("Punkte müssen gleiche Dimension haben")

    v = P2 - P1
    if v.norm() == 0:
        raise ValueError("Punkte müssen verschieden sein")

    return P1, v

def normal_to_coordinate(n, p0):
    """
    Wandelt die Normalenform einer Hyperebene in die Koordinatenform um.

    Die Normalenform n · (x - p0) = 0 wird in die Form n1*x1 + n2*x2 + ... + c = 0 
    überführt. Dabei ist c das Skalarprodukt aus -n und p0.

    Args:
        n (list, tuple oder sp.Matrix): Der Normalenvektor der Ebene (steht senkrecht).
        p0 (list, tuple oder sp.Matrix): Ein Stützpunkt (Ankerpunkt) in der Ebene.

    Returns:
        tuple (sp.Matrix, sp.Expr): Ein Tupel bestehend aus:
            - Dem Normalenvektor n (entspricht den Koeffizienten a, b, c...).
            - Dem berechneten Skalarwert c (Konstante der Ebenengleichung).

    Raises:
        ValueError: Wenn der Normalenvektor n und der Stützpunkt p0 unterschiedliche 
            Dimensionen haben.

    Mathematischer Hintergrund:
        Die Gleichung lautet: n · x + c = 0.
        Durch Einsetzen des Punktes p0 erhält man: n · p0 + c = 0 => c = -(n · p0).
        In 3D ergibt dies: n1*x + n2*y + n3*z + c = 0.

    Beispiel:
        >>> n = [1, 2, 3], p0 = [1, 1, 1]
        >>> normal_to_coordinate(n, p0)
        (Matrix([[1], [2], [3]]), -6) 
        # Resultierende Gleichung: 1x + 2y + 3z - 6 = 0
    """
    n = sp.Matrix(n)
    p0 = sp.Matrix(p0)

    if n.shape != p0.shape:
        raise ValueError("n und p0 müssen gleiche Dimension haben")

    c = -n.dot(p0)
    return n, sp.simplify(c)

def coordinate_to_normal(n, c):
    """
    Wandelt die Koordinatenform einer Hyperebene zurück in die Normalenform.

    Aus der Gleichung n1*x1 + n2*x2 + ... + c = 0 extrahiert die Funktion den 
    Normalenvektor n und berechnet einen möglichen Stützpunkt p0, der die 
    Gleichung erfüllt.

    Args:
        n (list, tuple oder sp.Matrix): Der Normalenvektor (die Koeffizienten 
            vor den Variablen x1, x2, ...).
        c (int, float oder sp.Expr): Die Konstante der Ebenengleichung 
            (der Teil ohne Variablen).

    Returns:
        tuple (sp.Matrix, sp.Matrix): Ein Tupel bestehend aus:
            - Dem Normalenvektor n.
            - Einem berechneten Stützvektor p0, der auf der Ebene liegt.

    Raises:
        ValueError: Wenn der Normalenvektor der Nullvektor ist (n.norm() == 0), 
            da dieser keine Ebene im Raum definieren kann.

    Mathematischer Hintergrund:
        Ein Punkt p0 liegt auf der Ebene, wenn gilt: n · p0 + c = 0.
        Die Funktion findet einen einfachen Punkt, indem sie die erste Variable, 
        deren Koeffizient nicht Null ist, so setzt, dass die Gleichung gelöst wird 
        (alle anderen Variablen werden auf Null gesetzt).

    Beispiel:
        >>> n = [2, -1, 0], c = -4
        >>> coordinate_to_normal(n, c)
        (Matrix([[2], [-1], [0]]), Matrix([[2], [0], [0]]))
        # Probe: 2*2 + (-1)*0 + 0*0 - 4 = 0 (Korrekt)
    """
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
    """
    Erzeugt einen Normalenvektor für eine Gerade im R², die durch zwei Punkte definiert ist.

    Im 2D-Raum kann ein Normalenvektor n direkt aus dem Richtungsvektor v = [vx, vy] 
    abgeleitet werden, indem die Komponenten vertauscht werden und ein Vorzeichen 
    gedreht wird: n = [vy, -vx].

    Args:
        P1 (list, tuple oder sp.Matrix): Der erste Punkt auf der Geraden 
            (dient als Stützpunkt).
        P2 (list, tuple oder sp.Matrix): Der zweite Punkt auf der Geraden.

    Returns:
        tuple (sp.Matrix, sp.Matrix): Ein Tupel bestehend aus:
            - Dem Normalenvektor n, der senkrecht auf der Geraden steht.
            - Dem Punkt P1, der als Stützpunkt für die Normalenform dient.

    Raises:
        ValueError: Wenn die Eingabepunkte nicht zweidimensional sind.

    Mathematischer Hintergrund:
        Das Skalarprodukt des Richtungsvektors v = [vx, vy] und des berechneten 
        Normalenvektors n = [vy, -vx] ist: vx*vy + vy*(-vx) = 0. 
        Damit ist die Orthogonalität im R² garantiert.

    Beispiel:
        >>> P1 = [1, 2], P2 = [4, 6]
        >>> # Richtungsvektor v = [3, 4]
        >>> points_to_normal_2d(P1, P2)
        (Matrix([[4], [-3]]), Matrix([[1], [2]]))
        # Die Gerade lautet dann: 4*(x - 1) - 3*(y - 2) = 0
    """
    P1 = sp.Matrix(P1)
    P2 = sp.Matrix(P2)

    if len(P1) != 2:
        raise ValueError("Nur in R² definiert")

    v = P2 - P1
    n = sp.Matrix([v[1], -v[0]])

    return n, P1

def parameter_to_normal_2d(p, v):
    """
    Wandelt die Parameterform einer Geraden im R² in die Normalenform um.

    Nutzt den Orthogonalitäts-Trick für zwei Dimensionen: Ein Richtungsvektor 
    v = [vx, vy] wird durch Vertauschen der Komponenten und Negieren einer 
    Komponente in einen Normalenvektor n = [vy, -vx] umgewandelt.

    Args:
        p (list, tuple oder sp.Matrix): Der Stützvektor der Geraden.
        v (list, tuple oder sp.Matrix): Der Richtungsvektor der Geraden.

    Returns:
        tuple (sp.Matrix, sp.Matrix): Ein Tupel bestehend aus:
            - Dem Normalenvektor n (senkrecht zu v).
            - Dem Stützvektor p (unverändert übernommen).

    Raises:
        ValueError: Wenn die Eingabevektoren nicht zweidimensional sind.

    Mathematischer Hintergrund:
        Im R² ist der Vektor n = [v_y, -v_x] immer orthogonal zu v = [v_x, v_y], 
        da das Skalarprodukt n · v = v_y*v_x + (-v_x)*v_y = 0 ergibt.
        Die resultierende Normalenform lautet: n · (x - p) = 0.

    Beispiel:
        >>> p = [3, 1], v = [2, 5]
        >>> parameter_to_normal_2d(p, v)
        (Matrix([[5], [-2]]), Matrix([[3], [1]]))
        # Entspricht der Gleichung: 5*(x - 3) - 2*(y - 1) = 0
    """
    p = sp.Matrix(p)
    v = sp.Matrix(v)

    if len(p) != 2:
        raise ValueError("Nur in R² definiert")

    n = sp.Matrix([v[1], -v[0]])
    return n, p



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
    Berechnet die Schnittmenge von mehreren Hyperebenen im n-dimensionalen Raum.

    Die Funktion löst das durch die Ebenen definierte lineare Gleichungssystem Ax = b. 
    Dabei werden verschiedene geometrische Fälle (eindeutiger Punkt, Gerade, Ebene 
    oder leere Menge) unterschieden und die Lösung in Parameterform zurückgegeben.

    Args:
        koeffizienten_liste (list of lists): Die Koeffizienten der Ebenengleichungen.
            Jede innere Liste repräsentiert eine Ebene [a1, a2, ..., an].
        ergebnisse_liste (list): Die Ergebnisse der Gleichungen (rechte Seite / d-Werte).

    Returns:
        str: 
            - "Die Schnittmenge ist leer", falls das System widersprüchlich ist.
            - "Die Ebenen sind identisch", falls das System unterbestimmt ohne Einschränkung ist.
            - "Eindeutiger Schnittpunkt: ...", falls genau eine Lösung existiert.
        dict: Falls die Schnittmenge ein Unterraum (Gerade, Ebene etc.) ist:
            - "Typ": Beschreibung der Dimension (z.B. "Unterraum der Dimension 1" = Gerade).
            - "Stützvektor": sp.Matrix des Aufpunkts.
            - "Richtungsvektoren": Liste von sp.Matrizen, die den Unterraum aufspannen.

    Mathematischer Hintergrund:
        Die Dimension des Lösungsraums entspricht der Anzahl der freien Variablen 
        (n - Rang(A)). Die Funktion nutzt symbolische Differentiation, um die 
        Richtungsvektoren direkt aus der allgemeinen Lösung des LGS zu extrahieren.

    Beispiel (Schnitt zweier Ebenen im R^3 ergibt eine Gerade):
        >>> coeffs = [[1, 1, 1], [1, -1, 0]]
        >>> results = [3, 1]
        >>> schnittmenge_ebenen(coeffs, results)
        { 'Typ': 'Unterraum der Dimension 1', ... }
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
    Analysiert ein lineares Gleichungssystem hinsichtlich seiner Pivot- und freien Variablen.

    Die Funktion berechnet die reduzierte Zeilenstufenform (RREF) der augmentierten Matrix
    und bestimmt, welche Variablen gebunden (Pivots) und welche frei wählbar sind. Zudem
    wird die Lösbarkeit auf Basis des Rangs geprüft.

    Args:
        koeffizienten_matrix (list of lists): Koeffizienten der linken Gleichungsseite (Matrix A).
        konstanten (list): Werte der rechten Gleichungsseite (Vektor b).

    Returns:
        dict: Ein Ergebnis-Dictionary mit folgenden Schlüsseln:
            - "pivot_variablen" (list): Namen der gebundenen Variablen (z.B. ['x1', 'x2']).
            - "freie_variablen" (list): Namen der Variablen, die als freie Parameter dienen.
            - "rref_matrix" (sp.Matrix): Die augmentierte Matrix in reduzierter Stufenform.
            - "loesbar" (bool): False, wenn ein Widerspruch (z.B. 0 = 1) vorliegt.

    Mathematischer Hintergrund:
        Ein LGS ist nur lösbar, wenn der Rang der Koeffizientenmatrix gleich dem Rang 
        der augmentierten Matrix ist. Erscheint ein Pivot in der letzten Spalte der 
        RREF-Matrix, bedeutet dies eine Zeile der Form [0, 0, ..., 0 | 1], was einen 
        Widerspruch darstellt.

    Beispiel:
        >>> A = [[1, 2, 3], [0, 1, 1]]
        >>> b = [5, 2]
        >>> finde_pivot_aus_gleichungen(A, b)
        {
            'pivot_variablen': ['x1', 'x2'],
            'freie_variablen': ['x3'],
            'loesbar': True,
            ...
        }
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
    """"
    Berechnet das Spatprodukt (skalares Tripelprodukt) für drei Vektoren im R³.

    Das Spatprodukt wird berechnet durch das Skalarprodukt des Kreuzprodukts zweier 
    Vektoren mit dem dritten Vektor: (a x b) · c. Es entspricht dem orientierten 
    Volumen des Parallelepipeds (Spats), das von diesen drei Vektoren aufgespannt wird.

    Args:
        a (list, tuple oder sp.Matrix): Der erste Vektor.
        b (list, tuple oder sp.Matrix): Der zweite Vektor.
        c (list, tuple oder sp.Matrix): Der dritte Vektor.

    Returns:
        sp.Expr: Der Skalarwert des Spatprodukts. 
            - Der Betrag entspricht dem Volumen.
            - Das Vorzeichen gibt die Orientierung des Vektorsystems an (Rechtssystem).

    Mathematischer Hintergrund:
        1. Kreuzprodukt (a x b) liefert einen Vektor, dessen Betrag der Fläche des 
           aufgespannten Parallelogramms entspricht.
        2. Das Skalarprodukt mit c projiziert den dritten Vektor auf die Höhe des Spats.
        Alternativ: Das Spatprodukt ist identisch mit der Determinante der Matrix, 
        die aus den Vektoren a, b und c als Spalten gebildet wird.

    Beispiel:
        >>> spatprodukt_sympy([1,0,0], [0,1,0], [0,0,1])
        1  # Einheitswürfel hat Volumen 1
    """
    vec_a = sp.Matrix(a)
    vec_b = sp.Matrix(b)
    vec_c = sp.Matrix(c)
    
    # Kreuzprodukt von a und b, dann Skalarprodukt mit c
    return vec_a.cross(vec_b).dot(vec_c)

def determinante(vektoren):
    """
    Berechnet die Determinante einer Matrix, die aus n Vektoren des R^n gebildet wird.

    In der Geometrie entspricht der Absolutbetrag dieser Determinante dem Volumen 
    des n-dimensionalen Parallelepipeds (Spat), das von diesen Vektoren aufgespannt wird.

    Args:
        vektoren (list of lists oder sp.Matrix): Eine Liste von n Vektoren der Länge n.
            Die Vektoren können entweder als Zeilen oder Spalten der Matrix 
            interpretiert werden (da det(A) = det(A^T)).

    Returns:
        sp.Expr: Der berechnete Determinantenwert.
            - det = 0: Die Vektoren sind linear abhängig (Volumen ist Null).
            - det != 0: Die Vektoren bilden eine Basis des R^n.

    Mathematischer Hintergrund:
        Die Determinante misst den "Skalierungsfaktor" einer linearen Transformation.
        - Im R²: Fläche des durch zwei Vektoren aufgespannten Parallelogramms.
        - Im R³: Volumen des durch drei Vektoren aufgespannten Spats.
        - Im R^n: Das n-dimensionale Hypervolumen.

    Beispiel:
        >>> # Einheitsquadrat im R²
        >>> determinante([[1, 0], [0, 1]])
        1
    """
    # Erstellt eine n x n Matrix aus den n Vektoren
    M = sp.Matrix(vektoren)
    return M.det()

def analysiere_matrix(A_liste, printout = False):
    """
    Führt eine umfassende algebraische Analyse einer Matrix durch.

    Die Funktion untersucht die fundamentalen Eigenschaften einer Matrix: 
    Invertierbarkeit (über die Determinante), den Rang und den Kern (Nullraum). 
    Dies ist essenziell, um die Struktur linearer Abbildungen zu verstehen.

    Args:
        A_liste (list of lists oder sp.Matrix): Die zu analysierende Matrix.
        printout (bool, optional): Wenn True, werden die Ergebnisse formatiert 
            in der Konsole ausgegeben. Standard ist False.

    Returns:
        dict: Ein Ergebnis-Dictionary mit:
            - "status" (str): Klassifizierung (Regulär, Singulär oder Nicht-quadratisch).
            - "determinante" (sp.Expr/None): Der Wert der Determinante (nur bei n x n).
            - "rang" (int): Anzahl der linear unabhängigen Zeilen/Spalten.
            - "dimension" (tuple): (Zeilen n, Spalten m).
            - "ist_singulaer" (bool/None): True, wenn det == 0; None bei nicht-quadratisch.
            - "kern_basis" (list): Eine Liste von Basisvektoren des Nullraums.

    Mathematischer Hintergrund:
        - Eine Matrix ist **regulär**, wenn sie vollen Rang hat und det != 0.
        - Der **Kern (Nullraum)** beschreibt alle Vektoren x, für die Ax = 0 gilt. 
          Die Dimension des Kerns (Defekt) plus der Rang ergibt immer die Spaltenanzahl m.

    Beispiel:
        >>> A = [[1, 2], [2, 4]] # Zeilen sind linear abhängig
        >>> analysiere_matrix(A, printout=True)
        # Output: Status: Singulär, Rang: 1, Kern_Basis: [Matrix([[-2], [1]])]
    """
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

def check_mapping_properties(A_liste):
    """
    Überprüft eine Matrix auf ihre Eigenschaften als lineare Abbildung.

    Die Funktion analysiert, ob die Abbildung f(x) = A*x injektiv, surjektiv 
    oder bijektiv ist. Dies basiert auf dem Vergleich des Rangs der Matrix 
    mit den Dimensionen des Definitionsraums (Spalten) und des Zielraums (Zeilen).

    Args:
        A_liste (list of lists oder sp.Matrix): Die Abbildungsmatrix A.

    Returns:
        dict: Ein Ergebnis-Dictionary mit:
            - "Dimension" (str): Darstellung des Mapping-Raums (R^m -> R^n).
            - "Rang" (int): Anzahl der linear unabhängigen Spalten/Zeilen.
            - "Injektiv (eindeutig)" (bool): True, wenn jeder Bildvektor höchstens 
              ein Urbild hat (Kern ist leer).
            - "Surjektiv (deckend)" (bool): True, wenn der gesamte Zielraum 
              erreicht wird (Bild = Zielraum).
            - "Bijektiv (perfekt)" (bool): True, wenn die Abbildung eine 
              eindeutige 1-zu-1-Entsprechung ist (invertierbar).
            - "Kern_Dimension" (int): Die Dimension des Nullraums (Defekt).

    Mathematischer Hintergrund:
        - Injektiv: Rang = Anzahl der Spalten (n_cols). Keine Information geht verloren.
        - Surjektiv: Rang = Anzahl der Zeilen (n_rows). Der Zielraum wird voll ausgefüllt.
        - Bijektiv: Rang = n_rows = n_cols. Nur bei quadratischen Matrizen möglich.

    Beispiel:
        >>> # Eine Projektion vom R^3 in den R^2
        >>> A = [[1, 0, 0], [0, 1, 0]]
        >>> check_mapping_properties(A)
        # Ergebnis: Injektiv: False (Kern existiert), Surjektiv: True (Rang=2)
    """
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
    """
    Prüft, ob zwei Vektoren orthogonal (senkrecht zueinander) sind.

    Die Funktion berechnet das Skalarprodukt beider Vektoren. Zwei Vektoren sind 
    genau dann orthogonal, wenn ihr Skalarprodukt exakt Null ist. Dabei wird 
    sowohl eine numerische (NumPy) als auch eine symbolische (SymPy) Prüfung 
    durchgeführt, um maximale Genauigkeit zu gewährleisten.

    Args:
        v1 (list, tuple oder np.array): Der erste Vektor.
        v2 (list, tuple oder np.array): Der zweite Vektor.

    Returns:
        dict: Ein Ergebnis-Dictionary mit:
            - "Skalarprodukt" (float/int): Das numerische Ergebnis (NumPy).
            - "Orthogonal" (bool): Das exakte Ergebnis der Prüfung (SymPy). 
              Gibt True zurück, wenn die Vektoren im Winkel von 90° zueinander stehen.

    Mathematischer Hintergrund:
        Zwei Vektoren v1, v2 sind orthogonal <=> v1 · v2 = 0.
        Geometrisch bedeutet dies, dass die Projektion von v1 auf v2 ein Punkt (Nullvektor) ist.

    Beispiel:
        >>> check_orthogonalität([1, 0], [0, 1])
        {'Skalarprodukt': 0, 'Orthogonal': True}
        >>> # Vorteil SymPy: Erkennt Orthogonalität bei Wurzeln exakt
        >>> check_orthogonalität([sp.sqrt(2), sp.sqrt(2)], [1, -1])
        {'Skalarprodukt': 0.0, 'Orthogonal': True}
    """
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



def projektion_und_lot(v_liste, richtung_liste):
    """
    Berechnet die Projektion eines Vektors v auf einen Richtungsvektor u
    sowie das zugehörige Lot (den orthogonalen Anteil).
    
    Args:
        v_liste (list): Der zu projizierende Vektor.
        richtung_liste (list): Der Vektor, auf den projiziert wird.
        
    Returns:
        dict: Enthält die Projektion (v_par), das Lot (v_perp) und die Bestätigung der Orthogonalität.
    """
    v = sp.Matrix(v_liste)
    u = sp.Matrix(richtung_liste)
    
    if u.norm() == 0:
        raise ValueError("Die Projektionsrichtung darf nicht der Nullvektor sein.")
        
    # Formel für Projektion: proj_u(v) = (v · u) / (u · u) * u
    skalar_v_u = v.dot(u)
    skalar_u_u = u.dot(u)
    
    v_projektion = (skalar_v_u / skalar_u_u) * u
    
    # Das Lot (v_ortho) ist die Differenz: v - v_projektion
    v_lot = v - v_projektion
    
    return {
        "projektion": v_projektion,
        "lot": v_lot,
        "ist_orthogonal": v_lot.dot(u) == 0  # Test: Skalarprodukt muss 0 sein
    }


def spiegel_punkt_an_gerade(punkt_P, stuetz_G, richtung_G):
    """
    Spiegelt einen Punkt P an einer Geraden G: x = stuetz_G + t * richtung_G.
    Funktioniert für n Dimensionen (R^2, R^3, R^n).
    
    Args:
        punkt_P (list/Matrix): Der zu spiegelnde Punkt.
        stuetz_G (list/Matrix): Stützvektor der Geraden.
        richtung_G (list/Matrix): Richtungsvektor der Geraden.
        
    Returns:
        sp.Matrix: Der gespiegelte Punkt P'.
    """
    P = sp.Matrix(punkt_P)
    A = sp.Matrix(stuetz_G)
    v = sp.Matrix(richtung_G)
    
    if v.norm() == 0:
        raise ValueError("Der Richtungsvektor der Geraden darf nicht der Nullvektor sein.")
        
    # 1. Hilfsvektor vom Stützpunkt zum Punkt P
    AP = P - A
    
    # 2. Projektion von AP auf den Richtungsvektor v
    # Formel: proj_v(AP) = (AP · v / v · v) * v
    AP_proj = (AP.dot(v) / v.dot(v)) * v
    
    # 3. Lotfußpunkt F auf der Geraden
    F = A + AP_proj
    
    # 4. Spiegelpunkt P' berechnen
    # Formel: P' = P + 2 * (F - P)  =>  P' = 2*F - P
    P_spiegel = 2 * F - P
    
    return P_spiegel

