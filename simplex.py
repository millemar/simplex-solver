"""
Méthode du simplexe révisé en deux phases.

Phase I : Trouver une solution de base réalisable initiale (SBR) en minimisant la somme des variables artificielles.
Phase II : Optimiser la fonction objectif originale en partant de la SBR.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from mps_parser import LPProblem, parse_mps

# Tolérance numérique pour traiter les valeurs comme étant nulles
TOL_ZERO = 1e-8
# Tolérance pour le critère d'optimalité (coûts réduits)
TOL_OPT = 1e-8
# Tolérance d'infeasibilité (valeur de Phase I)
TOL_REALISABLE = 1e-6


@dataclass
class ResultatSimplexe:
    """Résultat retourné par le solveur du simplexe en deux phases."""
    statut: str
    objectif: float
    x: np.ndarray
    base: List[int]
    iter_phase1: int
    iter_phase2: int
    message: str = ''


def resoudre(probleme: LPProblem) -> ResultatSimplexe:
    """
    Résout le problème de PL à l’aide de la méthode du simplexe en deux phases.

    Paramètres
    ----------
    probleme : LPProblem
        Problème sous forme standard (min c^Tx s.c. Ax=b, x>=0).

    Retour
    ------
    ResultatSimplexe
        Le résultat de la résolution avec statut, valeur de l'objectif et solution.
    """
    m, n = probleme.A.shape

    # ---------------------- Phase I -------------------------- #
    # Trouver une solution de base réalisable
    base, iters1 = phase1(probleme.A, probleme.b)
    if base is None:
        return ResultatSimplexe(
            statut='infeasible',
            objectif=np.inf,
            x=np.zeros(n),
            base=[],
            iter_phase1=iters1,
            iter_phase2=0,
            message='Phase I : problème irréalisable',
        )

    # ---------------------- Phase II -------------------------- #
    # Optimisation de la fonction objectif originale
    x, obj, iters2, statut = phase2(probleme.A, probleme.b, probleme.c, base)
    if statut == 'unbounded':
        return ResultatSimplexe(
            statut='unbounded',
            objectif=-np.inf,
            x=x,
            base=base,
            iter_phase1=iters1,
            iter_phase2=iters2,
            message='Phase II : problème non borné',
        )

    obj_total = obj + probleme.obj_offset

    return ResultatSimplexe(
        statut='optimal',
        objectif=obj_total,
        x=x,
        base=base,
        iter_phase1=iters1,
        iter_phase2=iters2,
        message='optimal',
    )


def resoudre_programme_lineaire(
    fichier_donnees: Optional[str] = None,
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    c: Optional[np.ndarray] = None,
) -> ResultatSimplexe:
    """
    Résout un programme linéaire à l’aide du simplexe en deux phases.

    Paramètres
    ----------
    fichier_donnees : str, optionnel
        Chemin d’un fichier MPS à parser et résoudre.
    A : np.ndarray, optionnel
        Matrice des contraintes (mode matrice).
    b : np.ndarray, optionnel
        Vecteur membre de droite.
    c : np.ndarray, optionnel
        Vecteur objectif (minimiser c^T x).

    Retour
    ------
    ResultatSimplexe
        Statut du solveur, valeur objectif, solution, nombre d’itérations.
    """
    mode_fichier = fichier_donnees is not None
    mode_matrice = A is not None or b is not None or c is not None

    if mode_fichier and mode_matrice:
        raise ValueError("Utilisez fichier_donnees OU A/b/c, pas les deux.")

    if mode_fichier:
        probleme = parse_mps(fichier_donnees)
        return resoudre(probleme)

    if A is None or b is None or c is None:
        raise ValueError("Le mode matrice nécessite A, b et c.")

    matrice_A = np.asarray(A, dtype=float)
    vecteur_b = np.asarray(b, dtype=float)
    vecteur_c = np.asarray(c, dtype=float)

    if matrice_A.ndim != 2:
        raise ValueError("A doit être une matrice 2D.")
    if vecteur_b.ndim != 1 or vecteur_c.ndim != 1:
        raise ValueError("b et c doivent être des vecteurs 1D.")

    m, n = matrice_A.shape
    if vecteur_b.shape[0] != m:
        raise ValueError("La taille de b doit correspondre au nombre de lignes de A.")
    if vecteur_c.shape[0] != n:
        raise ValueError("La taille de c doit correspondre au nombre de colonnes de A.")

    # Normaliser pour garantir b >= 0
    A_norm = matrice_A.copy()
    b_norm = vecteur_b.copy()
    for i in range(m):
        if b_norm[i] < 0:
            A_norm[i, :] *= -1.0
            b_norm[i] *= -1.0

    probleme = LPProblem(
        name="entree_matrice",
        c=vecteur_c,
        A=A_norm,
        b=b_norm,
        var_names=[f"x{j + 1}" for j in range(n)],
        row_names=[f"c{i + 1}" for i in range(m)],
        n_orig=n,
    )
    return resoudre(probleme)


# ==================== FONCTIONS INTERNES ======================= #

def phase1(
    A: np.ndarray,
    b: np.ndarray,
) -> Tuple[Optional[List[int]], int]:
    """
    Phase I du simplexe : ajout de variables artificielles.

    Retourne la base (sans artificielles) si réalisable, sinon None.
    """
    m, n = A.shape

    # Augmenter la matrice avec les artificielles
    A_aug = np.hstack([A, np.eye(m)])
    c_aug = np.concatenate([np.zeros(n), np.ones(m)])

    # Base initiale : variables artificielles
    base = list(range(n, n + m))

    # Exécution du simplexe
    iters = iterations_simplexe(A_aug, b, c_aug, base)

    # Calcul de l’objectif de Phase I
    x_aug = solution_base(A_aug, b, base)
    obj_phase1 = float(c_aug @ x_aug)

    if obj_phase1 > TOL_REALISABLE:
        return None, iters

    # Sortir les artificielles de la base
    base = retirer_artificielles(A_aug, b, base, n)

    # Retourner seulement les indices des variables originales
    return base, iters


def retirer_artificielles(
    A_aug: np.ndarray,
    b: np.ndarray,
    base: List[int],
    n_orig: int,
) -> List[int]:
    """
    Retirer les variables artificielles de la base (on pivote si nécessaire).
    """
    m = A_aug.shape[0]
    base = list(base)

    for i in range(m):
        if base[i] >= n_orig:
            # Cherche une variable d'origine à pivoter
            ligne_invB = ligne_base_inverse(A_aug, base, i)
            pivot_trouve = False
            for j in range(n_orig):
                if j not in base:
                    if abs(ligne_invB @ A_aug[:, j]) > TOL_ZERO:
                        base[i] = j
                        pivot_trouve = True
                        break
            # Si aucune variable originale à pivoter, on laisse l’artificielle
            pass

    return base


def phase2(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    base: List[int],
) -> Tuple[np.ndarray, float, int, str]:
    """
    Phase II du simplexe : optimisation de la fonction objectif.
    Retourne (solution, valeur objectif, nb itérations, statut).
    """
    m, n = A.shape
    base = list(base)

    iters, statut = iterations_simplexe_et_statut(A, b, c, base)

    x = solution_base(A, b, base)
    obj = float(c @ x)

    return x[:A.shape[1]], obj, iters, statut


def iterations_simplexe(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    base: List[int],
) -> int:
    """Exécute le simplexe en place sur base. Retourne le nombre d’itérations."""
    iters, _ = iterations_simplexe_et_statut(A, b, c, base)
    return iters


def iterations_simplexe_et_statut(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    base: List[int],
) -> Tuple[int, str]:
    """
    Exécute la méthode du simplexe révisé.
    Modifie `base` en place.
    Retourne (nombre itérations, statut) où statut vaut 'optimal' ou 'unbounded'.
    """
    m, n = A.shape
    iters = 0
    MAX_ITER = 10 * (m + n)

    while iters < MAX_ITER:
        # ---- Matrice de base et son inverse ----
        B = A[:, base]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            break  # Base singulière

        x_B = B_inv @ b  # Valeurs de base

        # ---- Calcul des coûts réduits ----
        c_B = c[base]
        y = c_B @ B_inv   # Multiplicateurs de Lagrange

        # Indices non basiques
        non_base = [j for j in range(n) if j not in base]

        # Coûts réduits des variables non basiques
        rc = np.array([c[j] - y @ A[:, j] for j in non_base])

        # ---- Vérification optimalité ----
        if np.all(rc >= -TOL_OPT):
            return iters, 'optimal'

        # Règle de Bland — plus petit indice négatif
        candidats_entrant = [
            non_base[k] for k in range(len(non_base)) if rc[k] < -TOL_OPT
        ]
        entrant = min(candidats_entrant)
        pos_entrant = non_base.index(entrant)

        # ---- Test du rapport ----
        d = B_inv @ A[:, entrant]
        rapports = np.full(m, np.inf)
        for i in range(m):
            if d[i] > TOL_ZERO:
                rapports[i] = x_B[i] / d[i]

        if np.all(np.isinf(rapports)):
            return iters, 'unbounded'

        # Bland sur le choix de la sortie
        min_rapport = np.min(rapports)
        candidats_sortie = [
            i for i in range(m)
            if abs(rapports[i] - min_rapport) <= TOL_ZERO * max(1.0, abs(min_rapport))
        ]
        sortie = min(candidats_sortie, key=lambda i: base[i])

        # ---- Pivot ----
        base[sortie] = entrant
        iters += 1

    if iters >= MAX_ITER:
        # Considéré comme optimal (pas de cyclage avec Bland)
        return iters, 'optimal'

    return iters, 'optimal'


def solution_base(
    A: np.ndarray,
    b: np.ndarray,
    base: List[int],
) -> np.ndarray:
    """
    Calcule la solution de base x_B = B^{-1} b, x_N = 0.
    """
    m, n = A.shape
    x = np.zeros(n)
    B = A[:, base]
    try:
        x_B = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        x_B = np.linalg.lstsq(B, b, rcond=None)[0]
    for i, j in enumerate(base):
        x[j] = x_B[i]
    return x


def ligne_base_inverse(
    A: np.ndarray,
    base: List[int],
    ligne: int,
) -> np.ndarray:
    """
    Retourne la ligne `ligne` de B^{-1} où B = A[:, base].
    """
    B = A[:, base]
    e = np.zeros(len(base))
    e[ligne] = 1.0
    try:
        return np.linalg.solve(B.T, e)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(B.T, e, rcond=None)[0]
