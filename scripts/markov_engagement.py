# markov_engagement.py
"""
Funciones para modelar engagement con cadenas de Markov sobre los datos de Globant.
No hace plots ni escribe archivos: solo lógica reutilizable.
"""

import numpy as np
import pandas as pd

# Estados por defecto (Engagement Group 1..5)
DEFAULT_STATES = [1, 2, 3, 4, 5]


def build_transition_matrix(df: pd.DataFrame,
                            col_state: str = "Engagement Group",
                            states=None) -> np.ndarray:
    """
    Construye una matriz de transición de estados usando transiciones
    intra-personales ordenadas por fecha.

    Parámetros
    ----------
    df : DataFrame
        Debe contener al menos las columnas: ["Name", "Date", col_state].
    col_state : str
        Nombre de la columna que contiene el estado discreto.
    states : lista
        Lista de estados posibles, en orden. Por defecto [1,2,3,4,5].

    Retorna
    -------
    P : np.ndarray (k x k)
        Matriz de transición, donde k = len(states).
    """
    if states is None:
        states = DEFAULT_STATES

    # Ordenar por persona y tiempo
    df_sorted = df.sort_values(["Name", "Date"])
    state_to_idx = {s: i for i, s in enumerate(states)}
    k = len(states)

    # Matriz de conteos
    C = np.zeros((k, k), dtype=float)

    # Construir transiciones intra-persona
    for _, grp in df_sorted.groupby("Name"):
        seq = grp[col_state].dropna().astype(int).tolist()
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if a in state_to_idx and b in state_to_idx:
                i = state_to_idx[a]
                j = state_to_idx[b]
                C[i, j] += 1

    # Normalizar por filas
    P = np.zeros_like(C)
    row_sums = C.sum(axis=1, keepdims=True)
    np.divide(C, row_sums, out=P, where=row_sums != 0)
    return P


def prob_mejorar(P: np.ndarray, states=None) -> float:
    """
    Probabilidad promedio de mejorar de estado (ir a estados superiores).
    """
    if states is None:
        states = DEFAULT_STATES
    k = len(states)
    return float(np.mean([P[i, i+1:].sum() for i in range(k)]))


def prob_empeorar(P: np.ndarray, states=None) -> float:
    """
    Probabilidad promedio de empeorar de estado (ir a estados inferiores).
    """
    if states is None:
        states = DEFAULT_STATES
    k = len(states)
    return float(np.mean([P[i, :i].sum() for i in range(k)]))


def estacionaria(P: np.ndarray) -> np.ndarray:
    """
    Calcula una distribución estacionaria π tal que πP = π.

    Retorna
    -------
    pi : np.ndarray (k,)
        Vector de probabilidades estacionarias.
    """
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1))
    pi = np.real(vecs[:, idx])
    pi = pi / pi.sum()
    return pi


def engagement_estacionario(pi: np.ndarray, states=None) -> float:
    """
    Calcula el nivel de engagement esperado en estado estacionario:
        sum_i (state_i * pi_i)
    """
    if states is None:
        states = DEFAULT_STATES
    return float(sum(s * p for s, p in zip(states, pi)))


def markov_summary_by_group(df: pd.DataFrame,
                            group_cols,
                            col_state: str = "Engagement Group",
                            states=None,
                            min_rows: int = 50) -> pd.DataFrame:
    """
    Calcula métricas de cadenas de Markov por combinación de columnas.

    Parámetros
    ----------
    df : DataFrame
        Debe contener al menos: ["Name", "Date", col_state] + group_cols.
    group_cols : list of str
        Columnas categóricas para agrupar, e.g. ["Project Tag","Seniority"].
    col_state : str
        Nombre de la columna de estado.
    states : list
        Lista de estados posibles.
    min_rows : int
        Mínimo de filas por grupo para considerarlo.

    Retorna
    -------
    DataFrame con columnas:
        group_cols + ["n_registros", "prob_mejorar", "prob_empeorar",
                      "engagement_estacionario"]
    """
    if states is None:
        states = DEFAULT_STATES

    # Quitar filas con NaN en las columnas de agrupamiento
    df_clean = df.dropna(subset=group_cols)

    results = []
    grouped = df_clean.groupby(group_cols)

    for combo_values, sub in grouped:
        # combo_values es una tupla si hay más de una columna de agrupación
        if isinstance(combo_values, tuple):
            combo_dict = {col: val for col, val in zip(group_cols, combo_values)}
        else:
            combo_dict = {group_cols[0]: combo_values}

        if sub.shape[0] < min_rows:
            continue

        P = build_transition_matrix(sub, col_state=col_state, states=states)
        if P.sum() == 0:
            continue

        pi = estacionaria(P)
        p_up = prob_mejorar(P, states=states)
        p_down = prob_empeorar(P, states=states)
        E_inf = engagement_estacionario(pi, states=states)

        row = {
            **combo_dict,
            "n_registros": int(sub.shape[0]),
            "prob_mejorar": float(p_up),
            "prob_empeorar": float(p_down),
            "engagement_estacionario": float(E_inf),
        }
        results.append(row)

    if not results:
        return pd.DataFrame(
            columns=group_cols
            + ["n_registros", "prob_mejorar", "prob_empeorar", "engagement_estacionario"]
        )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("engagement_estacionario", ascending=False).reset_index(
        drop=True
    )
    return res_df