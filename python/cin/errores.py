# errores local y global, R(h) y OCG
import numpy as np

from .edo import ESQUEMAS, integrar


def error_local(esquema, alpha0, q, u0, h, exacta):
    # un solo paso desde t = 0
    paso = ESQUEMAS[esquema] if isinstance(esquema, str) else esquema
    u1 = paso(u0, 0.0, h, alpha0, q)
    return abs(exacta(h) - u1)


def error_global(esquema, alpha0, q, u0, tau, hs, exacta):
    eps, uL = [], []
    for h in hs:
        _, u = integrar(esquema, alpha0, q, u0, h, tau)
        uL.append(u[-1])
        eps.append(abs(exacta(tau) - u[-1]))
    return np.array(eps), np.array(uL)


def rapidez_y_orden(eps):
    # los h van de mayor a menor dividiendo entre 2; el ultimo queda en NaN
    eps = np.asarray(eps, dtype=float)
    R = np.full_like(eps, np.nan)
    R[:-1] = eps[:-1] / eps[1:]
    return R, np.log(R) / np.log(2.0)
