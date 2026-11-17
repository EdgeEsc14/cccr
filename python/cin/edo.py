# esquemas de un paso para du/dt = -alpha0 u + q(t)
# cada paso_* recibe u_l y regresa u_{l+1}; q puede ser numero o funcion
import numpy as np


def _q(q, t):
    return q(t) if callable(q) else q


def paso_euler_explicito(u, t, h, alpha0, q):
    return u + h * (-alpha0 * u + _q(q, t))


def paso_euler_implicito(u, t, h, alpha0, q):
    # u_{l+1} (1 + alpha0 h) = u_l + h q_{l+1}
    return (u + h * _q(q, t + h)) / (1.0 + alpha0 * h)


def paso_crank_nicolson(u, t, h, alpha0, q):
    # u_{l+1} (1 + h alpha0/2) = u_l (1 - h alpha0/2) + h (q_l + q_{l+1})/2
    q_prom = 0.5 * (_q(q, t) + _q(q, t + h))
    return ((1.0 - h * alpha0 / 2) * u + h * q_prom) / (1.0 + h * alpha0 / 2)


ESQUEMAS = {
    "EE": paso_euler_explicito,
    "EI": paso_euler_implicito,
    "CN": paso_crank_nicolson,
}


def integrar(esquema, alpha0, q, u0, h, tau):
    L = int(round(tau / h))
    if abs(L * h - tau) > 1e-12 * max(1.0, tau):
        raise ValueError(f"tau/h = {tau / h} no es entero: el último nodo no cae en tau.")
    paso = ESQUEMAS[esquema] if isinstance(esquema, str) else esquema
    t = h * np.arange(L + 1)
    u = np.empty(L + 1)
    u[0] = u0
    for l in range(L):
        u[l + 1] = paso(u[l], t[l], h, alpha0, q)
    return t, u


def factor_amplificacion(esquema, z):
    # z = -alpha0 h
    z = np.asarray(z, dtype=float)
    if esquema == "EE":
        return 1.0 + z
    if esquema == "EI":
        return 1.0 / (1.0 - z)
    if esquema == "CN":
        return (1.0 + z / 2) / (1.0 - z / 2)
    raise ValueError(f"Esquema desconocido: {esquema}")


def solucion_exacta_fuente_constante(t, alpha0, q0, u0):
    # u(t) = u* + (u0 - u*) e^(-alpha0 t), u* = q0/alpha0
    u_fijo = q0 / alpha0
    return u_fijo + (u0 - u_fijo) * np.exp(-alpha0 * np.asarray(t, dtype=float))
