"""
Problema 8. Difusión 2D transitoria y verificación del orden.

Notación de los apuntes (Sesiones 006, 007, 009 y Resumen a mano):
    ell      : índice de tiempo (el k del enunciado)
    I        : número de subintervalos en x; n = I - 1 nodos interiores
    h        : tamaño de subintervalo (h = Dx = Dy)
    tau      : tiempo final (el T del enunciado); L = tau/dt pasos de tiempo
    B = G G^T: factorización de Cholesky del lado izquierdo de CN

Genera (con prefijo p08_):
    tablas/p08_c_radio.tex       inciso (c): radio espectral de A_2D contra n
    tablas/p08_d_errores.tex     inciso (d): error, R(dt) y OCG(dt)
    figuras/p08_d_orden.pdf/png  inciso (d): error contra dt en log-log
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Carpetas de salida: <raíz>/figuras y <raíz>/tablas,
# con este archivo en <raíz>/python/problemas/
# ---------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
FIG = RAIZ / "figuras"
TAB = RAIZ / "tablas"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)


# =====================================================================
# Matrices del inciso (a)
# =====================================================================
def matriz_A1(n, D, h):
    """A_1 = (D/h^2) tridiag(-1, 2, -1), de tamaño n x n (sin Sigma_a)."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 * D / h**2
    for i in range(n - 1):
        A[i, i + 1] = -D / h**2
        A[i + 1, i] = -D / h**2
    return A


def matriz_A2D_kron(n, D, h, Sa=0.0):
    """A_2D = I (x) A_1 + A_1 (x) I + Sigma_a I  (producto de Kronecker)."""
    A1 = matriz_A1(n, D, h)
    In = np.eye(n)
    return np.kron(In, A1) + np.kron(A1, In) + Sa * np.eye(n * n)


def matriz_A2D_estencil(n, D, h, Sa=0.0):
    """A_2D ensamblada nodo por nodo con el esténcil de 5 puntos,
    ordenamiento fila por fila k = i + (j-1) n (como el programa EPDE)."""
    N = n * n
    A = np.zeros((N, N))
    for j in range(1, n + 1):
        for i in range(1, n + 1):
            k = i + (j - 1) * n - 1           # -1: Python cuenta desde 0
            A[k, k] = 4 * D / h**2 + Sa
            if i > 1:
                A[k, k - 1] = -D / h**2       # vecino (i-1, j)
            if i < n:
                A[k, k + 1] = -D / h**2       # vecino (i+1, j)
            if j > 1:
                A[k, k - n] = -D / h**2       # vecino (i, j-1)
            if j < n:
                A[k, k + n] = -D / h**2       # vecino (i, j+1)
    return A


def mu_pq(p, q, n, D, h):
    """Valores propios de A_2D con Sigma_a = 0 (inciso c)."""
    th_p = p * np.pi / (n + 1)
    th_q = q * np.pi / (n + 1)
    return 2 * D / h**2 * (2 - np.cos(th_p) - np.cos(th_q))


# =====================================================================
# Cholesky de una matriz tridiagonal simétrica (fórmulas de los apuntes)
#   B = G G^T,  G bidiagonal inferior
#   g_11 = sqrt(b_11)
#   g_{i,i-1} = b_{i,i-1} / g_{i-1,i-1}
#   g_ii = sqrt(b_ii - g_{i,i-1}^2),   i = 2, ..., n
# =====================================================================
def cholesky_tridiagonal(b_diag, b_sub):
    n = len(b_diag)
    g_diag = np.zeros(n)
    g_sub = np.zeros(n - 1)            # g_sub[i-1] = g_{i,i-1}
    g_diag[0] = np.sqrt(b_diag[0])
    for i in range(1, n):
        g_sub[i - 1] = b_sub[i - 1] / g_diag[i - 1]
        radicando = b_diag[i] - g_sub[i - 1] ** 2
        if radicando <= 0:
            raise ValueError("B no es definida positiva")
        g_diag[i] = np.sqrt(radicando)
    return g_diag, g_sub


def resolver_cholesky_tridiagonal(g_diag, g_sub, r):
    """Resuelve G G^T x = r: primero G y = r (adelante), luego G^T x = y (atrás)."""
    n = len(g_diag)
    y = np.zeros(n)
    y[0] = r[0] / g_diag[0]
    for i in range(1, n):
        y[i] = (r[i] - g_sub[i - 1] * y[i - 1]) / g_diag[i]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / g_diag[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - g_sub[i] * x[i + 1]) / g_diag[i]
    return x


# =====================================================================
# Verificación numérica de los incisos (a), (b) y (c)
# =====================================================================
def verificar_a_b_c():
    print("=" * 70)
    print("Verificación de (a), (b) y (c)")
    print("=" * 70)
    D, v, Sa, dt = 1.0, 1.0, 0.3, 0.01
    for n in (3, 4, 7):
        h = 1.0 / (n + 1)
        Ak = matriz_A2D_kron(n, D, h, Sa)
        Ae = matriz_A2D_estencil(n, D, h, Sa)
        B = np.eye(n * n) + v * dt / 2 * Ak
        G = np.linalg.cholesky(B)       # falla si B no es SPD
        print(f"n = {n}:  max|A_kron - A_estencil| = {np.max(np.abs(Ak - Ae)):.1e}, "
              f"max|B - B^T| = {np.max(np.abs(B - B.T)):.1e}, "
              f"min eig(B) = {np.linalg.eigvalsh(B).min():.6f}, "
              f"max|G G^T - B| = {np.max(np.abs(G @ G.T - B)):.1e}")
        # valores propios con Sigma_a = 0 contra la fórmula del inciso (c)
        A0 = matriz_A2D_kron(n, D, h, 0.0)
        num = np.sort(np.linalg.eigvalsh(A0))
        form = np.sort([mu_pq(p, q, n, D, h)
                        for p in range(1, n + 1) for q in range(1, n + 1)])
        print(f"        max|mu_numérico - mu_fórmula| = {np.max(np.abs(num - form)):.1e}")


def tabla_c():
    """Radio espectral de A_2D (D = 1, [a,b] = [0,1]) contra n."""
    D, v = 1.0, 1.0
    filas = []
    for n in (3, 9, 19, 39, 99, 199):
        h = 1.0 / (n + 1)
        rho_f = 4 * D / h**2 * (1 + np.cos(np.pi / (n + 1)))
        if n <= 39:
            rho_n = np.linalg.eigvalsh(matriz_A2D_kron(n, D, h)).max()
            rho_n_txt = f"{rho_n:.6f}"
        else:
            rho_n_txt = "---"
        dt_fe = 2 / (v * rho_f)
        filas.append((n, h, rho_f, rho_n_txt, rho_f * h**2 / (8 * D),
                      dt_fe, h**2 / (4 * v * D)))
    def cient(x):
        m, e = f"{x:.5e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    with open(TAB / "p08_c_radio.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rcccccc}\\toprule\n")
        f.write("$n$ & $h$ & $\\rho$ (fórmula) & $\\rho$ (numérico) & "
                "$\\rho h^2/(8D)$ & $2/(v\\rho)$ & $h^2/(4vD)$\\\\\\midrule\n")
        for n, h, rf, rn, cociente, dfe, dcota in filas:
            f.write(f"{n} & $1/{n + 1}$ & {rf:.6f} & {rn} & {cociente:.6f} & "
                    f"{cient(dfe)} & {cient(dcota)}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("\nTabla (c) escrita en", TAB / "p08_c_radio.tex")


# =====================================================================
# Inciso (d): CN en 1D y orden en el tiempo
# =====================================================================
def cn_1d(dt, tau, n, D, v, h, phi0):
    """CN:  (I + v dt/2 A_1) phi^{l+1} = (I - v dt/2 A_1) phi^l   (S = 0).
    B se factoriza UNA sola vez, antes del ciclo en el tiempo."""
    L = int(round(tau / dt))
    sigma = v * dt * D / h**2                # sigma = v dt D / h^2
    # Lado izquierdo B: diagonal 1 + sigma, sub/supra diagonal -sigma/2
    b_diag = (1 + sigma) * np.ones(n)
    b_sub = (-sigma / 2) * np.ones(n - 1)
    g_diag, g_sub = cholesky_tridiagonal(b_diag, b_sub)   # B = G G^T (una vez)
    # Lado derecho C: diagonal 1 - sigma, sub/supra diagonal +sigma/2
    phi = phi0.copy()
    for ell in range(L):
        r = (1 - sigma) * phi                # r^l = C phi^l
        r[1:] += (sigma / 2) * phi[:-1]
        r[:-1] += (sigma / 2) * phi[1:]
        phi = resolver_cholesky_tridiagonal(g_diag, g_sub, r)
    return phi, L, sigma, g_diag, g_sub


def inciso_d():
    print("\n" + "=" * 70)
    print("Inciso (d): CN en 1D")
    print("=" * 70)
    a, b = 0.0, 1.0
    D, v, Sa = 1.0, 1.0, 0.0
    h = 1.0 / 20
    I = int(round((b - a) / h))              # subintervalos
    n = I - 1                                # nodos interiores
    x = a + h * np.arange(1, n + 1)          # x_1, ..., x_{I-1}
    tau = 0.1
    phi0 = np.sin(np.pi * x)

    mu = 2 * D / h**2 * (1 - np.cos(np.pi * h))
    phi_semi = np.exp(-v * mu * tau) * np.sin(np.pi * x)           # exacta semidiscreta
    phi_edp = np.exp(-v * D * np.pi**2 * tau) * np.sin(np.pi * x)  # exacta de la EDP
    print(f"I = {I}, n = {n}, mu = {mu:.10f}, pi^2 - mu = {np.pi**2 - mu:.6e}")
    print(f"exp(-v mu tau) = {np.exp(-v * mu * tau):.10f}, "
          f"exp(-v D pi^2 tau) = {np.exp(-v * D * np.pi**2 * tau):.10f}")
    C_teo = np.exp(-v * mu * tau) * (v * mu) ** 3 * tau / 12
    print(f"constante teórica C = e^(-v mu tau)(v mu)^3 tau/12 = {C_teo:.6f}")

    dts = [0.02, 0.01, 0.005, 0.0025]
    res = []
    i_medio = n // 2                         # x_10 = 0.5 (índice 9 en Python)
    for dt in dts:
        phi, L, sigma, g_diag, g_sub = cn_1d(dt, tau, n, D, v, h, phi0)
        eps = np.max(np.abs(phi - phi_semi))         # norma infinito
        eps_edp = np.max(np.abs(phi - phi_edp))
        # comprobación: CN debe dar g^L sin(pi x_i), g = (1 - z/2)/(1 + z/2)
        z = v * mu * dt
        gcn = (1 - z / 2) / (1 + z / 2)
        dif_teo = np.max(np.abs(phi - gcn**L * np.sin(np.pi * x)))
        res.append(dict(dt=dt, L=L, sigma=sigma, phi=phi[i_medio],
                        semi=phi_semi[i_medio], eps=eps, eps_edp=eps_edp,
                        g=gcn, dif_teo=dif_teo, g11=g_diag[0], g21=g_sub[0],
                        g22=g_diag[1]))
        print(f"dt = {dt:<7} L = {L:<3} sigma = {sigma:<5} g_11 = {g_diag[0]:.6f} "
              f"g_21 = {g_sub[0]:.6f} g_22 = {g_diag[1]:.6f}")
        print(f"     phi_10^L = {phi[i_medio]:.10f}  eps = {eps:.6e}  "
              f"eps/dt^2 = {eps / dt**2:.6f}  |CN - g^L w_1| = {dif_teo:.1e}  "
              f"error contra EDP = {eps_edp:.6e}")

    # Rapidez y orden de convergencia (Sesión 02): R(dt) = eps(dt)/eps(dt/2)
    for k in range(len(res) - 1):
        R = res[k]["eps"] / res[k + 1]["eps"]
        res[k]["R"] = R
        res[k]["OCG"] = np.log(R) / np.log(2)
        R_edp = res[k]["eps_edp"] / res[k + 1]["eps_edp"]
        print(f"dt = {res[k]['dt']:<7} R = {R:.6f}  OCG = {res[k]['OCG']:.6f}   "
              f"(contra EDP: R = {R_edp:.4f}, OCG = {np.log(R_edp) / np.log(2):.4f})")

    # ----------------------------- tabla -----------------------------
    def cient(x):
        m, e = f"{x:.5e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"

    with open(TAB / "p08_d_errores.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{cccccccc}\\toprule\n")
        f.write("$\\Delta t$ & $L$ & $\\varphi_{10}^{L}$ (CN) & "
                "$\\varphi_{10}(\\tau)$ & $\\varepsilon(\\Delta t)$ & "
                "$\\varepsilon/\\Delta t^2$ & $R(\\Delta t)$ & "
                "$\\OCG(\\Delta t)$\\\\\\midrule\n")
        for r in res:
            R_txt = f"{r['R']:.6f}" if "R" in r else "---"
            O_txt = f"{r['OCG']:.6f}" if "OCG" in r else "---"
            f.write(f"{r['dt']} & {r['L']} & {r['phi']:.6f} & {r['semi']:.6f} & "
                    f"{cient(r['eps'])} & {r['eps'] / r['dt']**2:.6f} & "
                    f"{R_txt} & {O_txt}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("Tabla (d) escrita en", TAB / "p08_d_errores.tex")

    # ----------------------------- figura ----------------------------
    dt_arr = np.array([r["dt"] for r in res])
    eps_arr = np.array([r["eps"] for r in res])
    edp_arr = np.array([r["eps_edp"] for r in res])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(dt_arr, eps_arr, "o-", color="C0", lw=1.8, ms=7,
              label=r"$\varepsilon(\Delta t)$: CN contra la semidiscreta")
    ax.loglog(dt_arr, C_teo * dt_arr**2, "k--", lw=1.2,
              label=rf"teoría $C\,\Delta t^2$, $C={C_teo:.4f}$ (pendiente 2)")
    ax.loglog(dt_arr, edp_arr, "s:", color="C3", ms=6,
              label=r"CN contra la exacta de la EDP (incluye error en $h$)")
    ax.set_xticks(dt_arr)
    ax.set_xticklabels([f"{d:g}" for d in dt_arr])
    ax.minorticks_off()
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"error en norma infinito en $\tau = 0.1$")
    ax.set_title(r"CN 1D, $h = 1/20$: orden 2 en $\Delta t$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "p08_d_orden.pdf")
    fig.savefig(FIG / "p08_d_orden.png", dpi=200)
    plt.close(fig)
    print("Figura escrita en", FIG / "p08_d_orden.pdf")


if __name__ == "__main__":
    verificar_a_b_c()
    tabla_c()
    inciso_d()
