# Problema 3 - decaimiento con fuente: N' = -lambda N + S, lambda = 2, S = 4, N0 = 0
# correr con: python -m problemas.p3  (desde python/)
from fractions import Fraction
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import cin

RAIZ = Path(__file__).resolve().parents[2]
FIGURAS = RAIZ / "latex" / "figuras"
TABLAS = RAIZ / "latex" / "tablas"

ALPHA0 = 2.0   # lambda
Q0 = 4.0       # S
U0 = 0.0       # N0
U_FIJO = Q0 / ALPHA0


def exacta(t):
    return cin.solucion_exacta_fuente_constante(t, ALPHA0, Q0, U0)


def inciso_b():
    print("\n(b) h = 1/4")
    h = 0.25
    r = (1 - h * ALPHA0 / 2) / (1 + h * ALPHA0 / 2)
    c = h * Q0 / (1 + h * ALPHA0 / 2)
    print(f"    esquema: u_(l+1) = {r:.6g} u_l + {c:.6g}")
    t, u = cin.integrar("CN", ALPHA0, Q0, U0, h, tau=0.5)
    err = np.abs(exacta(t) - u)
    for l in (1, 2):
        print(f"    l={l}  t={t[l]:.2f}  u_l={u[l]:.6f}  exacta={exacta(t[l]):.6f}  error={err[l]:.6g}")
    cin.tabla_latex(
        TABLAS / "p3_b.tex",
        [r"$\ell$", r"$t_\ell$", r"$u_\ell$ (CN)", r"$u(t_\ell)$", r"$|u(t_\ell)-u_\ell|$"],
        [[1, 2], list(t[1:]), list(u[1:]), list(exacta(t[1:])), list(err[1:])],
    )


def inciso_c():
    print("\n(c) punto fijo u* = q0/alpha0 =", U_FIJO)
    for esquema in ("CN", "EI"):
        for h in (0.01, 0.25, 1.0, 10.0):
            u_sig = cin.ESQUEMAS[esquema](U_FIJO, 0.0, h, ALPHA0, Q0)
            assert abs(u_sig - U_FIJO) < 1e-12
        print(f"    {esquema}: u_l = u*  =>  u_(l+1) = u*  (verificado para varios h)")


def inciso_d():
    print("\n(d) error global en tau = 1")
    tau = 1.0
    hs = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8), Fraction(1, 16)]
    hf = [float(h) for h in hs]
    resultados = {}
    for esquema in ("CN", "EI"):
        eps, uL = cin.error_global(esquema, ALPHA0, Q0, U0, tau, hf, exacta)
        R, ocg = cin.rapidez_y_orden(eps)
        resultados[esquema] = eps
        print(f"  {esquema}:")
        for h, L, u, e, rr, p in zip(hs, [round(tau / x) for x in hf], uL, eps, R, ocg):
            print(f"    h={str(h):>5}  L={L:>2}  u_L={u:.6f}  eps={e:.6g}  R={rr:.6g}  OCG={p:.6g}")
        if esquema == "CN":
            cin.tabla_latex(
                TABLAS / "p3_d.tex",
                [r"$h$", r"$L$", r"$u_L$", r"$\varepsilon(h)$", r"$R(h)$", r"$\mathrm{OCG}(h)$"],
                [[f"{h.numerator}/{h.denominator}" for h in hs],
                 [round(tau / x) for x in hf], uL, eps, R, ocg],
            )
    return np.array(hf), resultados


def figura_solucion():
    h, tau = 0.25, 2.0
    t_fino = np.linspace(0, tau, 400)
    t, u_cn = cin.integrar("CN", ALPHA0, Q0, U0, h, tau)
    _, u_ei = cin.integrar("EI", ALPHA0, Q0, U0, h, tau)
    u_ex = exacta(t)
    cin.tabla_latex(
        TABLAS / "p3_b_todos.tex",
        [r"$\ell$", r"$t_\ell$", r"$u_\ell$ (CN)", r"$u_\ell$ (EI)", r"$u(t_\ell)$",
         "error CN", "error EI"],
        [list(range(len(t))), list(t), list(u_cn), list(u_ei), list(u_ex),
         list(np.abs(u_ex - u_cn)), list(np.abs(u_ex - u_ei))],
    )
    fig, ax = plt.subplots()
    ax.plot(t_fino, exacta(t_fino), "k-", lw=1.5, label=r"Exacta $2(1-e^{-2t})$")
    ax.plot(t, u_cn, "^", ms=7, color="0.25", label=r"CN, $h=1/4$")
    ax.plot(t, u_ei, "x", ms=7, mew=1.5, color="tab:red", label=r"Euler implícito, $h=1/4$")
    ax.axhline(U_FIJO, ls="--", color="gray", lw=1, label=r"$u^*=q_0/\alpha_0$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$u$")
    ax.legend(loc="lower right")
    cin.guardar(fig, FIGURAS / "p3_solucion.pdf")


def figura_error(hs, resultados):
    fig, ax = plt.subplots()
    ax.loglog(hs, resultados["CN"], "o-", label="Crank-Nicolson")
    ax.loglog(hs, resultados["EI"], "s-", mfc="none", label="Euler implícito")
    ref = hs / hs[0]
    ax.loglog(hs, resultados["CN"][0] * ref**2, "k--", lw=1, label="pendiente 2")
    ax.loglog(hs, resultados["EI"][0] * ref, "k:", lw=1, label="pendiente 1")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"1/{round(1 / h)}" for h in hs])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\varepsilon(h)=|u(\tau)-u_L|$")
    ax.legend()
    cin.guardar(fig, FIGURAS / "p3_error.pdf")


def main():
    cin.estilo()
    print("Problema 3. Decaimiento con fuente")
    inciso_b()
    inciso_c()
    hs, resultados = inciso_d()
    figura_solucion()
    figura_error(hs, resultados)
    print(f"\nFiguras en {FIGURAS}\nTablas en {TABLAS}")


if __name__ == "__main__":
    main()
