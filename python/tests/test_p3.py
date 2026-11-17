"""Pruebas del Problema 3. Ejecutar desde python/:  pytest -q"""
import math

import cin

ALPHA0, Q0, U0 = 2.0, 4.0, 0.0


def exacta(t):
    return cin.solucion_exacta_fuente_constante(t, ALPHA0, Q0, U0)


def test_inciso_b():
    _, u = cin.integrar("CN", ALPHA0, Q0, U0, 0.25, 0.5)
    assert math.isclose(u[1], 0.8)
    assert math.isclose(u[2], 1.28)


def test_punto_fijo_cn_y_ei():
    for esquema in ("CN", "EI"):
        for h in (0.1, 0.5, 3.0):
            assert math.isclose(cin.ESQUEMAS[esquema](2.0, 0.0, h, ALPHA0, Q0), 2.0)


def test_forma_cerrada():
    h = 0.125
    r = (1 - h * ALPHA0 / 2) / (1 + h * ALPHA0 / 2)
    _, u = cin.integrar("CN", ALPHA0, Q0, U0, h, 1.0)
    assert math.isclose(u[-1], (Q0 / ALPHA0) * (1 - r**8))


def test_orden_cn_es_2():
    eps, _ = cin.error_global("CN", ALPHA0, Q0, U0, 1.0, [1/2, 1/4, 1/8, 1/16], exacta)
    _, ocg = cin.rapidez_y_orden(eps)
    assert abs(ocg[-2] - 2) < 0.05


def test_orden_ei_es_1():
    eps, _ = cin.error_global("EI", ALPHA0, Q0, U0, 1.0, [1/8, 1/16, 1/32, 1/64], exacta)
    _, ocg = cin.rapidez_y_orden(eps)
    assert abs(ocg[-2] - 1) < 0.1
