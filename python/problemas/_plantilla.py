"""Plantilla de script de problema.

Copia como  pN.py  (mismo nombre que el .tex, p. ej. p3.py) y ejecútalo desde python/ con
    python -m problemas.pN
o desde la raíz con  make figs.
Las figuras van a latex/figuras/pN_*.pdf y las tablas a latex/tablas/pN_*.tex.
"""
from pathlib import Path

import numpy as np  # noqa: F401

import cin

RAIZ = Path(__file__).resolve().parents[2]
FIGURAS = RAIZ / "latex" / "figuras"
TABLAS = RAIZ / "latex" / "tablas"


def main():
    cin.estilo()
    # 1) Datos del problema
    # 2) Cálculos (imprime los resultados con al menos 6 cifras significativas)
    # 3) Tablas:  cin.tabla_latex(TABLAS / "pN_algo.tex", encabezados, columnas)
    # 4) Figuras: cin.guardar(fig, FIGURAS / "pN_algo.pdf")


if __name__ == "__main__":
    main()
