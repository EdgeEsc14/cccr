# tablas para el documento de LaTeX
from pathlib import Path
import math


def fmt(x, cifras=6):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{cifras}g}"


def tabla_latex(ruta, encabezados, columnas, cifras=6):
    # columnas: una lista por columna, todas del mismo largo
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    n = len(columnas[0])
    renglones = [" & ".join(fmt(col[i], cifras) for col in columnas) + r" \\" for i in range(n)]
    contenido = "\n".join([
        "% Generado automáticamente por Python. No editar a mano.",
        r"\begin{tabular}{" + "c" * len(encabezados) + "}",
        r"\toprule",
        " & ".join(encabezados) + r" \\",
        r"\midrule",
        *renglones,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    ruta.write_text(contenido, encoding="utf-8")
    return ruta
