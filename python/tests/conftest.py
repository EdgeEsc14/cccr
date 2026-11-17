import sys
from pathlib import Path

# Permite "import cin" al correr pytest desde cualquier carpeta.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
