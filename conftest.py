import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
