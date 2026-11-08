import sys
from pathlib import Path

# Agar modul di src/ dapat diimpor langsung dari tests/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
