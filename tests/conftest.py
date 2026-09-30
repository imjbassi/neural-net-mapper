import sys
from pathlib import Path

import matplotlib

# Headless backend for CI and containers
matplotlib.use("Agg")

# Make `src` and `data` importable when running pytest from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))
