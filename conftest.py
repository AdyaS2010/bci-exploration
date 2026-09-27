"""Put src/ on the import path so tests can `import kestrel` without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
