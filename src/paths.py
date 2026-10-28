"""Resolve the single shared MAPE-PPI implementation independently of cwd."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONE_SOURCE = ROOT / 'MAPE-PPI' / 'src'
if str(BACKBONE_SOURCE) not in sys.path:
    sys.path.insert(0, str(BACKBONE_SOURCE))
