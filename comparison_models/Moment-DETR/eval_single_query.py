"""Repaired shared evaluator; run --help for the new explicit-input CLI."""
from pathlib import Path
import runpy
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
if __name__=='__main__':
    runpy.run_path(str(ROOT/'comparison_models/Moment-DETR/run_evaluation.py'),run_name='__main__')
