"""Compatibility entry point. Uses the repaired shared CLI; see REPAIR_NOTES.md."""
from pathlib import Path
import runpy
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'dataset_preprocessing/build_dataset.py'), run_name='__main__')
