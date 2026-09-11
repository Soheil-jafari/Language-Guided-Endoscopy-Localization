"""Legacy dense-position loaders retired; use checked source-frame helpers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from data_contract import FrameStore,AnnotationIndex,sample_indices,load_video_metadata
from benchmark import prepare_pairs
