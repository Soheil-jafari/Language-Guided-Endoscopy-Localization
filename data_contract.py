"""Canonical concepts and frame metadata. No model imports or inferred frame rates.

Frame IDs always refer to the zero-based source video frame. Metadata is a JSON
mapping video_id -> {source_fps, frame_count}; frame_count is the SOURCE count.
Phase annotations describe intervals until the next phase row. Tool annotations
are observations at exact source frames; missing observations stay unknown (-100).
"""
import json
import re
import zipfile
import os
from pathlib import Path
import numpy as np
import pandas as pd

IGNORE_INDEX = -100
PHASES = ('Preparation', 'CalotTriangleDissection', 'ClippingCutting',
          'GallbladderDissection', 'GallbladderPackaging', 'CleaningCoagulation',
          'GallbladderRetraction')
TOOLS = ('grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimenbag')
PHRASES = ('Preparation phase', 'Calot triangle dissection phase',
           'Clipping and cutting phase', 'Gallbladder dissection phase',
           'Gallbladder packaging phase', 'Cleaning and coagulation phase', 'Retraction phase',
           'a grasper is present', 'a bipolar forceps is present', 'a hook cautery is present',
           'scissors are present', 'a clip applier is present', 'an irrigator is present',
           'a specimen retrieval bag is present')


def normalized(text):
    return re.sub(r'[^a-z0-9]', '', str(text).lower())


ALIASES = {normalized(name): ('phase', i) for i, name in enumerate(PHASES)}
ALIASES.update({normalized(name): ('tool', i) for i, name in enumerate(TOOLS)})
for i, phrase in enumerate(PHRASES):
    ALIASES[normalized(phrase)] = ('phase', i) if i < 7 else ('tool', i - 7)
for name, kind, idx in [('calot','phase',1), ('dissection','phase',3),
                        ('clipping','phase',2), ('cutting','phase',2),
                        ('packaging','phase',4), ('cleaning','phase',5),
                        ('coagulation','phase',5), ('retraction','phase',6),
                        ('clip','tool',4), ('clip applier','tool',4),
                        ('suction','tool',5), ('a suction instrument is present','tool',5),
                        ('specimen','tool',6), ('bag','tool',6)]:
    ALIASES[normalized(name)] = (kind, idx)


def query_spec(text, kind=None, concept_id=None):
    # Explicit metadata allows arbitrary human-verified paraphrases. It labels
    # targets only: the original sentence still goes to the text encoder.
    if kind is not None and not pd.isna(kind):
        if kind not in ('phase', 'tool') or pd.isna(concept_id):
            raise ValueError('query_kind must be phase/tool with a concept_id in 0..6')
        idx = int(concept_id)
        if idx != float(concept_id) or not 0 <= idx < 7:
            raise ValueError(f'Invalid concept_id: {concept_id}')
        known=ALIASES.get(normalized(text))
        if known is not None and known != (kind,idx):
            raise ValueError(f'Concept metadata contradicts the known query: {text}')
        return kind, idx
    key = normalized(text)
    if key not in ALIASES:
        raise ValueError(f'Unlabelled query {text!r}: supply query_kind and concept_id; '
                         'unknown wording must not become a negative label.')
    return ALIASES[key]


def canonical_query(spec):
    kind, idx = spec
    return PHRASES[idx if kind == 'phase' else idx + 7]


def path_identity(path):
    parts = str(path).replace('\\', '/').split('/')
    match = re.fullmatch(r'frame_(\d+)\.jpg', parts[-1], flags=re.I)
    if len(parts) < 2 or match is None:
        raise ValueError(f'Expected video_id/frame_SOURCEINDEX.jpg: {path}')
    return parts[-2], int(match.group(1))


def load_video_metadata(path):
    with open(path, encoding='utf-8') as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError('Video metadata must be a nonempty mapping.')
    for video, row in metadata.items():
        fps, count = float(row['source_fps']), int(row['frame_count'])
        if not np.isfinite(fps) or fps <= 0 or count <= 0 or count != row['frame_count']:
            raise ValueError(f'Invalid source_fps/frame_count for {video}')
        row['source_fps'], row['frame_count'] = fps, count
    return metadata


def sample_indices(frame_count, source_fps, sample_fps):
    if frame_count <= 0 or not 0 < sample_fps <= source_fps:
        raise ValueError('Require frame_count > 0 and 0 < sample_fps <= source_fps.')
    # Round absolute times, not a rounded stride (which drifts for fractional FPS).
    times = np.arange(0, frame_count / source_fps, 1.0 / sample_fps)
    indices = np.rint(times * source_fps).astype(np.int64)
    return np.unique(indices[indices < frame_count])


def window_positions(length, window_length, stride=None):
    stride = max(1, window_length // 2) if stride is None else int(stride)
    if length <= 0 or window_length <= 0 or not 0 < stride <= window_length:
        raise ValueError('Invalid window length/stride; gaps between windows are forbidden.')
    starts = list(range(0, max(1, length-window_length+1), stride))
    last = max(0, length-window_length)
    if starts[-1] != last:
        starts.append(last)
    return starts


class FrameStore:
    """Read loose frames or flat per-video ZIPs; never manufacture image data."""
    def __init__(self, root):
        self.root = Path(root)
        self._names = {}
        self._zips = {}
        self._pid = os.getpid()

    def names(self, video):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if video not in self._names:
            archive = self.root / (video + '.zip')
            if archive.exists():
                zf = zipfile.ZipFile(archive)
                self._zips[video] = zf
                names = zf.namelist()
            else:
                folder = self.root / video
                if not folder.is_dir():
                    raise FileNotFoundError(folder)
                names = [p.name for p in folder.iterdir()]
            mapping = {}
            for name in names:
                match = re.fullmatch(r'frame_(\d+)\.jpg', name, re.I)
                if match:
                    idx = int(match.group(1))
                    if idx in mapping:
                        raise ValueError(f'Duplicate frame ID {video}:{idx}')
                    mapping[idx] = name
            if not mapping:
                raise ValueError(f'No source-indexed frames for {video}')
            self._names[video] = mapping
        return self._names[video]

    def read(self, video, idx):
        from PIL import Image
        import io
        names = self.names(video)
        if int(idx) not in names:
            raise FileNotFoundError(f'Missing source frame {video}:{idx}; re-extract on the declared grid.')
        name = names[int(idx)]
        if video in self._zips:
            with Image.open(io.BytesIO(self._zips[video].read(name))) as image:
                return image.convert('RGB')
        with Image.open(self.root / video / name) as image:
            return image.convert('RGB')

    def close(self):
        for zf in self._zips.values():
            zf.close()
        self._zips = {}
        self._names = {}

    def __getstate__(self):
        return dict(root=self.root, _names={}, _zips={}, _pid=os.getpid())


class AnnotationIndex:
    def __init__(self, csv_path):
        df = csv_path.copy() if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)
        required = {'standardized_video_id', 'frame_idx'}
        if not required.issubset(df.columns):
            raise ValueError(f'Annotations require {required}')
        self.phases, self.tools = {}, {}
        self.maximum_frame = {}
        tool_cols = [(col, ALIASES.get(normalized(col))) for col in df.columns]
        tool_cols = [(col, spec[1]) for col, spec in tool_cols if spec and spec[0] == 'tool']
        phase_rows = {}
        for row in df.to_dict('records'):
            video, idx = str(row['standardized_video_id']), int(row['frame_idx'])
            if idx < 0 or idx != float(row['frame_idx']):
                raise ValueError('Annotation frame_idx must be a nonnegative source index.')
            self.maximum_frame[video]=max(idx,self.maximum_frame.get(video,-1))
            raw = row.get('original_label')
            if pd.notna(raw) and str(raw).strip():
                phase = ALIASES.get(normalized(raw))
                if phase is None:
                    # Accept legacy lines such as "1234\tCalotTriangleDissection".
                    phase = ALIASES.get(normalized(str(raw).split()[-1]))
                if phase is None:
                    raise ValueError(f'Unrecognized original_label {raw!r}; regenerate parsed annotations from the original phase/tool files')
                if phase and phase[0] == 'phase':
                    key = (video, idx)
                    if key in phase_rows and phase_rows[key] != phase[1]:
                        raise ValueError(f'Conflicting phase annotations: {key}')
                    phase_rows[key] = phase[1]
            for col, tool_id in tool_cols:
                value = row.get(col)
                if pd.isna(value):
                    continue
                if float(value) not in (0, 1):
                    raise ValueError(f'Invalid tool annotation {video}:{idx}:{col}')
                key = video, idx, tool_id
                if key in self.tools and self.tools[key] != int(value):
                    raise ValueError(f'Conflicting tool observations: {key}')
                self.tools[key] = int(value)
        for (video, idx), phase in sorted(phase_rows.items()):
            self.phases.setdefault(video, []).append((idx, phase))
        # Dense phase rows carry the same interval label repeatedly. Retain only
        # changes so each DataLoader worker does not duplicate a per-frame table.
        for video,rows in self.phases.items():
            array=np.asarray(rows,dtype=np.int64)
            self.phases[video]=array[np.r_[True,np.diff(array[:,1])!=0]]

    def label(self, video, idx, spec):
        kind, concept_id = spec
        if kind == 'tool':
            return self.tools.get((video, int(idx), concept_id), IGNORE_INDEX)
        rows = self.phases.get(video)
        if rows is None:
            return IGNORE_INDEX
        pos = np.searchsorted(rows[:, 0], idx, side='right') - 1
        return IGNORE_INDEX if pos < 0 else int(rows[pos, 1] == concept_id)


def validate_triplets(df):
    if df.empty or not {'frame_path', 'text_query'}.issubset(df.columns):
        raise ValueError('Nonempty triplets must contain frame_path and text_query.')
    specs = [query_spec(r['text_query'], r.get('query_kind'), r.get('concept_id'))
             for r in df.to_dict('records')]
    seen, query_specs = {}, {}
    for row, spec in zip(df.to_dict('records'), specs):
        video, idx = path_identity(row['frame_path'])
        query = str(row['text_query'])
        if query in query_specs and query_specs[query] != spec:
            raise ValueError(f'Query has inconsistent concept metadata: {query}')
        query_specs[query] = spec
        label = row.get('relevance_label')
        if label is not None and not pd.isna(label):
            if float(label) not in (0, 1, IGNORE_INDEX):
                raise ValueError(f'Invalid relevance_label: {label}')
            key = video, idx, spec
            if key in seen and seen[key] != label:
                raise ValueError(f'Conflicting triplet labels: {key}')
            seen[key] = label
    return specs
