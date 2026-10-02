#!/usr/bin/env python3
"""One-command pipeline: raw Cholec80 download -> frames -> labels -> splits -> train -> evaluate.

    python main.py --preset full --root /scratch/me/lgel \
        --data-url "$LGEL_DATA_URL" --weights-url "$LGEL_WEIGHTS_URL"

Every stage is idempotent: it writes a marker in <root>/state when it finishes and is
skipped on the next invocation, so a job killed by a wall-clock limit is resumed simply
by submitting the same command again (training resumes from the last finished epoch).
All heavy lifting is done by the repository's existing, tested entry points
(dataset_preprocessing/build_dataset.py, audit_data.py, train.py, predict.py,
build_reference.py, evaluate.py); this file only sequences them, fetches inputs, runs the
frame extraction in parallel and repairs the known trailing-row defect in two Cholec80
phase-annotation files.

Stages (run `python main.py --list-stages`):
  preflight fetch_models fetch_weights fetch_data extract_frames sanitize_annotations
  parse_annotations splits triplets audit train predict evaluate summary
"""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures as cf
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
import zlib
from pathlib import Path

try:
    import fcntl                                 # POSIX advisory locks (not available on Windows)
except ImportError:                              # pragma: no cover
    fcntl = None

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

STAGES = ['preflight', 'fetch_models', 'fetch_weights', 'fetch_data', 'extract_frames',
          'sanitize_annotations', 'parse_annotations', 'splits', 'triplets', 'audit',
          'train', 'predict', 'evaluate', 'summary']
ALWAYS_RUN = {'preflight', 'summary'}          # cheap, never skipped
# Stages whose output belongs to one training run (<root>/runs/<run>, <root>/results/<run>): their
# 'finished' markers are kept per run, so several runs (pilot, full, other seeds) can share one --root
# and reuse its data stages.
RUN_SCOPED = {'train', 'predict', 'evaluate'}

# name -> defaults. CLI flags override any of these.
PRESETS = {
    # Plumbing check on a handful of videos: minutes, not hours.
    'smoke': dict(max_videos=6, epochs=1, warmup=1, subset_ratio=0.1, batch_size=2, accum=2,
                  audit='sample', min_free_gb=10),
    # "Does it learn?" run: all videos, 10 % of the training windows, 3 epochs.
    'pilot': dict(max_videos=None, epochs=3, warmup=1, subset_ratio=0.1, batch_size=8, accum=4,
                  audit='all', min_free_gb=200),
    # The real run: project defaults (20 epochs, effective batch 192), all training windows.
    'full': dict(max_videos=None, epochs=20, warmup=3, subset_ratio=1.0, batch_size=8, accum=24,
                 audit='all', min_free_gb=200),
}


class StageError(RuntimeError):
    pass


CHECKPOINT_REJECTED = 3            # exit code of `main.py _check_model` when the checkpoint file itself is unusable


# --------------------------------------------------------------------------- utilities
_LOGFILE = None


def log(msg=''):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    if _LOGFILE is not None:
        with open(_LOGFILE, 'a') as f:
            f.write(line + '\n')


def sha(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:16]


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()[:16]


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def cpu_count():
    try:
        return len(os.sched_getaffinity(0))      # respects scheduler/cgroup limits
    except AttributeError:
        return os.cpu_count() or 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')     # unique per process: no clash between jobs
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def gb(n):
    return n / 1024 ** 3


class Paths:
    def __init__(self, root, run):
        self.root = Path(root).expanduser().resolve()
        self.run_name = run
        r = self.root
        self.raw, self.weights_dir, self.hf = r / 'raw', r / 'weights', r / 'hf_cache'
        self.zip, self.extracted, self.inventory = self.raw / 'dataset.zip', self.raw / 'extracted', self.raw / 'inventory.json'
        self.weights = self.weights_dir / 'checkpoint.pth'
        self.data = r / 'data'
        self.frames, self.shards = self.data / 'frames', self.data / 'metadata_shards'
        self.metadata, self.clean = self.data / 'video_metadata.json', self.data / 'annotations_clean'
        self.parsed, self.splits = self.data / 'parsed_annotations.csv', self.data / 'splits.json'
        self.triplets, self.audit = self.data / 'triplets', self.data / 'preflight_audit.json'
        self.state, self.logs, self.tmp = r / 'state', r / 'logs', r / 'tmp'
        self.weights_ref = self.state / 'weights_ref.json'
        self.run, self.run_cfg = r / 'runs' / run, r / 'runs' / f'{run}.config.json'
        self.run_inputs = r / 'runs' / f'{run}.inputs.json'
        self.prep_lock, self.run_lock = self.state / 'prepare.lock', self.state / f'run.{run}.lock'
        self.results = r / 'results' / run

    def triplet_csv(self, split):
        return self.triplets / f'cholec80_{split}_triplets.csv'

    def marker(self, stage):
        if stage in RUN_SCOPED:
            return self.state / f'{stage}.{self.run_name}.done.json'
        return self.state / f'{stage}.done.json'


def stage_env(a, p):
    env = os.environ.copy()
    env.update(HF_HOME=str(p.hf), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=str(REPO) + os.pathsep + env.get('PYTHONPATH', ''))
    if a.offline or p.marker('fetch_models').exists():
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')   # cache was populated already
    return env


def run(cmd, a, p, name, progress_path=None, heartbeat=600, env=None, on_line=None):
    """Run a subprocess, stream its output to the console and logs/<name>.log."""
    p.logs.mkdir(parents=True, exist_ok=True)
    logfile = p.logs / f'{name}.log'
    cmd = [str(c) for c in cmd]
    log(f'$ {" ".join(cmd)}')
    start, stop = time.time(), threading.Event()

    def beat():
        while not stop.wait(heartbeat):
            extra = ''
            if progress_path is not None and Path(progress_path).exists():
                extra = f', {gb(Path(progress_path).stat().st_size):.1f} GB so far'
            log(f'... {name} still running ({(time.time() - start) / 60:.0f} min{extra})')

    threading.Thread(target=beat, daemon=True).start()
    try:
        with open(logfile, 'a') as lf:
            lf.write(f'\n===== {time.ctime()} :: {" ".join(cmd)}\n')
            proc = subprocess.Popen(cmd, cwd=REPO, env=env or stage_env(a, p), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
                if on_line is not None:
                    on_line(line)
            rc = proc.wait()
    finally:
        stop.set()
    if rc != 0:
        hint = ''
        if rc == -9:
            hint = (' (killed with SIGKILL: almost always the machine ran out of RAM, or the scheduler enforced its '
                    'memory limit - request more --mem, or lower --train-workers / --batch-size)')
        elif rc == -15:
            hint = ' (terminated with SIGTERM: usually the scheduler hit the job time limit - just submit the same command again)'
        err = StageError(f'`{cmd[0]} {cmd[1] if len(cmd) > 1 else ""}` exited with code {rc}{hint}. See {logfile}')
        err.rc = rc
        raise err
    return rc


def disk_needed_gb(a, p, selected):
    """Free space the stages that are still to run need: the preset's figure before the download, and
    only what is left to write afterwards (frames, checkpoints), so a resubmitted or offline job is not
    refused because of the space the earlier stages legitimately used. --min-free-gb caps all values."""
    def tree_bytes(path):
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def frames_need(video_bytes):                      # JPEGs at 1 fps: well under half the video size
        return 0.5 * video_bytes / 1024 ** 3 + 10

    todo = [s for s in selected if s not in ALWAYS_RUN and not p.marker(s).exists()]
    if 'fetch_data' in todo and (a.synthetic or a.data_dir):
        videos = 0
        for f in (Path(a.data_dir).rglob('*.mp4') if a.data_dir else ()):
            try:
                videos += f.stat().st_size
            except OSError:                                  # broken link etc.: find_inventory reports it
                pass
        need, what = frames_need(videos), 'frame extraction'
    elif 'fetch_data' in todo:
        present = tree_bytes(p.raw / 'extracted.partial')   # a partial unpack is replaced, not added to
        if p.zip.exists():                                   # a (partial) download is continued
            present += p.zip.stat().st_size
        if a.data_zip:                                       # the archive lives elsewhere: no download needed
            try:
                present += Path(a.data_zip).stat().st_size
            except OSError:
                pass
        need, what = max(frames_need(0), a.min_free_gb - present / 1024 ** 3), 'downloading and unpacking the dataset'
    elif 'extract_frames' in todo:
        video_bytes = 0
        if p.inventory.exists():
            for files in json.loads(p.inventory.read_text())['videos'].values():
                try:
                    video_bytes += Path(files['video']).stat().st_size
                except OSError:
                    pass
        need, what = frames_need(video_bytes), 'frame extraction'
    elif any(s in RUN_SCOPED for s in selected):
        need, what = 10, 'checkpoints and results'
    else:
        return 0, ''
    return min(need, a.min_free_gb), what


def native_bf16():
    """True only on GPUs with bf16 tensor cores (compute capability >= 8: A100, L40, RTX 30xx+).
    torch.cuda.is_bf16_supported() also counts *emulated* bf16 on V100/T4, which is much slower."""
    import torch
    return bool(torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)


# A lock whose holder stopped refreshing it for this long belongs to a job that was killed.
LEASE_STALE = float(os.environ.get('LGEL_LOCK_STALE_SECONDS', 300))
LEASE_BEAT = max(1.0, LEASE_STALE / 10)


class FileLock:
    """Exclusive lock on a file under <root>/state, used so that concurrent jobs on one --root take turns.

    Normally an OS advisory lock (fcntl), which the system releases by itself when a job ends or is
    killed. On file systems without lock support (or with LGEL_LOCK_MODE=lease, e.g. for Lustre
    mounted with node-local `localflock`), a lease file is used instead: created atomically, kept
    fresh by a heartbeat, and considered abandoned when its heartbeat is older than LEASE_STALE."""

    mode = None                                   # 'fcntl' or 'lease', decided once per process
    _held_leases = []

    def __init__(self, path):
        self.path, self.fh, self.lease, self._stop = Path(path), None, None, None

    @property
    def held(self):
        return self.fh is not None or self.lease is not None

    @property
    def lease_path(self):
        return self.path.with_name(self.path.name + '.lease')

    def acquire(self, blocking, waiting_for=''):
        if self.held:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if FileLock.mode is None:
            FileLock.mode = 'lease' if (fcntl is None or os.environ.get('LGEL_LOCK_MODE') == 'lease') else 'fcntl'
        if FileLock.mode == 'fcntl':
            got = self._fcntl_acquire(blocking, waiting_for)
            if got is not None:
                return got
            FileLock.mode = 'lease'
            log('NOTE: this file system does not support file locks; using lock files with a heartbeat instead '
                f'(a lock left behind by a killed job expires after {LEASE_STALE / 60:.0f} min).')
        return self._lease_acquire(blocking, waiting_for)

    def _fcntl_acquire(self, blocking, waiting_for):
        """True / False, or None if the file system cannot lock."""
        fh = open(self.path, 'a+')
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                fh.close()
                return None
            if not blocking:
                fh.close()
                return False
            log(f'waiting: another job is {waiting_for} in this --root; continuing as soon as it is done ...')
            fcntl.lockf(fh, fcntl.LOCK_EX)
        self.fh = fh
        return True

    def _lease_age(self, path=None):
        """Seconds since the lease was refreshed, measured with the file system's own clock."""
        probe = self.path.with_name(f'.clock.{socket.gethostname()}.{os.getpid()}')
        try:
            probe.write_text('')
            now = probe.stat().st_mtime
            probe.unlink()
            return now - (path or self.lease_path).stat().st_mtime
        except FileNotFoundError:
            return None

    @staticmethod
    def _token(path):
        try:
            return json.loads(Path(path).read_text()).get('token')
        except (OSError, ValueError, AttributeError):
            return None

    def _lease_acquire(self, blocking, waiting_for):
        lease, told = self.lease_path, False
        give_up = time.time() + LEASE_STALE + 2 * LEASE_BEAT     # non-blocking: long enough to tell live from dead
        while True:
            try:
                fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                stale_token = self._token(lease)
                age = self._lease_age()
                if age is None:
                    continue                                  # it was just released: try again
                if age > LEASE_STALE:                         # holder stopped refreshing: it was killed
                    grave = lease.with_name(f'{lease.name}.stale.{os.getpid()}.{time.time_ns()}')
                    try:
                        os.rename(lease, grave)
                    except FileNotFoundError:
                        continue
                    if self._token(grave) != stale_token or (self._lease_age(grave) or 0) <= LEASE_STALE:
                        # Another job replaced the stale lease in the meantime: put its lease back.
                        try:
                            os.link(grave, lease)
                        except OSError:
                            pass
                    grave.unlink(missing_ok=True)
                    continue
                if not blocking and time.time() > give_up:
                    return False
                if not told:
                    log(f'waiting: another job is {waiting_for or "using this lock"} in this --root '
                        f'(or one was killed less than {LEASE_STALE / 60:.0f} min ago); checking again regularly ...')
                    told = True
                time.sleep(min(15.0, LEASE_BEAT))
                continue
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), since=time.ctime(),
                                        token=f'{socket.gethostname()}.{os.getpid()}.{time.time_ns()}')))
            self.lease, self._stop = lease, threading.Event()
            threading.Thread(target=self._heartbeat, args=(lease, self._stop, self._token(lease)), daemon=True).start()
            if not FileLock._held_leases:
                # A scheduler stops a job with SIGTERM: turn it into a normal exit so leases are removed.
                try:
                    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
                except ValueError:                            # not in the main thread
                    pass
            FileLock._held_leases.append(self)
            return True

    @staticmethod
    def _heartbeat(lease, stop, token):
        warned = False
        while not stop.wait(LEASE_BEAT):
            if FileLock._token(lease) != token:
                # Missing for a moment (another job checking a stale lease puts it back) or, in the
                # worst case, taken over: keep trying, never refresh someone else's lease.
                if not warned and lease.exists():
                    log(f'WARNING: lock {lease.name} is now held by another job; do not run two jobs on this --root.')
                    warned = True
                continue
            try:
                os.utime(lease, None)
            except FileNotFoundError:
                continue

    def release(self):
        if self.fh is not None:
            try:
                fcntl.lockf(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None
        if self.lease is not None:
            self._stop.set()
            try:
                if json.loads(self.lease.read_text()).get('pid') == os.getpid():
                    self.lease.unlink()
            except (OSError, ValueError):
                pass
            if self in FileLock._held_leases:
                FileLock._held_leases.remove(self)
            self.lease = None

    @staticmethod
    def busy(path):
        """Is this lock held by a live job right now? (Does not wait.)"""
        path = Path(path)
        if FileLock.mode == 'lease' or (FileLock.mode is None and (fcntl is None or os.environ.get('LGEL_LOCK_MODE') == 'lease')):
            probe = FileLock(path)
            age = probe._lease_age() if probe.lease_path.exists() else None
            return age is not None and age <= LEASE_STALE
        if not path.exists():
            return False
        with open(path, 'a+') as fh:              # never called on a lock this process holds
            try:
                fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.lockf(fh, fcntl.LOCK_UN)
                return False
            except OSError as e:
                return e.errno in (errno.EACCES, errno.EAGAIN)


def _release_leases():
    for lock in list(FileLock._held_leases):
        lock.release()


atexit.register(_release_leases)


def busy_runs(p):
    """Names of runs whose lock is held by a live job right now."""
    if not p.state.exists():
        return []
    names = {f.name[len('run.'):].removesuffix('.lease').removesuffix('.lock')
             for f in p.state.glob('run.*.lock*') if '.stale.' not in f.name}
    return sorted(n for n in names if FileLock.busy(p.state / f'run.{n}.lock'))


def input_fingerprint(p):
    """Hashes of every prepared file a training run consumes (and the test manifest it is evaluated on)."""
    files = dict(splits=p.splits, train_triplets=p.triplet_csv('train'), val_triplets=p.triplet_csv('val'),
                 test_triplets=p.triplet_csv('test'), video_metadata=p.metadata, parsed_annotations=p.parsed)
    return {k: (file_sha(f) if f.exists() else None) for k, f in files.items()}


def check_run_inputs(p, run_name):
    """A run's checkpoints, predictions and metrics belong to the exact data they were made from.
    If the shared data was rebuilt differently since (other split seed, other fps, --force ...),
    reusing them would e.g. evaluate a model on test videos it was trained on. Refuse instead."""
    current = input_fingerprint(p)
    missing = sorted(k for k, v in current.items() if v is None)
    if missing:
        raise StageError(f'the prepared data is incomplete (missing: {missing}); run the data stages first '
                         f'(the full command, or --to-stage audit).')
    if p.run_inputs.exists():
        saved = json.loads(p.run_inputs.read_text())
        changed = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
        produced = (p.run.exists() and any(p.run.glob('*.pth'))) or \
            (p.results.exists() and any(p.results.glob('proposed_*')))
        if changed and not produced:
            log(f'run "{run_name}" had not produced anything yet; recording the current data for it')
            write_json(p.run_inputs, current)
            return
        if changed:
            raise StageError(
                f'the prepared data changed since run "{run_name}" was started (changed: {changed}). Its '
                f'checkpoints and results belong to the old data and must not be reused or evaluated on the new '
                f'data. Start a new run with a different --run-name (or use a new --root), or restore the '
                f'original data settings. (Run folders: {p.run}, {p.results}.)')
        return
    checkpoints = sorted(p.run.glob('*.pth')) if p.run.exists() else []
    if checkpoints:
        # A run started before this check existed: compare with the provenance train.py stored.
        import torch
        ck = torch.load(checkpoints[0], map_location='cpu', weights_only=False)
        prov = ck.get('provenance') or {}
        pairs = dict(TRAIN_TRIPLETS_CSV_PATH='train_triplets', VAL_TRIPLETS_CSV_PATH='val_triplets',
                     CHOLEC80_PARSED_ANNOTATIONS='parsed_annotations', VIDEO_METADATA_PATH='video_metadata')
        bad = [ours for theirs, ours in pairs.items() if str(prov.get(theirs, ''))[:16] != current[ours]]
        if bad:
            raise StageError(f'run "{run_name}" was trained on different data ({bad}); use a new --run-name.')
    write_json(p.run_inputs, current)


GPU_CHECK = r"""
import sys, torch
d = torch.device('cuda')
x = torch.randn(512, 512, device=d, requires_grad=True)
(x @ x).relu().sum().backward()
conv = torch.nn.Conv3d(3, 8, (1, 16, 16), stride=(1, 16, 16)).to(d)
conv(torch.randn(2, 3, 2, 32, 32, device=d)).sum().backward()
with torch.autocast('cuda', dtype=torch.bfloat16 if sys.argv[1] == 'bf16' else torch.float16):
    y = torch.nn.functional.scaled_dot_product_attention(*(torch.randn(2, 4, 64, 32, device=d),) * 3)
float(y.float().sum())
torch.cuda.synchronize()
print('GPU check OK:', torch.cuda.get_device_name(0), 'compute capability', torch.cuda.get_device_capability(0),
      '| this PyTorch build supports', torch.cuda.get_arch_list())
"""


def pin_single_gpu(count):
    """The code trains on one GPU. If the job sees several, expose only the first to every step, so
    the run (whose checkpoints record the visible-device count for exact resume) can be resumed by a
    job that sees one GPU, and vice versa."""
    if count <= 1:
        return None
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    first = visible.split(',')[0].strip() if visible else '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = first
    return first


# --------------------------------------------------------------------------- settings
def resolve_settings(a):
    preset = PRESETS[a.preset]
    for key, default in preset.items():
        if getattr(a, key, None) is None:
            setattr(a, key, default)
    a.workers = a.workers or min(16, max(1, cpu_count()))
    if a.run_name is None:
        a.run_name = f'{a.preset}_seed{a.seed}'
    for attr in ('data_zip', 'data_dir', 'weights_path', 'splits_json'):
        if getattr(a, attr):
            setattr(a, attr, str(Path(getattr(a, attr)).expanduser().resolve()))
    if Path(a.text_model).expanduser().exists():                  # a local directory, not a HF model id
        a.text_model = str(Path(a.text_model).expanduser().resolve())
    a.data_url = a.data_url or os.environ.get('LGEL_DATA_URL')
    a.weights_url = a.weights_url or os.environ.get('LGEL_WEIGHTS_URL')
    return a


def data_source_id(a):
    # Deliberately NOT the URL/path: download links often expire or carry a token, and a resumed or
    # --offline job must not be rejected just because the link is different or absent.
    # (To switch to a different dataset, use a fresh --root.)
    return sha(f'videos<={a.max_videos}|synthetic={a.synthetic or 0}')


def stage_signature(stage, a):
    """Parameters that define a stage's output. A finished stage whose signature changed is an
    error (it would silently mix settings); use a fresh --root or --force to redo it."""
    sigs = {
        'fetch_models': dict(text_model=a.text_model),
        'fetch_weights': dict(random_init=a.random_init, text_model=a.text_model),
        'fetch_data': dict(src=data_source_id(a)),
        'extract_frames': dict(fps=a.sample_fps, data=data_source_id(a)),
        'sanitize_annotations': dict(max_extra=a.max_phantom_rows, data=data_source_id(a)),
        'parse_annotations': dict(data=data_source_id(a)),
        'splits': dict(seed=a.split_seed, val=a.val_ratio, test=a.test_ratio,
                       given=sha(file_sha(a.splits_json) if a.splits_json else '')),
        'triplets': dict(fps=a.sample_fps),
        'audit': dict(mode=a.audit),
        'train': dict(run=a.run_name),
        'predict': dict(run=a.run_name), 'evaluate': dict(run=a.run_name),
    }
    return sigs.get(stage, {})


# --------------------------------------------------------------------------- stage: preflight
def stage_preflight(a, p, selected):
    try:
        import torch
    except Exception as e:                                      # noqa: BLE001
        raise StageError(f'PyTorch is not importable ({e}): the environment is missing or incomplete. '
                         f'Run `bash setup_env.sh` on a machine with internet.')
    info = dict(host=platform.node(), python=sys.version.split()[0], platform=platform.platform(),
                cpus=cpu_count(), torch=torch.__version__, cuda=torch.cuda.is_available(), gpus=[])
    if info['cuda']:
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            info['gpus'].append(dict(name=pr.name, memory_gb=round(gb(pr.total_memory), 1)))
        info['bf16_native'] = native_bf16()
        pinned = pin_single_gpu(torch.cuda.device_count())
        if pinned is not None:
            info['pinned_cuda_visible_devices'] = pinned
            log(f'{torch.cuda.device_count()} GPUs visible; this code uses one, so every step will see only '
                f'CUDA_VISIBLE_DEVICES={pinned} (request a single GPU to avoid reserving idle ones)')
    free = gb(shutil.disk_usage(p.root).free)
    info['free_disk_gb'] = round(free, 1)
    for mod in ['cv2', 'pandas', 'sklearn', 'transformers', 'einops', 'PIL']:
        try:
            __import__(mod)
        except Exception as e:                                  # noqa: BLE001
            raise StageError(f'Python package {mod!r} is not importable ({e}). Run `bash setup_env.sh`.')
    log('preflight: ' + json.dumps(info))
    problems = []
    if 'train' in selected and not info['cuda'] and not a.allow_cpu:
        problems.append('no CUDA GPU is visible (request one from the scheduler, or pass --allow-cpu for a toy run)')
    need, what = disk_needed_gb(a, p, selected)
    if need and free < need:
        problems.append(f'only {free:.0f} GB free under {p.root}, but {what} needs about {need:.0f} GB '
                        f'(override with --min-free-gb if you are sure)')
    if any(s in selected for s in ('fetch_data',)) and not (a.data_url or a.data_zip or a.data_dir or a.synthetic) \
            and not p.marker('fetch_data').exists():
        problems.append('no dataset source: pass --data-url / --data-zip / --data-dir (or set LGEL_DATA_URL)')
    if 'fetch_weights' in selected and not (a.weights_url or a.weights_path or a.random_init) \
            and not p.marker('fetch_weights').exists():
        problems.append('no pretrained backbone: pass --weights-path / --weights-url (or LGEL_WEIGHTS_URL), '
                        'or --random-init to deliberately train the backbone from scratch')
    if problems:
        raise StageError('preflight failed:\n  - ' + '\n  - '.join(problems))
    if info['cuda'] and any(s in selected for s in ('train', 'predict')):
        # Seeing a GPU is not the same as being able to run on it: run real kernels now, in seconds,
        # instead of failing after hours of data preparation.
        try:
            run([sys.executable, '-c', GPU_CHECK, 'bf16' if native_bf16() else 'fp16'], a, p, 'gpu_check')
        except StageError as e:
            raise StageError(f'the GPU cannot run PyTorch kernels with this environment ({e}). Typical causes: a '
                             f'GPU newer than the installed PyTorch build (RTX 50xx / Blackwell needs PyTorch >= 2.7 '
                             f'with CUDA 12.8; this environment pins 2.5.1 + CUDA 12.4), or an NVIDIA driver too old '
                             f'for CUDA 12.4. See logs/gpu_check.log.')
    write_json(p.logs / 'preflight.json', info)
    return info


# --------------------------------------------------------------------------- stage: fetch_models
def stage_fetch_models(a, p):
    code = ('import sys; from transformers import AutoTokenizer, CLIPTextModel; n=sys.argv[1]; '
            'AutoTokenizer.from_pretrained(n); m=CLIPTextModel.from_pretrained(n); '
            'print("text encoder ready:", n, "hidden", m.config.hidden_size)')
    env = stage_env(a, p)
    if a.offline:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    run([sys.executable, '-c', code, a.text_model], a, p, 'fetch_models', env=env)


# --------------------------------------------------------------------------- downloading
def _curl_supports(flag):
    try:
        out = subprocess.run(['curl', '--help', 'all'], capture_output=True, text=True).stdout
        return flag in out
    except Exception:                                           # noqa: BLE001
        return False


def remote_size(url):
    """Content-Length of the final (post-redirect) response, or None if the server does not say."""
    try:
        out = subprocess.run(['curl', '-sIL', '--max-time', '60', url], capture_output=True, text=True).stdout
    except Exception:                                           # noqa: BLE001
        return None
    sizes = re.findall(r'(?im)^content-length:\s*(\d+)', out)
    return int(sizes[-1]) if sizes else None


def download(url, dest, a, p, name):
    """Resumable download. Safe to call again after an interruption: a finished file (recorded in
    <file>.complete) is reused, a partial one is continued. Never touches the network with --offline."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    complete = dest.with_name(dest.name + '.complete')
    if dest.exists() and complete.exists():
        try:
            recorded = json.loads(complete.read_text()).get('size')
        except (OSError, ValueError):
            recorded = None
        if recorded == dest.stat().st_size:
            log(f'{dest.name} already downloaded ({gb(recorded):.2f} GB)')
            return
    complete.unlink(missing_ok=True)
    if a.offline:
        raise StageError(f'--offline: {dest.name} has not been (completely) downloaded yet and the network may not be '
                         f'used. Run the same command once without --offline on a machine with internet, adding '
                         f'--to-stage fetch_data, or give a local file (--weights-path / --data-zip / --data-dir).')
    if not url:
        raise StageError(f'{dest.name} is missing and no download link was given.')
    _download(url, dest, a, p, name)
    with open(dest, 'rb') as f:
        head = f.read(1024).lstrip().lower()
    if head.startswith((b'<!doctype html', b'<html', b'<?xml', b'<head', b'<body')):
        dest.unlink()
        raise StageError(f'the link for {dest.name} returned a web page, not the file (a share page, a login page or an '
                         f'expired link). Give a direct download link and run the same command again.')
    write_json(complete, dict(size=dest.stat().st_size, finished=time.ctime()))


def discard_download(path, keep=True):
    """Forget a downloaded file that turned out to be unusable, so the next run downloads it again.
    Small files are kept aside for inspection; huge ones (keep=False) are deleted to free the space."""
    path = Path(path)
    path.with_name(path.name + '.complete').unlink(missing_ok=True)
    if path.exists() and not keep:
        path.unlink()
        return None
    if path.exists():
        aside = path.with_name(f'{path.name}.rejected-{time.strftime("%Y%m%d-%H%M%S")}')
        os.replace(path, aside)
        return aside
    return None


def _download(url, dest, a, p, name):
    if any(h in url for h in ('drive.google.com', 'docs.google.com')):
        if shutil.which('gdown') is None and subprocess.run([sys.executable, '-m', 'gdown', '--version'],
                                                            capture_output=True).returncode != 0:
            raise StageError('This is a Google Drive link and needs `gdown` (pip install gdown).')
        run([sys.executable, '-m', 'gdown', '--fuzzy', url, '-O', dest], a, p, name, progress_path=dest)
        return
    if shutil.which('curl'):
        total = remote_size(url)
        if dest.exists() and total is not None and dest.stat().st_size == total:
            log(f'{dest.name} already fully downloaded ({gb(total):.2f} GB)')
            return
        if total is not None:
            log(f'downloading {gb(total):.2f} GB -> {dest}')
        base = ['curl', '-fL', '--retry', '20', '--retry-delay', '15', '--connect-timeout', '60']
        if _curl_supports('--retry-all-errors'):
            base += ['--retry-all-errors']
        try:
            run(base + ['-C', '-', '-o', dest, url], a, p, name, progress_path=dest)
        except StageError:
            # A server without byte-range support cannot resume: start over rather than trust a partial file.
            if dest.exists() and (total is None or dest.stat().st_size != total):
                log('resume failed (server without range support?); restarting the download from zero')
                dest.unlink()
                run(base + ['-o', dest, url], a, p, name, progress_path=dest)
            elif not dest.exists():
                raise
        if total is not None and dest.stat().st_size != total:
            raise StageError(f'Downloaded {dest.stat().st_size} bytes but the server announced {total}. '
                             f'Run the same command again to continue.')
    elif shutil.which('wget'):
        run(['wget', '-c', '--tries=20', '--waitretry=15', '-O', dest, url], a, p, name, progress_path=dest)
    else:
        raise StageError('Neither curl, wget nor gdown is available to download files.')


def stage_fetch_weights(a, p):
    weights = ''
    if a.random_init:
        log('WARNING: --random-init: the vision backbone will be trained from scratch. '
            'Results are NOT comparable to a run that starts from the pretrained M2CRL weights.')
    elif a.weights_path:
        if not Path(a.weights_path).is_file():
            raise StageError(f'--weights-path not found: {a.weights_path}')
        weights = str(Path(a.weights_path).resolve())
    else:
        download(a.weights_url, p.weights, a, p, 'fetch_weights')   # continues a partial file; instant if complete
        weights = str(p.weights)
    # Build the real model once: this is the only place a wrong checkpoint would surface, and it
    # is far better to learn that now than after hours of frame extraction.
    cfg = dict(MODEL=dict(M2CRL_WEIGHTS_PATH=weights, TEXT_ENCODER_MODEL=a.text_model))
    p.tmp.mkdir(parents=True, exist_ok=True)
    cfg_path = p.tmp / 'check_model.json'
    write_json(cfg_path, cfg)
    try:
        run([sys.executable, __file__, '_check_model', cfg_path], a, p, 'fetch_weights')
    except StageError as e:
        if not a.weights_path and getattr(e, 'rc', None) == CHECKPOINT_REJECTED:   # our download, really unusable
            aside = discard_download(p.weights)
            raise StageError(f'{e}\n   The downloaded checkpoint does not load (moved to {aside.name if aside else "-"}). '
                             f'Check the weights link (it must point to the M2CRL checkpoint file itself) and run the '
                             f'same command again.')
        raise
    # Remember where the checkpoint is, so a later invocation (e.g. the offline compute-node job,
    # which is not given --weights-path/--weights-url again) still finds it.
    write_json(p.weights_ref, dict(path=weights, random_init=bool(a.random_init)))
    return weights


def _check_model(cfg_path):
    """Internal: construct the model exactly as train.py will (validates checkpoint + text encoder)."""
    from project_config import config
    _apply_overrides(config, json.loads(Path(cfg_path).read_text()))
    from models import LocalizationFramework
    try:
        LocalizationFramework(config, initialize_backbone=True)
    except RuntimeError as e:
        # Only a checkpoint the loader actually rejects counts as a bad file; running out of memory,
        # being killed, or a text-encoder problem must not get a good download thrown away.
        if str(e).startswith('Found pretrained checkpoint') and not isinstance(e.__cause__, MemoryError):
            print(f'CHECKPOINT REJECTED: {e}', flush=True)
            sys.exit(CHECKPOINT_REJECTED)
        raise
    print('model construction OK (backbone checkpoint and text encoder both load)')


def _apply_overrides(obj, values):
    for k, v in values.items():                                 # same rule as train.py: unknown keys fail
        if not hasattr(obj, k):
            raise ValueError(f'Unknown configuration field: {k}')
        if isinstance(v, dict):
            _apply_overrides(getattr(obj, k), v)
        else:
            setattr(obj, k, v)


def read_weights_path(p, a):
    if a.random_init:
        return ''
    if a.weights_path:
        return str(Path(a.weights_path).resolve())
    if p.weights_ref.exists():
        ref = json.loads(p.weights_ref.read_text())
        if ref.get('path'):
            return ref['path']
    return str(p.weights)


# --------------------------------------------------------------------------- stage: fetch_data
def find_inventory(base, max_videos):
    base = Path(base)
    def collect(pattern, suffix):
        out = {}
        for f in base.rglob(pattern):
            if f.name.startswith('._') or '__MACOSX' in f.parts:
                continue
            key = f.name[: -len(suffix)]
            if key in out:
                raise StageError(f'Duplicate {pattern} for {key}: {out[key]} and {f}')
            out[key] = f
        return out
    phases, tools = collect('*-phase.txt', '-phase.txt'), collect('*-tool.txt', '-tool.txt')
    videos = collect('*.mp4', '.mp4')                       # duplicates are an error, as for annotations
    if not (phases and tools and videos):
        raise StageError(f'Could not find the dataset under {base}: found {len(videos)} .mp4, {len(phases)} '
                         f'*-phase.txt, {len(tools)} *-tool.txt files.')
    ids = set(phases) & set(tools) & set(videos)
    lost = (set(phases) | set(tools) | set(videos)) - ids
    if lost:
        raise StageError(f'Videos without a complete set of video/phase/tool files: {sorted(lost, key=natural_key)}')
    chosen = sorted(ids, key=natural_key)[: max_videos]
    return {v: dict(video=str(videos[v]), phase=str(phases[v]), tool=str(tools[v])) for v in chosen}


def stage_fetch_data(a, p):
    if a.synthetic:
        base = p.raw / 'synthetic'
        if not base.exists():
            partial = p.raw / 'synthetic.partial'
            shutil.rmtree(partial, ignore_errors=True)
            log(f'--synthetic: generating {a.synthetic} FAKE videos (plumbing test; numbers will be meaningless)')
            run([sys.executable, REPO / 'tools' / 'make_synthetic_cholec80.py', '--out', partial,
                 '--videos', a.synthetic, '--phantom-rows'], a, p, 'fetch_data')
            os.replace(partial, base)
    elif a.data_dir:
        base = Path(a.data_dir).resolve()
        if not base.is_dir():
            raise StageError(f'--data-dir not found: {base}')
    else:
        base = p.extracted
        if not base.exists():
            zpath = Path(a.data_zip).resolve() if a.data_zip else p.zip
            if not a.data_zip:
                download(a.data_url, zpath, a, p, 'fetch_data')
            log(f'verifying {zpath.name} ({gb(zpath.stat().st_size):.1f} GB) ...')
            partial = p.raw / 'extracted.partial'
            shutil.rmtree(partial, ignore_errors=True)
            try:
                if not zipfile.is_zipfile(zpath):
                    raise zipfile.BadZipFile('not a zip archive')
                if not a.skip_zip_check:
                    with zipfile.ZipFile(zpath) as z:
                        bad = z.testzip()
                    if bad:
                        raise zipfile.BadZipFile(f'CRC error in {bad}')
                log(f'extracting to {p.extracted} ...')
                with zipfile.ZipFile(zpath) as z:
                    names = z.namelist()
                    for i, n in enumerate(names, 1):
                        z.extract(n, partial)
                        if i % 20 == 0 or i == len(names):
                            log(f'  extracted {i}/{len(names)} files')
            except NotImplementedError as e:                  # valid archive, method Python cannot read
                shutil.rmtree(partial, ignore_errors=True)
                kept = ''
                if not a.data_zip:                            # keep it for manual unzipping, but fetch a new one next time
                    aside = discard_download(zpath)
                    kept = f' The downloaded file was kept as {aside}.' if aside else ''
                raise StageError(f'the dataset archive uses a compression method Python cannot unpack ({e}), e.g. '
                                 f'Deflate64 from Windows "compressed folders". Re-create the zip with standard '
                                 f'compression, or unzip it (e.g. with 7-Zip) and pass the folder with --data-dir.{kept}')
            except (zipfile.BadZipFile, zlib.error, EOFError) as e:
                shutil.rmtree(partial, ignore_errors=True)
                if a.data_zip:
                    raise StageError(f'{zpath} is not a usable zip archive ({e}).')
                discard_download(zpath, keep=False)            # 70 GB: make room for the new download
                raise StageError(f'the downloaded dataset archive is corrupt or is not a zip ({e}); it was discarded. '
                                 f'Run the same command again to download it again; if this repeats, check that the '
                                 f'data link points to the zip file itself.')
            os.replace(partial, p.extracted)
            if not a.data_zip and not a.keep_zip:
                zpath.unlink(missing_ok=True)
                zpath.with_name(zpath.name + '.complete').unlink(missing_ok=True)
                log('deleted the downloaded zip to save space (use --keep-zip to keep it)')
    inv = find_inventory(base, a.max_videos)
    write_json(p.inventory, dict(base=str(base), videos=inv))
    log(f'dataset located: {len(inv)} videos under {base}')


def load_inventory(p):
    if not p.inventory.exists():
        raise StageError('Dataset inventory missing: run the fetch_data stage first.')
    return json.loads(p.inventory.read_text())['videos']


# --------------------------------------------------------------------------- stage: extract_frames
def _worker_init():
    """Extraction workers stop on SIGTERM like any process (the parent's handler is not for them)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _extract_one(job):
    """Top-level (picklable) worker: reuse build_dataset.extract on a one-video folder."""
    src, frames_dir, shard, fps, tmp_root = job
    import cv2
    cv2.setNumThreads(1)
    from dataset_preprocessing.build_dataset import extract
    t0 = time.time()
    video_id = 'CHOLEC80__' + Path(src).stem
    dest = Path(frames_dir) / video_id
    Path(shard).unlink(missing_ok=True)                          # the old frames are about to be replaced
    shutil.rmtree(dest, ignore_errors=True)                      # leftovers of an interrupted run
    tmp = Path(tempfile.mkdtemp(dir=tmp_root))
    try:
        link = tmp / Path(src).name
        try:
            os.symlink(src, link)
        except OSError:
            os.link(src, link)
        shard_tmp = Path(str(shard) + '.tmp')
        shard_tmp.unlink(missing_ok=True)
        extract(tmp, frames_dir, shard_tmp, fps)
        os.replace(shard_tmp, shard)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return video_id, time.time() - t0


def missing_frames(meta, folder):
    """How many of the frames a video's metadata promises are not on disk (one directory listing)."""
    from data_contract import sample_indices
    try:
        wanted = {f'frame_{int(i):07d}.jpg' for i in sample_indices(meta['frame_count'], meta['source_fps'], meta['sample_fps'])}
        present = {e.name for e in os.scandir(folder)}
    except (KeyError, ValueError, OSError):
        return -1                                         # unreadable metadata or folder: redo the video
    return len(wanted - present)


def stage_extract_frames(a, p):
    inv = load_inventory(p)
    p.shards.mkdir(parents=True, exist_ok=True)
    p.tmp.mkdir(parents=True, exist_ok=True)
    p.frames.mkdir(parents=True, exist_ok=True)
    todo, missing_raw = [], []
    for v, files in inv.items():
        shard = p.shards / f'CHOLEC80__{v}.json'
        if shard.exists() and (p.frames / f'CHOLEC80__{v}').is_dir():
            try:
                done_fps = json.loads(shard.read_text())[f'CHOLEC80__{v}'].get('sample_fps')
            except (OSError, ValueError, KeyError, AttributeError):
                done_fps = None
            if done_fps == a.sample_fps:
                missing = missing_frames(json.loads(shard.read_text())[f'CHOLEC80__{v}'], p.frames / f'CHOLEC80__{v}')
                if not missing:
                    continue
                log(f'  {v}: {missing} frame file(s) are missing (deleted or purged?): extracting again')
            else:
                log(f'  {v}: existing frames were sampled at {done_fps} fps, now {a.sample_fps} fps: extracting again')
        if not Path(files['video']).exists():
            missing_raw.append(v)
            continue
        todo.append((files['video'], str(p.frames), str(shard), a.sample_fps, str(p.tmp)))
    if missing_raw:
        raise StageError(f'the raw videos needed to (re)extract frames are gone: {missing_raw[:5]}'
                         f'{" ..." if len(missing_raw) > 5 else ""} (deleted by --cleanup-raw?). Use a fresh --root, '
                         f'or restore the videos.')
    log(f'frame extraction: {len(inv) - len(todo)} videos already done, {len(todo)} to do, '
        f'{min(a.workers, max(1, len(todo)))} parallel workers')
    failures = []
    if todo:
        pool = cf.ProcessPoolExecutor(max_workers=min(a.workers, len(todo)), initializer=_worker_init)
        try:
            futures = {pool.submit(_extract_one, j): j for j in todo}
            for n, fut in enumerate(cf.as_completed(futures), 1):
                job = futures[fut]
                try:
                    vid, secs = fut.result()
                    log(f'  [{n}/{len(todo)}] {vid} done in {secs:.0f}s')
                except Exception as e:                          # noqa: BLE001
                    failures.append((Path(job[0]).name, repr(e)))
                    log(f'  [{n}/{len(todo)}] FAILED {Path(job[0]).name}: {e!r}')
        except BaseException:                                   # e.g. SIGTERM from the scheduler: stop now
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
    if failures:
        raise StageError('frame extraction failed for: ' + '; '.join(f'{n}: {e}' for n, e in failures)
                         + '. Re-run the same command to retry only these videos.')
    merged = {}
    for v in inv:
        merged.update(json.loads((p.shards / f'CHOLEC80__{v}.json').read_text()))
    if set(merged) != {f'CHOLEC80__{v}' for v in inv}:
        raise StageError('metadata shards do not cover the inventory')
    write_json(p.metadata, merged)
    total = sum(m['frame_count'] for m in merged.values())
    log(f'video_metadata.json written: {len(merged)} videos, {total:,} source frames')
    if a.cleanup_raw and not a.data_dir and not a.data_zip:
        freed = 0
        for files in inv.values():                               # videos only: the annotation files are still needed
            v = Path(files['video'])
            if v.exists() and p.extracted in v.parents:
                freed += v.stat().st_size
                v.unlink()
        log(f'--cleanup-raw: deleted the raw .mp4 videos, freeing {gb(freed):.1f} GB (frames + metadata are all '
            f'later stages need)')


# --------------------------------------------------------------------------- stage: sanitize_annotations
def stage_sanitize_annotations(a, p):
    """Copy annotations to a clean directory, dropping trailing rows that point past the end of the
    video. Official Cholec80 has this defect in a few phase files (one phantom row that repeats the
    previous phase). Anything larger than --max-phantom-rows, or a phantom row that changes the
    phase, is treated as real corruption and stops the run. Originals are never modified."""
    import pandas as pd
    meta = json.loads(p.metadata.read_text())
    inv = load_inventory(p)
    shutil.rmtree(p.clean, ignore_errors=True)
    (p.clean / 'phase').mkdir(parents=True)
    (p.clean / 'tool').mkdir(parents=True)
    report = {}
    for v, files in inv.items():
        count = meta[f'CHOLEC80__{v}']['frame_count']
        for kind, key in (('phase', 'phase'), ('tool', 'tool')):
            df = pd.read_csv(files[key], sep=r'\s+')
            if 'Frame' not in df.columns:
                raise StageError(f'{files[key]}: no "Frame" column (columns: {list(df.columns)})')
            beyond = df[df['Frame'] >= count]
            if len(beyond):
                if len(beyond) > a.max_phantom_rows:
                    raise StageError(f'{files[key]}: {len(beyond)} rows beyond the last frame ({count - 1}); '
                                     f'more than --max-phantom-rows={a.max_phantom_rows}, refusing to guess.')
                kept = df[df['Frame'] < count]
                if kind == 'phase':
                    last_phase = kept['Phase'].iloc[-1]
                    if (beyond['Phase'] != last_phase).any():
                        raise StageError(f'{files[key]}: rows beyond the video end change the phase '
                                         f'({last_phase} -> {list(beyond["Phase"].unique())}); not a benign defect.')
                report.setdefault(v, {})[kind] = dict(dropped_rows=len(beyond), frame_count=count,
                                                      dropped_frames=beyond['Frame'].tolist())
                df = kept
            df.to_csv(p.clean / kind / f'{v}-{kind}.txt', sep='\t', index=False)
    write_json(p.data / 'annotations_sanitized.json', report)
    log(f'annotations copied to {p.clean}; repaired {len(report)} video(s): {sorted(report, key=natural_key)}')


# --------------------------------------------------------------------------- label/manifest stages
def stage_parse_annotations(a, p):
    from dataset_preprocessing.build_dataset import parse_annotations
    p.parsed.unlink(missing_ok=True)
    parse_annotations(p.clean / 'phase', p.clean / 'tool', p.parsed)
    log(f'parsed annotations -> {p.parsed}')


def stage_splits(a, p):
    p.splits.unlink(missing_ok=True)
    if a.splits_json:
        split = json.loads(Path(a.splits_json).read_text())
        if set(split) != {'train', 'val', 'test'}:
            raise StageError('--splits-json needs exactly train/val/test lists')
        known = set(json.loads(p.metadata.read_text()))
        unknown = {v for vs in split.values() for v in vs} - known
        if unknown:
            raise StageError(f'--splits-json mentions videos that are not in the data: {sorted(unknown)[:5]} ...')
        if a.max_videos:
            split = {k: [v for v in vs if v in known] for k, vs in split.items()}
        origin = f'given: {a.splits_json}'
    else:
        import random
        ids = sorted(json.loads(p.metadata.read_text()))
        random.Random(a.split_seed).shuffle(ids)
        n_val, n_test = max(1, round(len(ids) * a.val_ratio)), max(1, round(len(ids) * a.test_ratio))
        if len(ids) - n_val - n_test < 1:
            raise StageError(f'{len(ids)} videos are too few for a train/val/test split')
        split = dict(train=ids[: len(ids) - n_val - n_test], val=ids[len(ids) - n_val - n_test: len(ids) - n_test],
                     test=ids[len(ids) - n_test:])
        origin = f'new random split, seed {a.split_seed} (NOT the historical dissertation split)'
    write_json(p.splits, split)
    write_json(p.splits.with_suffix('.provenance.json'), dict(origin=origin, sizes={k: len(v) for k, v in split.items()}))
    log(f'splits: {({k: len(v) for k, v in split.items()})} [{origin}]')


def stage_triplets(a, p):
    from dataset_preprocessing.build_dataset import build
    shutil.rmtree(p.triplets, ignore_errors=True)
    build(p.parsed, p.metadata, p.frames, p.splits, p.triplets, a.sample_fps)
    log(f'triplet manifests written to {p.triplets}')


def stage_audit(a, p):
    p.audit.unlink(missing_ok=True)
    cmd = [sys.executable, 'audit_data.py', '--train', p.triplet_csv('train'), '--val', p.triplet_csv('val'),
           '--test', p.triplet_csv('test'), '--annotations', p.parsed, '--metadata', p.metadata,
           '--frames', p.frames, '--output', p.audit, '--sample-fps', a.sample_fps]
    if a.audit == 'all':
        cmd.append('--decode-all')
    run(cmd, a, p, 'audit')


# --------------------------------------------------------------------------- stage: train
def train_config(a, p, workers, amp_dtype):
    return dict(
        EXTRACTED_FRAMES_DIR=str(p.frames), VIDEO_METADATA_PATH=str(p.metadata),
        CHOLEC80_PARSED_ANNOTATIONS=str(p.parsed),
        TRAIN_TRIPLETS_CSV_PATH=str(p.triplet_csv('train')), VAL_TRIPLETS_CSV_PATH=str(p.triplet_csv('val')),
        TEST_TRIPLETS_CSV_PATH=str(p.triplet_csv('test')),
        CHECKPOINT_DIR=str(p.run), OUTPUT_DIR=str(p.results),
        MODEL=dict(M2CRL_WEIGHTS_PATH=read_weights_path(p, a), TEXT_ENCODER_MODEL=a.text_model),
        DATA=dict(NUM_WORKERS=workers, SAMPLE_FPS=a.sample_fps),
        TRAIN=dict(SEED=a.seed, NUM_EPOCHS=a.epochs, WARMUP_EPOCHS=a.warmup, BATCH_SIZE=a.batch_size,
                   GRADIENT_ACCUMULATION_STEPS=a.accum, SUBSET_RATIO=a.subset_ratio, AMP_DTYPE=amp_dtype))


def completed_epochs(p):
    """Epochs finished = the epoch stored in latest_model.pth (written atomically by train.py).
    The metrics journal is appended *after* that save, so it can lag behind by one row."""
    latest = p.run / 'latest_model.pth'
    if not latest.exists():
        return 0
    import torch
    return int(torch.load(latest, map_location='cpu', weights_only=False)['epoch'])


def repair_metrics_journal(p, done):
    """Make training_metrics.jsonl consistent with the checkpoint after an interrupted job: drop a
    half-written last line, rows beyond the saved epoch and replayed duplicates (the last one wins).
    Returns the epochs <= done that have no row (job killed between checkpoint save and append)."""
    f = p.run / 'training_metrics.jsonl'
    if not f.exists():
        return list(range(1, done + 1))
    original = f.read_text()
    rows = {}
    for line in original.splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get('epoch'), int) and 1 <= r['epoch'] <= done:
            rows[r['epoch']] = r
    text = ''.join(json.dumps(rows[e]) + '\n' for e in sorted(rows))
    if text != original:
        backup = f.with_name(f'{f.name}.before-repair-{time.strftime("%Y%m%d-%H%M%S")}')
        shutil.copy2(f, backup)
        tmp = f.with_name(f'{f.name}.{os.getpid()}.tmp')
        tmp.write_text(text)
        os.replace(tmp, f)
        log(f'repaired {f.name} after an interrupted job (original kept as {backup.name})')
    missing = [e for e in range(1, done + 1) if e not in rows]
    if missing:
        log(f'note: no metrics row for epoch(s) {missing} (the job stopped between saving the checkpoint and '
            f'logging); the checkpoints themselves are complete')
    return missing


def recover_unresumable_run(p):
    """No latest_model.pth: the job died before the first epoch finished saving. Nothing can be
    resumed, so the run starts again; a folder holding checkpoint files is kept aside, never deleted."""
    if (p.run / 'latest_model.pth').exists() or not p.run.exists():
        return None
    if any(p.run.rglob('*.pth')):
        aside = p.run.with_name(f'{p.run.name}.interrupted-{time.strftime("%Y%m%d-%H%M%S")}')
        os.replace(p.run, aside)
        log(f'{p.run.name} was interrupted during its first epoch; restarting it (partial files kept in {aside.name})')
        return aside
    shutil.rmtree(p.run)
    return None


def stage_train(a, p):
    check_run_inputs(p, a.run_name)
    amp = a.amp_dtype
    if amp == 'auto':
        amp = 'bf16' if native_bf16() else 'fp16'
    workers = a.train_workers if a.train_workers is not None else min(8, max(0, cpu_count() - 1))
    cfg = train_config(a, p, workers, amp)
    if p.run_cfg.exists():
        saved = json.loads(p.run_cfg.read_text())
        # node-dependent auto choices are frozen at first launch so a resume on another node still matches
        if a.train_workers is None:
            cfg['DATA']['NUM_WORKERS'] = saved['DATA']['NUM_WORKERS']
        if a.amp_dtype == 'auto':
            cfg['TRAIN']['AMP_DTYPE'] = saved['TRAIN']['AMP_DTYPE']
        if cfg != saved:
            diff = sorted(k for k in set(cfg) | set(saved) if cfg.get(k) != saved.get(k))
            if p.run.exists() and any(p.run.rglob('*.pth')):
                raise StageError(f'Run "{a.run_name}" already exists with different settings ({diff}). '
                                 f'Use a different --run-name, or restore the original flags to resume it.')
            # Nothing trained yet (e.g. the first attempt ran out of GPU memory and you lowered
            # --batch-size): there is nothing to protect, so take the new settings.
            log(f'settings changed since the last launch ({diff}) but no checkpoint exists yet: using the new settings')
            write_json(p.run_cfg, cfg)
    else:
        write_json(p.run_cfg, cfg)
    recover_unresumable_run(p)
    done = completed_epochs(p)
    repair_metrics_journal(p, done)
    if done >= cfg['TRAIN']['NUM_EPOCHS']:
        log(f'training already finished ({done} epochs)')
        return
    cmd = [sys.executable, 'train.py', '--config', p.run_cfg]
    latest = p.run / 'latest_model.pth'
    if latest.exists():
        log(f'resuming from {latest} (epochs finished so far: {done}/{cfg["TRAIN"]["NUM_EPOCHS"]})')
        cmd += ['--resume_from', latest]
    run(cmd, a, p, 'train', on_line=_epoch_progress(cfg['TRAIN']['NUM_EPOCHS']))
    done = completed_epochs(p)
    repair_metrics_journal(p, done)
    if done < cfg['TRAIN']['NUM_EPOCHS']:
        raise StageError('train.py ended before the configured number of epochs.')


def _epoch_progress(total):
    def on_line(line):
        if line.startswith('{"epoch"'):
            try:
                r = json.loads(line)
                log(f'epoch {r["epoch"]}/{total} done: train {r.get("train_seconds", 0) / 60:.1f} min, validation '
                    f'{r.get("val_seconds", 0) / 60:.1f} min, val NLL {r.get("val_frame_nll")}')
            except (ValueError, KeyError, TypeError):
                pass
    return on_line


# --------------------------------------------------------------------------- predict / evaluate
def stage_predict(a, p):
    check_run_inputs(p, a.run_name)
    ckpt = p.run / 'best_model.pth'
    if not ckpt.exists():
        raise StageError(f'{ckpt} not found; the train stage must finish first.')
    p.results.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(p.run_cfg.read_text())
    for split in ('val', 'test'):
        out = p.results / f'proposed_{split}.csv'
        for f in (out, Path(str(out) + '.manifest.json')):
            f.unlink(missing_ok=True)
        run([sys.executable, 'predict.py', '--checkpoint', ckpt, '--triplets', p.triplet_csv(split),
             '--metadata', p.metadata, '--frames', p.frames, '--annotations', p.parsed, '--output', out,
             '--batch-size', a.pred_batch_size, '--workers', cfg['DATA']['NUM_WORKERS']], a, p, f'predict_{split}')
        ref = p.results / f'{split}_reference.csv'
        ref.unlink(missing_ok=True)
        run([sys.executable, 'build_reference.py', '--triplets', p.triplet_csv(split), '--metadata', p.metadata,
             '--frames', p.frames, '--annotations', p.parsed, '--output', ref, '--sample-fps', a.sample_fps],
            a, p, f'reference_{split}')


def stage_evaluate(a, p):
    check_run_inputs(p, a.run_name)
    calib, test = p.results / 'proposed_calibration.json', p.results / 'proposed_test_metrics.json'
    for f in (calib, test):
        f.unlink(missing_ok=True)
    run([sys.executable, 'evaluate.py', '--predictions', p.results / 'proposed_val.csv', '--reference',
         p.results / 'val_reference.csv', '--split', 'validation', '--select-threshold', '--output', calib],
        a, p, 'evaluate_val')
    run([sys.executable, 'evaluate.py', '--predictions', p.results / 'proposed_test.csv', '--reference',
         p.results / 'test_reference.csv', '--split', 'test', '--calibration', calib, '--output', test],
        a, p, 'evaluate_test')


# --------------------------------------------------------------------------- summary
def _fmt(x):
    return 'n/a' if x is None else f'{x:.4f}'


def _duration(seconds):
    return f'{seconds / 60:.0f} min' if seconds < 5400 else f'{seconds / 3600:.1f} h'


def stage_summary(a, p):
    if p.run_inputs.exists():
        check_run_inputs(p, a.run_name)                       # never report old results next to new data
    p.results.mkdir(parents=True, exist_ok=True)
    lines = [f'# Run summary: {a.run_name}', '', f'Generated {time.ctime()} on {platform.node()}.', '']
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:                                           # noqa: BLE001
        commit = ''
    summary = dict(run=a.run_name, preset=a.preset, git_commit=commit, args={k: str(v) for k, v in vars(a).items()
                                                                             if 'url' not in k})
    if p.audit.exists():
        au = json.loads(p.audit.read_text())
        lines += ['## Data', '', '| split | videos | unique video-queries | labels (-100 / 0 / 1) |', '|---|---|---|---|']
        for s in ('train', 'val', 'test'):
            d = au[s]
            c = d['sampled_labels']
            lines.append(f'| {s} | {len(d["videos"])} | {d["unique_video_queries"]} | '
                         f'{c.get("-100", 0):,} / {c.get("0", 0):,} / {c.get("1", 0):,} |')
        lines.append('')
    tm = p.run / 'training_metrics.jsonl'
    if tm.exists():
        rows = []
        for l in tm.read_text().splitlines():
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
        summary['training'] = rows
        lines += ['## Training (validation = held-out videos)', '',
                  '| epoch | train loss | val NLL | AUROC | AP | F1 @ val threshold | train min | val min |',
                  '|---|---|---|---|---|---|---|---|']
        for r in rows:
            mins = [f'{r[k] / 60:.1f}' if isinstance(r.get(k), (int, float)) else 'n/a' for k in ('train_seconds', 'val_seconds')]
            lines.append(f'| {r["epoch"]} | {_fmt(r["train_loss"])} | {_fmt(r["val_frame_nll"])} | {_fmt(r["auroc"])} | '
                         f'{_fmt(r["average_precision"])} | {_fmt(r["validation_f1"])} | {mins[0]} | {mins[1]} |')
        lines.append('')
        timed = [r for r in rows if isinstance(r.get('train_seconds'), (int, float)) and isinstance(r.get('val_seconds'), (int, float))]
        if timed and a.preset == 'pilot' and a.subset_ratio:
            tr = sorted(r['train_seconds'] for r in timed)[len(timed) // 2]
            va = sorted(r['val_seconds'] for r in timed)[len(timed) // 2]
            full = PRESETS['full']
            epoch_s = tr * full['subset_ratio'] / a.subset_ratio + va
            summary['full_run_estimate_hours'] = round(full['epochs'] * epoch_s / 3600, 1)
            lines += ['## Time estimate for the `full` preset', '',
                      f'Pilot epoch (median): {tr / 60:.1f} min training on {a.subset_ratio:.0%} of the training windows + '
                      f'{va / 60:.1f} min validation. Scaling the training part to all windows gives about '
                      f'**{_duration(epoch_s)} per full epoch** and **{_duration(full["epochs"] * epoch_s)} for '
                      f'{full["epochs"]} epochs**, plus prediction. Rough: it assumes the same GPU and file system, and '
                      f'that training time grows linearly with the number of training windows. Each job must fit at '
                      f'least one whole epoch, because training resumes from the last finished epoch.', '']
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            ep = [r['epoch'] for r in rows]
            ax[0].plot(ep, [r['train_loss'] for r in rows], marker='o', label='train loss')
            ax[0].plot(ep, [r['val_frame_nll'] for r in rows], marker='o', label='val NLL')
            ax[0].set_xlabel('epoch'); ax[0].legend()
            ax[1].plot(ep, [r['auroc'] for r in rows], marker='o', label='AUROC')
            ax[1].plot(ep, [r['average_precision'] for r in rows], marker='o', label='AP')
            ax[1].set_xlabel('epoch'); ax[1].legend()
            fig.tight_layout(); fig.savefig(p.results / 'training_curves.png', dpi=130); plt.close(fig)
        except Exception as e:                                  # noqa: BLE001
            log(f'(training curve plot skipped: {e})')
    for title, fname, key in (('Validation (threshold selected here)', 'proposed_calibration.json', 'validation'),
                              ('Test (threshold fixed from validation)', 'proposed_test_metrics.json', 'test')):
        f = p.results / fname
        if f.exists():
            m = json.loads(f.read_text())
            summary[key] = {k: v for k, v in m.items() if k not in ('per_video', 'per_query')}
            lines += [f'## {title}', '', f'frames evaluated: {m["n"]:,} (positives {m["positives"]:,}); threshold {_fmt(m["threshold"])}', '',
                      '| AP | AUROC | F1 | precision | recall | Brier | ECE |', '|---|---|---|---|---|---|---|',
                      f'| {_fmt(m["average_precision"])} | {_fmt(m["auroc"])} | {_fmt(m["f1"])} | {_fmt(m["precision"])} | '
                      f'{_fmt(m["recall"])} | {_fmt(m["brier"])} | {_fmt(m["ece"])} |', '']
            if key == 'test' and 'per_query' in m:
                lines += ['### Per query (test)', '', '| query | AP | AUROC | F1 |', '|---|---|---|---|']
                for q, d in sorted(m['per_query'].items()):
                    lines.append(f'| {q} | {_fmt(d["average_precision"])} | {_fmt(d["auroc"])} | {_fmt(d["f1"])} |')
                lines.append('')
    write_json(p.results / 'summary.json', summary)
    (p.results / 'SUMMARY.md').write_text('\n'.join(lines))
    log(f'summary written: {p.results / "SUMMARY.md"}')


# --------------------------------------------------------------------------- driver
def select_stages(a):
    if a.stages:
        chosen = [s.strip() for s in a.stages.split(',') if s.strip()]
        bad = [s for s in chosen if s not in STAGES]
        if bad:
            raise SystemExit(f'unknown stage(s): {bad}. Valid: {STAGES}')
        return [s for s in STAGES if s in chosen]
    lo = STAGES.index(a.from_stage) if a.from_stage else 0
    hi = STAGES.index(a.to_stage) if a.to_stage else len(STAGES) - 1
    return STAGES[lo: hi + 1]


def main(argv=None):
    global _LOGFILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='full')
    ap.add_argument('--root', default=os.environ.get('LGEL_ROOT'),
                    help='working directory for ALL data/checkpoints/results (put it on scratch; needs ~200 GB). '
                         'Required for the pilot/full presets (or env LGEL_ROOT).')
    ap.add_argument('--run-name', help='default: <preset>_seed<seed>')
    g = ap.add_argument_group('inputs')
    g.add_argument('--data-url', help='direct link to the Cholec80 zip (or env LGEL_DATA_URL)')
    g.add_argument('--data-zip', help='already-downloaded zip file')
    g.add_argument('--data-dir', help='already-extracted folder containing the videos + phase/tool annotations')
    g.add_argument('--synthetic', type=int, metavar='N',
                   help='instead of real data, generate N small FAKE videos in the Cholec80 layout '
                        '(plumbing test only; results are meaningless)')
    g.add_argument('--weights-url', help='direct link to the M2CRL checkpoint (or env LGEL_WEIGHTS_URL)')
    g.add_argument('--weights-path', help='existing M2CRL checkpoint file')
    g.add_argument('--random-init', action='store_true', help='deliberately start the vision backbone from random weights')
    g.add_argument('--text-model', default='openai/clip-vit-base-patch32', help='HF model id or local directory')
    g.add_argument('--splits-json', help='use an existing {"train","val","test"} split instead of a new random one')
    g.add_argument('--keep-zip', action='store_true'); g.add_argument('--skip-zip-check', action='store_true')
    g.add_argument('--cleanup-raw', action='store_true', help='delete the extracted raw videos after frame extraction')
    g = ap.add_argument_group('experiment (None = take from preset)')
    g.add_argument('--max-videos', type=int); g.add_argument('--epochs', type=int); g.add_argument('--warmup', type=int)
    g.add_argument('--subset-ratio', type=float); g.add_argument('--batch-size', type=int); g.add_argument('--accum', type=int)
    g.add_argument('--audit', choices=['all', 'sample'])
    g.add_argument('--min-free-gb', type=float)
    g.add_argument('--seed', type=int, default=42); g.add_argument('--sample-fps', type=float, default=1.0)
    g.add_argument('--split-seed', type=int, default=42); g.add_argument('--val-ratio', type=float, default=0.1)
    g.add_argument('--test-ratio', type=float, default=0.1)
    g.add_argument('--amp-dtype', choices=['auto', 'fp16', 'bf16'], default='auto')
    g.add_argument('--max-phantom-rows', type=int, default=3)
    g = ap.add_argument_group('resources')
    g.add_argument('--workers', type=int, help='parallel frame-extraction processes (default: min(16, cpus))')
    g.add_argument('--train-workers', type=int, help='DataLoader workers (default: min(8, cpus-1))')
    g.add_argument('--pred-batch-size', type=int, default=8)
    g.add_argument('--allow-cpu', action='store_true'); g.add_argument('--offline', action='store_true',
                   help='never touch the network (needs fetch stages done earlier)')
    g = ap.add_argument_group('stage control')
    g.add_argument('--stages', help='comma-separated subset, e.g. train,predict,evaluate')
    g.add_argument('--from-stage', choices=STAGES); g.add_argument('--to-stage', choices=STAGES)
    g.add_argument('--force', help='comma-separated stages to redo even if finished')
    g.add_argument('--dry-run', action='store_true'); g.add_argument('--list-stages', action='store_true')
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == '_check_model':     # internal helper
        _check_model(sys.argv[2])
        return 0
    a = ap.parse_args(argv)
    if a.list_stages:
        print('\n'.join(STAGES))
        return 0
    if a.root is None:
        if a.preset != 'smoke':
            ap.error('--root is required for the pilot/full presets: give a folder on a large scratch filesystem '
                     '(~200 GB). The home directory is usually too small, and quotas are not visible to the '
                     'free-space check.')
        a.root = str(REPO / 'lgel_run')
    a = resolve_settings(a)
    p = Paths(a.root, a.run_name)
    selected = select_stages(a)
    redo = []
    for f in filter(None, (a.force or '').split(',')):
        if f not in STAGES:
            raise SystemExit(f'--force: unknown stage {f!r}')
        redo += [s for s in STAGES[STAGES.index(f):] if s not in redo]

    if a.dry_run:                                       # describe only: nothing is created or deleted
        print(f'preset={a.preset} run={a.run_name} root={p.root}')
        print('stages: ' + ' -> '.join(selected))
        for s in selected:
            if s in ALWAYS_RUN:
                what = 'run'
            elif s in redo:
                what = 'run (forced)'
            elif s == 'train':
                what = 'run (resumes, or confirms that training is finished)'
            else:
                what = 'skip (finished)' if p.marker(s).exists() else 'run'
            print(f'  {s}: {what}')
        if redo:
            print('(--force would first discard the "finished" markers of: ' + ', '.join(redo) + ')')
        return 0

    for d in (p.root, p.state, p.logs, p.tmp):
        d.mkdir(parents=True, exist_ok=True)
    _LOGFILE = p.logs / 'main.log'
    log(f'preset={a.preset} run={a.run_name} root={p.root}')
    log('stages: ' + ' -> '.join(selected))
    prep_lock, run_lock = FileLock(p.prep_lock), FileLock(p.run_lock)
    if redo:
        shared = [s for s in redo if s not in RUN_SCOPED and s not in ALWAYS_RUN]
        if shared:
            if not prep_lock.acquire(blocking=False):
                log('!! another job is preparing data in this --root right now; --force of data stages has to wait '
                    'until it has finished.')
                return 1
            busy = busy_runs(p)
            if busy:
                log(f'!! run(s) {busy} are using the data of this --root right now; --force of data stages would '
                    f'change it under them. Wait until they have finished.')
                return 1
            log(f'--force: data stages {shared} will be redone. Runs trained on the previous data cannot be '
                f'continued or evaluated afterwards (they are refused); start new runs instead.')
        for s in redo:
            p.marker(s).unlink(missing_ok=True)
    if any(s in RUN_SCOPED for s in selected) and not run_lock.acquire(blocking=False):
        # Refuse a duplicate submission at once, before it waits for or touches anything.
        log(f'!! run "{a.run_name}" is already being processed by another job on this --root. Let that job '
            f'finish (then resubmit if needed), or use a different --run-name.')
        return 1

    # Preflight runs first, before anything expensive, and catches the avoidable failures.
    t_all = time.time()
    try:
        stage_preflight(a, p, selected)
    except StageError as e:
        log(f'!! {e}')
        return 1
    funcs = dict(fetch_models=stage_fetch_models, fetch_weights=stage_fetch_weights, fetch_data=stage_fetch_data,
                 extract_frames=stage_extract_frames, sanitize_annotations=stage_sanitize_annotations,
                 parse_annotations=stage_parse_annotations, splits=stage_splits, triplets=stage_triplets,
                 audit=stage_audit, train=stage_train, predict=stage_predict, evaluate=stage_evaluate,
                 summary=stage_summary)
    for stage in selected:
        if stage == 'preflight':
            continue                                            # already run above
        marker = p.marker(stage)
        sig = stage_signature(stage, a)
        if stage in RUN_SCOPED:
            # Wait while another job is (re)building the shared data, then let other jobs use it.
            prep_lock.acquire(blocking=True, waiting_for='preparing the shared data')
            prep_lock.release()
        elif stage not in ALWAYS_RUN and not marker.exists():
            # One job prepares the shared data; a second job started at the same time waits here and
            # then finds the stages finished.
            prep_lock.acquire(blocking=True, waiting_for='preparing the shared data')
        if stage not in ALWAYS_RUN and marker.exists():
            old = json.loads(marker.read_text())
            if old.get('signature') != sig and stage != 'train':
                log(f'!! stage "{stage}" was already finished with different settings {old.get("signature")} '
                    f'(now {sig}). Use a fresh --root, or --force {stage} to redo it and every later stage.')
                return 1
            if stage != 'train':
                log(f'== {stage}: already done, skipping')
                continue
        log(f'== {stage}: starting')
        t0 = time.time()
        try:
            funcs[stage](a, p)
        except StageError as e:
            log(f'!! stage {stage} FAILED: {e}')
            log('   Fix the cause and run the same command again: finished stages are skipped and '
                'training resumes from its last finished epoch.')
            return 1
        except Exception as e:                                  # noqa: BLE001
            import traceback
            log(f'!! stage {stage} crashed: {e!r}\n{traceback.format_exc()}')
            return 1
        if stage not in ALWAYS_RUN:
            write_json(marker, dict(signature=sig, finished=time.ctime(), seconds=round(time.time() - t0)))
        log(f'== {stage}: finished in {(time.time() - t0) / 60:.1f} min')
    log(f'ALL DONE in {(time.time() - t_all) / 3600:.2f} h. Results: {p.results}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
