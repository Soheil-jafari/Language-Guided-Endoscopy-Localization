#!/usr/bin/env python3
"""One-command pipeline: raw Cholec80 download -> frames -> labels -> splits -> train -> evaluate.

    bash hpc/submit.sh full --repeat 2                       # on the Slurm cluster: README_HPC.md, section 0
    python main.py --preset full --root /scratch/me/lgel     # elsewhere; the private download links are read
                                                             # from LGEL_DATA_URL / LGEL_WEIGHTS_URL in the
                                                             # environment, never typed on a command line

Every stage is idempotent: it writes a marker in <root>/state when it finishes and is
skipped on the next invocation, so a job killed by a wall-clock limit is resumed simply
by submitting the same command again (training resumes from the last finished epoch).
All heavy lifting is done by the repository's existing, tested entry points
(dataset_preprocessing/build_dataset.py, audit_data.py, train.py, predict.py,
build_reference.py, evaluate.py); this file only sequences them, fetches inputs, runs the
frame extraction in parallel and repairs the known trailing-row defect in two Cholec80
phase-annotation files.

Stages (run `python main.py --list-stages`):
  preflight fetch_models fetch_weights check_advanced fetch_data extract_frames sanitize_annotations
  parse_annotations splits triplets audit train predict evaluate summary

The `full` preset trains two models, each a separate run on the same prepared data: the baseline,
then the advanced model (Mamba temporal head, bi-level consistency loss, evidential uncertainty,
confidence fusion). `--variant` selects baseline, advanced or both.

The `paper` preset runs the complete experimental protocol of the paper (paper_protocol.py) on the
standard Cholec80 split (40/8/32 videos): both models with three seeds, one ablation per add-on, the
comparison methods (CLIP, PeskaVLP, X-CLIP, UniVTG), the unseen-wording test, and finally
paper_analysis.py, which writes every table and figure to <root>/paper/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import zipfile
import zlib
from pathlib import Path

try:
    import fcntl                                 # POSIX advisory locks (not available on Windows)
except ImportError:                              # pragma: no cover
    fcntl = None

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import paper_protocol as protocol  # noqa: E402  (standard library only at import time)

STAGES = ['preflight', 'fetch_models', 'fetch_weights', 'check_advanced', 'fetch_data', 'extract_frames',
          'sanitize_annotations', 'parse_annotations', 'splits', 'triplets', 'audit',
          'train', 'predict', 'evaluate', 'summary']
ALWAYS_RUN = {'preflight', 'check_advanced', 'summary'}          # cheap, never skipped
# Stages whose output belongs to one training run (<root>/runs/<run>, <root>/results/<run>): their
# 'finished' markers are kept per run, so several runs (pilot, full, other seeds) can share one --root
# and reuse its data stages.
RUN_SCOPED = {'train', 'predict', 'evaluate'}

# The advanced model is the baseline with these add-ons switched on (names as in project_config.py):
# the Mamba temporal head, the bi-level (semantic + optical-flow) consistency loss, evidential
# uncertainty and confidence-aware fusion. Nothing else differs between the two models.
ADVANCED_FLAGS = dict(MODEL=dict(TEMPORAL_HEAD_TYPE='SSM', SSM_USE_OFFICIAL_MAMBA=True, USE_UNCERTAINTY=True,
                                 USE_CONFIDENCE_FUSION=True),
                      TRAIN=dict(USE_BILEVEL_CONSISTENCY=True))
VARIANTS = ('baseline', 'advanced', 'both')

# name -> defaults. CLI flags override any of these.
PRESETS = {
    # Plumbing check on a handful of videos: minutes, not hours. (Also checks that the advanced
    # model's add-ons - Mamba etc. - run on this GPU, without training it.)
    'smoke': dict(max_videos=6, epochs=1, warmup=1, subset_ratio=0.1, batch_size=2, accum=2,
                  audit='sample', min_free_gb=10, variant='baseline'),
    # "Does it learn?" run: all videos, 10 % of the training windows, 3 epochs, baseline model.
    'pilot': dict(max_videos=None, epochs=3, warmup=1, subset_ratio=0.1, batch_size=8, accum=4,
                  audit='all', min_free_gb=200, variant='baseline'),
    # The real run: project defaults (20 epochs, effective batch 192), all training windows - first
    # the baseline model, then the advanced model, on the same data and splits.
    'full': dict(max_videos=None, epochs=20, warmup=3, subset_ratio=1.0, batch_size=8, accum=24,
                 audit='all', min_free_gb=200, variant='both'),
    # The paper's complete protocol (paper_protocol.py): standard split, 10 epochs per model (the 20-epoch
    # baseline peaked at epoch 3), every run of the protocol in priority order, then the analysis.
    'paper': dict(max_videos=None, epochs=10, warmup=3, subset_ratio=1.0, batch_size=8, accum=24,
                  audit='all', min_free_gb=300, variant='both'),
}


class StageError(RuntimeError):
    pass


CHECKPOINT_REJECTED = 3            # exit code of `main.py _check_model` when the checkpoint file itself is unusable


# --------------------------------------------------------------------------- utilities
_LOGFILE = None


# The two download links are private (they may carry access tokens). They are replaced by placeholders,
# and every other URL is cut down to its host, in everything printed, logged or put in a report.
_SECRETS = {}
_URL = re.compile(r'(?i)\b((?:https?|ftp)://)(?:[^\s/@\'"<>]*@)?([^\s/?#@\'"<>]+)([^\s\'"<>]*)')


def register_secret(value, label):
    """Hide a private link everywhere - and the parts of it that identify the file on their own, because
    tools sometimes print only those (e.g. '/uc?id=<file id>' in a Google Drive connection error)."""
    if not value or len(str(value)) < 8:
        return
    value = str(value)
    _SECRETS[value] = f'<{label}>'
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return
    found = re.findall(r'(?:/d/|[?&]id=)([-\w]{10,})', value)                  # Google Drive file ids
    for piece, is_query in [(s, False) for s in parts.path.split('/')] + \
                           [(q.partition('=')[2], True) for q in parts.query.split('&')]:
        for token in {piece, urllib.parse.unquote(piece)}:
            random_looking = bool(re.search(r'\d', token) and re.search(r'[A-Za-z]', token)) and '.' not in token
            if (len(token) >= 10 and random_looking) or (is_query and len(token) >= 16):
                found.append(token)
    if parts.netloc and (parts.query or found):          # path?query as written (not a bare /dataset.zip)
        found.append(value.split(parts.netloc, 1)[1])
    for token in found:
        if len(token) >= 10 and token not in _SECRETS:
            _SECRETS[token] = f'<{label} part>'


def redact(text):
    if not text:
        return text
    for value, label in sorted(_SECRETS.items(), key=lambda kv: -len(kv[0])):    # whole links first
        if value in text:
            text = text.replace(value, label)
    return _URL.sub(lambda m: m.group(1) + m.group(2) + ('/...' if m.group(3) or '@' in m.group(0) else ''), text)


_CONSOLE_LOST = False


def console(text):
    """Echo to the terminal or the Slurm job log. If that output is lost (a closed terminal, a broken
    pipe), the pipeline carries on: everything still goes to logs/main.log and the stage logs."""
    global _CONSOLE_LOST
    if _CONSOLE_LOST:
        return
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (OSError, ValueError, AttributeError):
        _CONSOLE_LOST = True
        try:                            # nothing left to flush at exit (a failed exit flush changes the exit code)
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass


def log(msg=''):
    line = redact(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}')
    console(line + '\n')
    if _LOGFILE is not None:
        try:
            with open(_LOGFILE, 'a') as f:
                f.write(line + '\n')
        except OSError:                 # logging is never what stops the pipeline (or a clean stop)
            pass


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
    """CPUs this job may use: the Slurm allocation if there is one, else the CPU affinity mask."""
    try:
        n = len(os.sched_getaffinity(0))         # respects scheduler/cgroup limits
    except AttributeError:
        n = os.cpu_count() or 1
    slurm = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if slurm.isdigit() and int(slurm) > 0:         # some clusters do not pin CPUs: never exceed the request
        n = min(n, int(slurm))
    return max(1, n)


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
        self.outbox = r / 'outbox'
        self.results = r / 'results' / run
        self.paper = r / 'paper'                       # tables and figures of the paper protocol
        self.log_scope = None                          # set while a run's own stages execute: logs/<run>/

    def triplet_csv(self, split):
        return self.triplets / f'cholec80_{split}_triplets.csv'

    def marker(self, stage):
        if stage in RUN_SCOPED:
            return self.state / f'{stage}.{self.run_name}.done.json'
        return self.state / f'{stage}.done.json'


def stage_env(a, p, online=False):
    """Environment of every step. online=True: a step that downloads models (a comparison method's own
    models, fetched when its run starts) may use the network even after the shared models were fetched -
    never with --offline."""
    env = os.environ.copy()
    # Not inherited from the machine: some images switch on the optional 'hf_transfer' downloader without
    # installing it (RunPod's does, and every model download then fails), and an inherited offline switch
    # would stop the first download. Offline mode is decided below (--offline, or models already fetched).
    for k in ('HF_HUB_ENABLE_HF_TRANSFER', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        env.pop(k, None)
    env.update(HF_HOME=str(p.hf), TORCH_HOME=str(p.root / 'torch_cache'), PYTHONUNBUFFERED='1',
               TOKENIZERS_PARALLELISM='false',
               PYTHONPATH=os.pathsep.join(x for x in (str(REPO), env.get('PYTHONPATH', '')) if x))
    if a.offline or (p.marker('fetch_models').exists() and not online):
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')   # cache was populated already
    return env


def run(cmd, a, p, name, progress_path=None, heartbeat=600, env=None, on_line=None, timeout=None):
    """Run a subprocess, stream its output to the console and logs/<name>.log. With `timeout` (seconds), a step
    that is still running then is stopped (its whole process group) and reported as failed (err.timed_out); only
    for optional checks - training steps never get a timeout."""
    folder = p.logs / p.log_scope if getattr(p, 'log_scope', None) else p.logs
    folder.mkdir(parents=True, exist_ok=True)
    logfile = folder / f'{name}.log'
    cmd = [str(c) for c in cmd]
    shown = redact(' '.join(cmd))
    log(f'$ {shown}')
    start, stop = time.time(), threading.Event()

    def beat():
        while not stop.wait(heartbeat):
            extra = ''
            if progress_path is not None:
                target = Path(progress_path)                    # (gdown writes to <name>*.part until done)
                try:
                    files = [target] if target.exists() else list(target.parent.glob(target.name + '*.part'))
                    if files:
                        extra = f', {gb(sum(f.stat().st_size for f in files)):.1f} GB so far'
                except OSError:
                    pass
            log(f'... {name} still running ({(time.time() - start) / 60:.0f} min{extra})')

    threading.Thread(target=beat, daemon=True).start()
    proc = None
    timed_out = threading.Event()

    def watchdog():
        if not stop.wait(timeout) and proc is not None and proc.poll() is None:
            timed_out.set()
            log(f'!! {name} is still running after {timeout / 60:.0f} min: stopping it')
            stop_process_group(proc)
    try:
        with open(logfile, 'a') as lf:
            lf.write(f'\n===== {time.ctime()} :: {shown}\n')
            # Own process group: on a stop (time limit, scancel, Ctrl-C) the whole step - including any
            # processes it started - is ended before this job writes its report and exits.
            proc = subprocess.Popen(cmd, cwd=REPO, env=env or stage_env(a, p), stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                    bufsize=1, start_new_session=hasattr(os, 'killpg'))
            if timeout is not None:
                threading.Thread(target=watchdog, daemon=True).start()
            for line in proc.stdout:
                line = redact(line)
                console(line)
                lf.write(line)
                if on_line is not None:
                    on_line(line)
            rc = proc.wait()
    except BaseException:
        if proc is not None:                         # even if its first process has already ended
            try:
                if proc.poll() is None:
                    log(f'stopping {name} ...')
            finally:
                stop_process_group(proc)
        raise
    finally:
        stop.set()
    if rc != 0:
        hint = ''
        if timed_out.is_set():
            hint = f' (stopped: still running after the time limit of {timeout:.0f} s for this step)'
        elif rc == -9:
            hint = (' (killed with SIGKILL: almost always the machine ran out of RAM, or the scheduler enforced its '
                    'memory limit - request more --mem, or lower --train-workers / --batch-size)')
        elif rc == -15:
            hint = ' (terminated with SIGTERM: usually the scheduler hit the job time limit - just submit the same command again)'
        err = StageError(f'`{cmd[0]} {cmd[1] if len(cmd) > 1 else ""}` exited with code {rc}{hint}. See {logfile}')
        err.rc = rc
        err.timed_out = timed_out.is_set()
        raise err
    return rc


def stop_process_group(proc, grace=15):
    """SIGTERM a step's whole process group, then SIGKILL whatever is left after `grace` seconds."""
    def send(sig):
        try:
            if hasattr(os, 'killpg'):
                os.killpg(proc.pid, sig)
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
    def settled():                                   # the step's first process AND everything it started
        proc.poll()                                  # (reaped, so a zombie does not count as running)
        if not hasattr(os, 'killpg'):
            return proc.returncode is not None
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def wait(seconds):
        deadline = time.time() + seconds
        while not settled() and time.time() < deadline:
            time.sleep(0.2)
        return settled()

    if settled():
        return
    send(signal.SIGTERM)
    if not wait(grace):
        send(getattr(signal, 'SIGKILL', signal.SIGTERM))
        wait(grace)


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
        for part in [p.zip] + sorted(p.raw.glob(p.zip.name + '*.part')):   # a (partial) download is continued
            try:
                present += part.stat().st_size
            except OSError:
                pass
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


class FileLock:
    """Exclusive POSIX advisory lock on a file under <root>/state, so that concurrent jobs on one --root
    take turns. The operating system releases it by itself when a job ends or is killed, so a lock can
    never be left behind. On a file system without lock support the pipeline refuses to start unless
    it is told that only one job at a time uses the --root (--single-job); then locking is disabled.
    There is deliberately no home-made fallback (heartbeat files): it could let two jobs run at once."""

    disabled = False                               # --single-job

    def __init__(self, path):
        self.path, self.fh = Path(path), None

    @property
    def held(self):
        return FileLock.disabled or self.fh is not None

    def acquire(self, blocking, waiting_for=''):
        if self.held:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, 'a+')
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                fh.close()
                raise
            if not blocking:
                fh.close()
                return False
            log(f'waiting: another job is {waiting_for} in this --root; continuing as soon as it is done ...')
            fcntl.lockf(fh, fcntl.LOCK_EX)
        self.fh = fh
        return True

    def release(self):
        if self.fh is not None:
            try:
                fcntl.lockf(self.fh, fcntl.LOCK_UN)
            finally:
                self.fh.close()
                self.fh = None

    @staticmethod
    def busy(path):
        """Is this lock held by a live job right now? (Does not wait; never used on a lock this process holds.)"""
        path = Path(path)
        if FileLock.disabled or not path.exists():
            return False
        with open(path, 'a+') as fh:
            try:
                fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.lockf(fh, fcntl.LOCK_UN)
                return False
            except OSError:
                return True                            # held by another job, or cannot tell: assume busy


def locks_work(state_dir):
    """Can this file system take POSIX locks? (None = yes; otherwise the reason it cannot.)"""
    if fcntl is None:
        return 'POSIX file locks are not available on this operating system'
    probe = Path(state_dir) / '.lock-probe'
    try:
        with open(probe, 'a+') as fh:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.lockf(fh, fcntl.LOCK_UN)
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN):         # held by another job: locks do work
            return None
        return f'the file system under {state_dir} does not support file locks ({e})'
    return None


def busy_runs(p):
    """Names of runs whose lock is held by a live job right now."""
    if not p.state.exists():
        return []
    names = {f.name[len('run.'):].removesuffix('.lock') for f in p.state.glob('run.*.lock')}
    return sorted(n for n in names if FileLock.busy(p.state / f'run.{n}.lock'))


def input_fingerprint(p):
    """Hashes of every prepared file a training run consumes (and the test manifest it is evaluated on)."""
    files = dict(splits=p.splits, train_triplets=p.triplet_csv('train'), val_triplets=p.triplet_csv('val'),
                 test_triplets=p.triplet_csv('test'), video_metadata=p.metadata, parsed_annotations=p.parsed)
    out = {k: (file_sha(f) if f.exists() else None) for k, f in files.items()}
    if p.triplets.exists():                             # the unseen-wording test manifests (paper protocol)
        out.update({f.name: file_sha(f) for f in sorted(p.triplets.glob('cholec80_test_para*_triplets.csv'))})
    return out


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
        # A file that the record does not mention yet (recorded by an older version) is added to it, not
        # treated as a change; a recorded file that differs or is gone is a change.
        changed = sorted(k for k in saved if saved.get(k) != current.get(k))
        added = sorted(k for k in current if k not in saved)
        produced = (p.run.exists() and (any(p.run.rglob('*.pth')) or any(p.run.rglob('*.pt')))) or \
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
        if added:
            log(f'run "{run_name}": recording {added} with its inputs')
            write_json(p.run_inputs, current)
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
    a.variant_given = getattr(a, 'variant', None) is not None
    for key, default in preset.items():
        if getattr(a, key, None) is None:
            setattr(a, key, default)
    a.workers = a.workers or min(16, max(1, cpu_count()))
    a.run_name_given = a.run_name is not None
    # The paper protocol: the standard Cholec80 split (unless another one is given) and the unseen-wording
    # test manifests. Its runs are named by paper_protocol.py; the name here labels the job's report.
    a.paraphrases = a.preset == 'paper'
    if a.preset == 'paper' and getattr(a, 'variant_given', False):
        raise SystemExit('--variant does not apply to the paper preset: it runs every model of paper_protocol.py')
    if a.preset == 'paper':
        a.splits_json = a.splits_json or str(protocol.SPLIT_FILE)
        a.run_name = a.run_name or 'paper'
    if a.run_name is None:
        named_by_variant = a.variant == 'advanced' or (a.variant == 'baseline' and PRESETS[a.preset]['variant'] == 'both')
        a.run_name = f'{a.preset}_{a.variant}_seed{a.seed}' if named_by_variant else f'{a.preset}_seed{a.seed}'
    for attr in ('data_zip', 'data_dir', 'weights_path', 'splits_json'):
        if getattr(a, attr):
            setattr(a, attr, str(Path(getattr(a, attr)).expanduser().resolve()))
    if Path(a.text_model).expanduser().exists():                  # a local directory, not a HF model id
        a.text_model = str(Path(a.text_model).expanduser().resolve())
    a.data_url = a.data_url or os.environ.get('LGEL_DATA_URL')
    a.weights_url = a.weights_url or os.environ.get('LGEL_WEIGHTS_URL')
    register_secret(a.data_url, 'LGEL_DATA_URL')
    register_secret(a.weights_url, 'LGEL_WEIGHTS_URL')
    return a


def model_finished(p):
    """A run whose evaluation finished (its summary is rewritten on every pass, so it is not required)."""
    return p.marker('evaluate').exists() and (p.results / 'proposed_test_metrics.json').exists()


def variant_runs(a):
    """(variant, run name) of every model this invocation trains, in order. 'both' = the baseline
    first, then the advanced model, each a separate run with its own checkpoints and results."""
    if getattr(a, 'variant', 'baseline') != 'both':
        return [(getattr(a, 'variant', 'baseline'), a.run_name)]
    if getattr(a, 'run_name_given', False):
        return [(v, f'{a.run_name}_{v}') for v in ('baseline', 'advanced')]
    return [(v, f'{a.preset}_{v}_seed{a.seed}') for v in ('baseline', 'advanced')]


def experiment_plans(a):
    """Every run this invocation handles, in order: a copy of the settings per run, with its name, kind
    ('proposed' or a comparison method), model flags (the advanced model's add-ons switched on) and seed."""
    plans = []
    if a.preset == 'paper':
        for spec in protocol.plans(a.run_name if a.run_name_given else 'paper'):
            sub = argparse.Namespace(**vars(a))
            sub.run_name, sub.kind, sub.flags, sub.seed = spec['name'], spec['kind'], spec['flags'], spec['seed']
            sub.paraphrase_eval, sub.group, sub.label = spec['paraphrases'], spec['group'], spec['label']
            sub.variant = (('advanced' if spec['flags'] == protocol.ADVANCED else 'baseline' if not spec['flags']
                            else 'ablation') if spec['kind'] == 'proposed' else spec['kind'])
            plans.append(sub)
        return plans
    for variant, name in variant_runs(a):
        sub = argparse.Namespace(**vars(a))
        sub.variant, sub.run_name, sub.kind = variant, name, 'proposed'
        sub.flags = protocol.merge_flags(ADVANCED_FLAGS) if variant == 'advanced' else {}
        sub.paraphrase_eval, sub.group, sub.label = False, 'main', variant
        plans.append(sub)
    return plans


def run_flags(a):
    """The add-on flags of a proposed-model run (older callers only set `variant`)."""
    flags = getattr(a, 'flags', None)
    if flags is None:
        flags = ADVANCED_FLAGS if getattr(a, 'variant', 'baseline') == 'advanced' else {}
    return flags


def uses_flow(a):
    return bool(run_flags(a).get('TRAIN', {}).get('USE_BILEVEL_CONSISTENCY'))


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
        'triplets': dict(fps=a.sample_fps, **({'paraphrases': protocol.paraphrase_signature()}
                                               if getattr(a, 'paraphrases', False) else {})),
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
    if os.path.isdir('/dev/shm'):                       # DataLoader processes pass batches through shared memory
        info['shm_free_gb'] = round(gb(shutil.disk_usage('/dev/shm').free), 1)
        if info['shm_free_gb'] < 4 and 'train' in selected:
            log(f'note: only {info["shm_free_gb"]} GB free in /dev/shm, through which the data-loader processes pass '
                f'their batches; a "bus error" or "shared memory" failure of a training step would point to this.')
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
    record_environment(p)
    return info


# --------------------------------------------------------------------------- stage: fetch_models
def record_environment(p):
    """Exact package versions and GPU/driver state of this job, for reproducibility and debugging."""
    parts = []
    for title, cmd in (('pip freeze', [sys.executable, '-m', 'pip', 'freeze']), ('nvidia-smi', ['nvidia-smi']),
                       ('modules', ['bash', '-lc', 'module list 2>&1'])):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            parts.append(f'===== {title}\n{out.stdout}{out.stderr}')
        except Exception as e:                              # noqa: BLE001
            parts.append(f'===== {title}\n(unavailable: {e})')
    keys = ('SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_JOB_PARTITION', 'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK',
            'SLURM_MEM_PER_NODE', 'SLURM_GPUS', 'SLURM_GPUS_ON_NODE', 'SLURM_JOB_GPUS', 'SLURM_RESTART_COUNT',
            'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX', 'CONDA_ENVS_PATH', 'CONDA_PKGS_DIRS', 'LGEL_BASE', 'LGEL_ROOT',
            'LGEL_PRESET', 'LGEL_ENV_PREFIX', 'LGEL_SINGLE_JOB', 'PYTHONNOUSERSITE', 'PYTHONUSERBASE', 'PYTHONPATH')
    parts.append('===== environment variables\n' + '\n'.join(f'{k}={os.environ[k]}' for k in keys if k in os.environ))
    (p.logs / 'environment.txt').write_text(redact('\n'.join(parts)) + '\n')


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


@contextlib.contextmanager
def url_file(url, p, curl=True):
    """The link in a private (0600) temporary file, so it never appears in a process list:
    a curl config file (`curl -K`) or a plain list for `wget -i`."""
    p.tmp.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix='.link-', suffix='.txt', dir=p.tmp)
    try:
        with os.fdopen(fd, 'w') as f:
            if curl:
                f.write('url = "' + url.replace('\\', '\\\\').replace('"', '\\"') + '"\n')
            else:
                f.write(url + '\n')
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


DOWNLOAD_ATTEMPTS, DOWNLOAD_PAUSE, STALL_SECONDS = 20, 15, 600


def remote_size(url, p):
    """Size of the file behind the link: the Content-Length of the final (post-redirect) response, only if
    that response is a success and not a web page (an expired link's error page, a refused HEAD request
    or a redirect must never be mistaken for the file). None when unknown: then nothing is decided by size."""
    try:
        with url_file(url, p) as conf:
            done = subprocess.run(['curl', '-sIL', '--max-time', '60', '-K', conf], capture_output=True, text=True)
    except Exception:                                           # noqa: BLE001
        return None
    if done.returncode != 0:                                    # e.g. a proxy tunnel that failed after its "200"
        return None
    blocks = [b for b in re.split(r'(?im)^(?=HTTP/)', done.stdout) if b.strip()]
    final = blocks[-1] if blocks else ''
    status = re.match(r'(?i)HTTP/\S+\s+(\d{3})', final)
    if not status or not status.group(1).startswith('2') or re.search(r'(?im)^content-type:\s*text/html', final):
        return None
    sizes = re.findall(r'(?im)^content-length:\s*(\d+)', final)
    return int(sizes[-1]) if sizes else None


def same_start(url, path, p):
    """Whether the file behind the link begins with the same bytes as `path` (its first KB)."""
    with open(path, 'rb') as f:
        mine = f.read(1024)
    if not mine:
        return False
    try:
        with url_file(url, p) as conf:
            proc = subprocess.Popen(['curl', '-fsSL', '--max-time', '120', '-r', f'0-{len(mine) - 1}', '-K', conf],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                theirs = proc.stdout.read(len(mine))            # (a server that ignores -r sends more: not read)
            finally:
                proc.kill()
                proc.wait()
    except Exception:                                           # noqa: BLE001
        return False
    return theirs == mine


def is_web_page(path):
    with open(path, 'rb') as f:
        head = f.read(1024).lstrip().lower()
    return head.startswith((b'<!doctype html', b'<html', b'<?xml', b'<head', b'<body'))


def web_page_error(name):
    return StageError(f'the link for {name} returned a web page, not the file (a share page, a login page or an '
                      f'expired link). Give a direct download link and run the same command again.')


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
    if is_web_page(dest):
        dest.unlink()
        raise web_page_error(dest.name)
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
            raise StageError('This is a Google Drive link and needs `gdown` (pip install "gdown>=6,<7").')
        if dest.exists():
            # gdown keeps its partial data in <name>*.part and treats an existing file as finished, but this
            # one was never verified (e.g. left by a direct-link download before the link was changed).
            log(f'{dest.name}: discarding an unverified earlier file before the Google Drive download')
            dest.unlink()
        env = stage_env(a, p)
        env['LGEL_GDOWN_URL'] = url                             # not on the command line
        # gdown 6 dropped `fuzzy` (share links are always understood) and added timeouts/retries; gdown 4
        # could resume from the wrong file, so it is refused. Only the arguments this gdown accepts are passed.
        # quiet: no progress bar flooding the logs (the heartbeat reports the size); errors still show.
        code = ('import gdown, inspect, os, sys\n'
                'version = getattr(gdown, "__version__", "5")\n'
                'if int(version.split(".")[0]) < 5:\n'
                '    sys.exit("gdown " + version + " is too old for this pipeline: pip install \'gdown>=6,<7\'")\n'
                'wanted = dict(quiet=True, fuzzy=True, resume=True, timeout=(60, 600), retries=20)\n'
                'accepted = inspect.signature(gdown.download).parameters\n'
                'kw = {k: v for k, v in wanted.items() if k in accepted}\n'
                'sys.exit(0 if gdown.download(os.environ["LGEL_GDOWN_URL"], sys.argv[1], **kw) else 1)\n')
        run([sys.executable, '-c', code, dest], a, p, name, progress_path=dest, env=env)
        return
    if shutil.which('curl'):
        total = remote_size(url, p)
        if dest.exists() and total is not None:
            if dest.stat().st_size == total:
                log(f'{dest.name} already fully downloaded ({gb(total):.2f} GB)')
                return
            if dest.stat().st_size > total:                 # cannot be the start of the file behind the link
                raise StageError(
                    f'{dest} ({dest.stat().st_size} bytes so far) is larger than what the link returns now ({total} '
                    f'bytes): the link has probably expired (it returns an error message) or points to another file. '
                    f'The partial download is kept. Check the link; only if it really is a different file, delete '
                    f'{dest} and run the same command again.')
        if total is not None:
            log(f'downloading {gb(total):.2f} GB -> {dest}')
        # Every attempt continues from the bytes already on disk (curl's own --retry would cut the file back
        # to where that curl process started). Slower than 1 KB/s for STALL_SECONDS counts as broken.
        base = ['curl', '-fsSL', '--connect-timeout', '60', '--speed-limit', '1024', '--speed-time', str(STALL_SECONDS)]
        # If the server cannot continue a partial file (curl exit 33; also what an expired link that answers
        # with a web page looks like), a complete new copy is fetched next to it and replaces it only if usable.
        fresh = dest.with_name(dest.name + '.restart')
        fresh.unlink(missing_ok=True)                       # an unfinished copy of that kind cannot be continued
        resumable = True
        with url_file(url, p) as conf:
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                target = dest if resumable else fresh
                if not resumable:
                    fresh.unlink(missing_ok=True)
                before = dest.stat().st_size if (resumable and dest.exists()) else None
                try:
                    run(base + (['-C', '-'] if resumable else []) + ['-o', target, '-K', conf], a, p, name,
                        progress_path=target)
                except StageError as e:
                    code = getattr(e, 'rc', None)
                    if attempt == DOWNLOAD_ATTEMPTS:
                        fresh.unlink(missing_ok=True)
                        raise
                    if code == 33 and resumable:
                        log('the server does not continue a partial download: fetching a complete new copy')
                        resumable = False
                        continue
                    log(f'download interrupted (curl exit code {code}); continuing in {DOWNLOAD_PAUSE} s '
                        f'(attempt {attempt + 1} of {DOWNLOAD_ATTEMPTS})')
                    time.sleep(DOWNLOAD_PAUSE)
                    continue
                if not resumable:
                    if not fresh.exists() or is_web_page(fresh):
                        fresh.unlink(missing_ok=True)
                        raise web_page_error(dest.name)    # the partial download is kept for a corrected link
                    if total is not None and fresh.stat().st_size != total:
                        got = fresh.stat().st_size         # incomplete or not the announced file: never replace
                        fresh.unlink()
                        if attempt == DOWNLOAD_ATTEMPTS:
                            raise StageError(f'the new copy of {dest.name} has {got} bytes, but the server announced '
                                             f'{total}. The partial download is kept. Run the same command again.')
                        log(f'the new copy has {got} of {total} bytes; trying again in {DOWNLOAD_PAUSE} s')
                        time.sleep(DOWNLOAD_PAUSE)
                        continue
                    os.replace(fresh, dest)
                elif total is None and before and dest.stat().st_size == before and not same_start(url, dest, p):
                    # curl adds nothing and reports success when the link now returns something shorter than the
                    # partial file (e.g. an error page): that is not the end of the file
                    raise StageError(f'the link for {dest.name} no longer returns the file that was being downloaded '
                                     f'(expired?). The partial download is kept. Check the link and run the same '
                                     f'command again.')
                break
        if not dest.exists():
            raise StageError(f'the download of {dest.name} produced no file. Run the same command again.')
        if total is not None and dest.stat().st_size != total:
            raise StageError(f'Downloaded {dest.stat().st_size} bytes but the server announced {total}. '
                             f'Run the same command again to continue.')
    elif shutil.which('wget'):
        with url_file(url, p, curl=False) as listing:
            run(['wget', '-nv', '-c', '--tries=20', '--waitretry=15', '-O', dest, '-i', listing], a, p, name,
                progress_path=dest)
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


_ADVANCED_CHECKED = False


def flow_weights_file(p):
    """Where torchvision caches the optical-flow network (RAFT-small) used by the bi-level loss."""
    from torchvision.models.optical_flow import Raft_Small_Weights
    return p.root / 'torch_cache' / 'hub' / 'checkpoints' / Path(Raft_Small_Weights.DEFAULT.url).name


def require_flow_weights_offline(a, p):
    """--offline must never touch the network: refuse before anything would try to download them."""
    if a.offline and not flow_weights_file(p).exists():
        raise StageError(f'--offline: the weights of the optical-flow network used by the advanced model\'s bi-level '
                         f'loss ({flow_weights_file(p).name}) are not in {flow_weights_file(p).parent} yet. Run the same '
                         f'command once without --offline on a machine with internet (e.g. with --to-stage fetch_data).')


def stage_check_advanced(a, p):
    """Build the advanced model (every add-on switched on) and run two training steps (batch 2) on the GPU
    with random inputs. This proves - in about a minute, before any long job depends on it - that the Mamba
    kernels, evidential uncertainty, confidence fusion and the bi-level loss (including the download of
    the optical-flow network's weights) all work in this environment; a failure stops the job. The paper
    preset then also rehearses the training batch (rehearse_comparisons; never stops the job). It trains
    nothing."""
    global _ADVANCED_CHECKED
    if _ADVANCED_CHECKED:
        log('advanced model already checked in this job')
        return
    require_flow_weights_offline(a, p)
    amp = a.amp_dtype if a.amp_dtype != 'auto' else ('bf16' if native_bf16() else 'fp16')
    cfg = dict(MODEL=dict(ADVANCED_FLAGS['MODEL'], TEXT_ENCODER_MODEL=a.text_model),
               TRAIN=dict(ADVANCED_FLAGS['TRAIN'], AMP_DTYPE=amp))
    p.tmp.mkdir(parents=True, exist_ok=True)
    cfg_path = p.tmp / 'check_advanced.json'
    write_json(cfg_path, cfg)
    try:
        run([sys.executable, __file__, '_check_advanced', cfg_path], a, p, 'check_advanced')
    except StageError as e:
        if getattr(e, 'rc', None) == MAMBA_MISSING:
            raise StageError(f'{e}\n   The Mamba package (mamba_ssm) is missing or broken in this environment, so the '
                             f'advanced model cannot run. setup_env.sh installs it, and run.sh retries the download at the '
                             f'start of every job that is not --offline (see the job log and logs/check_advanced.log).')
        if getattr(e, 'rc', None) == FLOW_WEIGHTS_MISSING:
            raise StageError(f'{e}\n   The weights of the optical-flow network used by the advanced model\'s bi-level '
                             f'loss (torchvision RAFT-small, a few MB from download.pytorch.org) are not in '
                             f'{p.root / "torch_cache"} and could not be downloaded. Run the same command once with '
                             f'internet access (without --offline).')
        raise
    _ADVANCED_CHECKED = True
    if getattr(a, 'preset', None) == 'paper':
        try:                                    # a report only: whatever happens in it, the job goes on
            rehearse_comparisons(a, p)
        except Exception as e:                  # noqa: BLE001
            log(f'!! the full-size rehearsal could not run ({e!r}); the job continues.')


REHEARSAL_PARTS = ('advanced', 'closedset', 'tecno_cnn', 'tecno_tcn')
REHEARSAL_TIMEOUT = 900                                    # seconds per model (normally well under a minute)
# The code that decides what the rehearsed models allocate (sizes, crop, clip, layers, losses).
REHEARSAL_CODE = ('main.py', 'project_config.py', 'models.py', 'losses.py', 'backbone/vision_transformer.py',
                  'backbone/endomamba.py', 'paper_protocol.py', 'comparison_models/rehearsal.py',
                  'comparison_models/closedset/closedset.py', 'comparison_models/tecno/tecno.py')


def rehearsal_signature(a, gpu, amp):
    """Everything a passed rehearsal depends on: a rehearsal is repeated when any of it differs (another GPU
    type, training batch, precision, text encoder, model flags or sizes in the code, PyTorch or Mamba)."""
    import hashlib
    import torch
    from importlib import metadata
    code = hashlib.sha256()
    for name in REHEARSAL_CODE:
        f = REPO / name
        code.update(name.encode() + b'\0' + (f.read_bytes() if f.is_file() else b'missing') + b'\0')
    try:
        mamba = metadata.version('mamba_ssm')
    except Exception:                                   # noqa: BLE001  (the functional check has reported it)
        mamba = None
    tecno = protocol.COMPARISONS['tecno']
    return dict(gpu=gpu, batch=int(a.batch_size), amp=amp, text_model=a.text_model, advanced_flags=ADVANCED_FLAGS,
                closedset_batch=int(protocol.COMPARISONS['closedset']['batch_size']),
                tecno_batch=int(tecno['cnn_batch_size']), tecno_size=int(tecno['input_size']),
                torch=torch.__version__, mamba_ssm=mamba, code_sha256=code.hexdigest())


def rehearse_comparisons(a, p):
    """Paper preset: once per GPU type, two full training steps (forward, loss, backward, optimiser) at the size
    the protocol trains them, with random weights and inputs: our advanced model (the largest of our models: the
    training batch, every add-on, the optical-flow loss; _check_advanced), the closed-set model and TeCNO
    (comparison_models/rehearsal.py). A few minutes in all, so that a memory problem shows in the first report
    instead of days later. Never stops the job: a run that cannot train fails on its own (and is retried by the
    next job, after a fix); here it is only reported early, with the peak GPU memory of each."""
    import torch
    if not torch.cuda.is_available():
        return None
    gpu = torch.cuda.get_device_name(0)
    amp = a.amp_dtype if a.amp_dtype != 'auto' else ('bf16' if native_bf16() else 'fp16')
    signature = rehearsal_signature(a, gpu, amp)
    record = p.state / 'comparison_rehearsal.json'
    try:                                        # an unreadable record is no reason to skip: rehearse again
        done = json.loads(record.read_text()) if record.exists() else {}
    except (OSError, ValueError) as e:
        log(f'the earlier rehearsal record could not be read ({e}); rehearsing again')
        done = {}
    if isinstance(done, dict) and done.get('signature') == signature and \
            all(isinstance(done.get(part), dict) and done[part].get('ok') is True for part in REHEARSAL_PARTS):
        log(f'full-size rehearsal already passed with these settings on this GPU type ({gpu})')
        return done
    p.tmp.mkdir(parents=True, exist_ok=True)
    cfg_path = p.tmp / 'rehearsal.json'
    write_json(cfg_path, dict(MODEL=dict(M2CRL_WEIGHTS_PATH=''), TRAIN=dict(AMP_DTYPE=amp)))   # our crop and clip
    adv_path = p.tmp / 'rehearsal_advanced.config.json'
    write_json(adv_path, dict(MODEL=dict(ADVANCED_FLAGS['MODEL'], TEXT_ENCODER_MODEL=a.text_model),
                              TRAIN=dict(ADVANCED_FLAGS['TRAIN'], AMP_DTYPE=amp)))
    tecno_cfg = protocol.COMPARISONS['tecno']
    results = dict(gpu=gpu, batch=a.batch_size, signature=signature)
    # One process per model (so no model's memory remains allocated while the next is measured), each with a
    # deadline (a hung CUDA call must not hold up the job; training steps never get one).
    for part in REHEARSAL_PARTS:
        out = p.tmp / f'rehearsal_{part}.json'
        out.unlink(missing_ok=True)
        if part == 'advanced':
            cmd = [sys.executable, __file__, '_check_advanced', adv_path, a.batch_size, out]
        else:
            cmd = [sys.executable, 'comparison_models/rehearsal.py', '--part', part, '--config', cfg_path, '--output',
                   out, '--closedset-batch', protocol.COMPARISONS['closedset']['batch_size'], '--tecno-batch',
                   tecno_cfg['cnn_batch_size'], '--tecno-size', tecno_cfg['input_size']]
        try:
            run(cmd, a, p, f'rehearse_{part}', timeout=REHEARSAL_TIMEOUT)
            r = json.loads(out.read_text())[part]
            if not (isinstance(r, dict) and r.get('ok') is True):
                raise ValueError(f'unexpected result {r!r:.300}')
            results[part] = r
            log(f'full-size rehearsal of {part}: OK, peak {r.get("peak_gb")} GB of GPU memory '
                f'({r.get("reserved_gb")} GB reserved), {r.get("seconds")} s - {r.get("what")}')
        except (StageError, OSError, ValueError, KeyError, TypeError) as e:
            try:                                        # the part's own error message, when it got that far
                own = json.loads(out.read_text()).get(part)
            except (OSError, ValueError, AttributeError):
                own = None
            results[part] = dict(own) if isinstance(own, dict) else {}
            results[part]['ok'] = False
            results[part]['error'] = results[part].get('error') or str(e)[:500]
            log(f'!! the full-size rehearsal of {part} failed on this GPU ({e}); the job continues, and the runs '
                f'concerned will report their own error. See logs/rehearse_{part}.log.')
    write_json(record, results)
    return results


def _check_advanced(cfg_path, batch=2, report=None):
    """Internal (subprocess): training steps of the advanced model on random data - forward, loss (with the
    optical-flow network), backward and an AdamW step, as train.py. batch 2: the functional check of every job;
    with the training batch and `report`, the full-size memory rehearsal (rehearse_comparisons)."""
    import time as _time
    import torch
    from project_config import config
    _apply_overrides(config, json.loads(Path(cfg_path).read_text()))
    from losses import MasterLoss
    try:                                                        # works without a GPU: fetched once, then cached
        criterion = MasterLoss(config)                          # (TORCH_HOME = <root>/torch_cache)
    except Exception as e:                                      # noqa: BLE001
        print(f'OPTICAL-FLOW WEIGHTS UNAVAILABLE: {e!r}', flush=True)
        sys.exit(FLOW_WEIGHTS_MISSING)
    print('optical-flow network (bi-level loss) ready', flush=True)
    on_cpu = os.environ.get('LGEL_CHECK_ON_CPU') == '1'          # TEST ONLY: the same steps on a CPU (stub Mamba)
    if not torch.cuda.is_available() and not on_cpu:
        print('advanced-model check SKIPPED: no CUDA GPU here (the Mamba kernels only run on a GPU)', flush=True)
        return
    try:
        import mamba_ssm
        if not on_cpu:
            import selective_scan_cuda  # noqa: F401  pyflakes: ignore (Mamba's compiled GPU kernels)
    except Exception as e:                                      # noqa: BLE001
        print(f'MAMBA NOT AVAILABLE: {e!r}', flush=True)
        sys.exit(MAMBA_MISSING)
    cuda = not on_cpu
    print(f'mamba_ssm {mamba_ssm.__version__}, torch {torch.__version__}, '
          f'{"GPU " + torch.cuda.get_device_name(0) if cuda else "CPU (test)"}', flush=True)
    from models import LocalizationFramework
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False                      # as train.py
    torch.backends.cudnn.deterministic = True
    dev = 'cuda' if cuda else 'cpu'
    started = _time.time()
    model = LocalizationFramework(config, initialize_backbone=False).to(dev).train()
    criterion = criterion.to(dev)
    criterion.set_epoch(0)
    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=config.TRAIN.LEARNING_RATE,
                                  weight_decay=config.TRAIN.WEIGHT_DECAY)
    length, size = config.DATA.CLIP_LENGTH, config.DATA.TRAIN_CROP_SIZE
    video = torch.randn(batch, 3, length, size, size, device=dev)
    words = ['grasper', 'calot triangle dissection']
    text = model.text_encoder.tokenizer([words[i % 2] for i in range(batch)], padding='max_length', truncation=True,
                                        max_length=config.DATA.MAX_TEXT_LENGTH, return_tensors='pt')
    target = torch.zeros(batch, length, device=dev)
    target[:, length // 2:] = 1
    target[-1, 0] = -100                                        # an unlabelled frame, as in real windows
    dtype = torch.bfloat16 if config.TRAIN.AMP_DTYPE == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and dtype == torch.float16)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(2):                                   # (the optimiser state exists from the first step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev, dtype=dtype, enabled=cuda):
                outputs = model(video, text['input_ids'].to(dev), text['attention_mask'].to(dev))
                loss, primary, regularizer = criterion(outputs, video, target)
            scaler.scale(loss).backward()
            if step == 0:
                mixers = [m for m in model.temporal_head.modules() if type(m).__name__ == 'Mamba']
                mamba_grad = any(q.grad is not None and bool(q.grad.abs().sum() > 0) for m in mixers for q in m.parameters())
            scaler.step(optimizer)
            scaler.update()
        if cuda:
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(f'ADVANCED MODEL OUT OF GPU MEMORY at batch {batch}: {e}', flush=True)
        if report:
            Path(report).write_text(json.dumps({'advanced': dict(ok=False, error=f'out of GPU memory at batch {batch}')}))
        sys.exit(1)
    peak, reserved = (gb(torch.cuda.max_memory_allocated()), gb(torch.cuda.max_memory_reserved())) if cuda else (0.0, 0.0)
    evidence = outputs[-1]
    problems = []
    if not torch.isfinite(loss):
        problems.append(f'non-finite loss {float(loss)}')
    if evidence is None or tuple(evidence.shape) != (batch, length, 2) or bool((evidence < 0).any()):
        problems.append('the uncertainty (evidence) output is missing or malformed')
    if not float(regularizer) > 0:
        problems.append('the bi-level consistency loss is not active')
    if model.language_guided_head.confidence_module is None:
        problems.append('confidence fusion is not active')
    if not mixers:
        problems.append('the temporal head has no official Mamba layers')
    if not mamba_grad:
        problems.append('no gradient reaches the Mamba layers')
    # Validation and prediction run the model in eval mode, without autocast and without gradients.
    model.eval()
    with torch.no_grad():
        scores = model(video, text['input_ids'].to(dev), text['attention_mask'].to(dev))[0]
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0).any()) or bool((scores > 1).any()):
        problems.append('the evaluation-mode forward pass does not give probabilities in [0, 1]')
    if report:
        Path(report).write_text(json.dumps({'advanced': dict(
            ok=not problems, error='; '.join(problems) or None, peak_gb=round(peak, 2), reserved_gb=round(reserved, 2),
            seconds=round(_time.time() - started, 1),
            what=f'our advanced model, batch {batch} x {length} frames, {size} px, {config.TRAIN.AMP_DTYPE}, with the '
                 f'optical-flow loss and an AdamW step')}, indent=2))
    if problems:
        print('ADVANCED MODEL CHECK FAILED: ' + '; '.join(problems), flush=True)
        sys.exit(1)
    print(f'advanced model OK: Mamba head ({len(mixers)} layers), uncertainty, confidence fusion and bi-level loss '
          f'(loss {float(loss):.3f} = {float(primary):.3f} + {float(regularizer):.3f}); two training steps with batch '
          f'{batch} used {peak:.1f} GB of GPU memory ({reserved:.1f} GB reserved)', flush=True)


MAMBA_MISSING, FLOW_WEIGHTS_MISSING = 3, 4


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
    """Extraction workers stop on SIGTERM/SIGHUP like any process (the parent's handler is not for them);
    a signal that is deliberately ignored (SIGHUP under nohup) stays ignored."""
    for s in STOP_SIGNALS:
        if signal.getsignal(s) != signal.SIG_IGN:
            signal.signal(s, signal.SIG_DFL)


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
            workers = list(getattr(pool, '_processes', {}).values())
            for w in workers:
                try:
                    w.terminate()                               # their SIGTERM handler is the default one
                except Exception:                               # noqa: BLE001
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            for w in workers:
                try:
                    w.join(timeout=10)
                except Exception:                               # noqa: BLE001
                    pass
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
    if getattr(a, 'paraphrases', False):
        # The test manifest again with each query in unseen wordings (same frames and labels).
        written = protocol.write_paraphrase_triplets(p.triplet_csv('test'), p.triplets)
        log(f'unseen-wording test manifests: {sorted(written)}')
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
    flags = run_flags(a)
    return dict(
        EXTRACTED_FRAMES_DIR=str(p.frames), VIDEO_METADATA_PATH=str(p.metadata),
        CHOLEC80_PARSED_ANNOTATIONS=str(p.parsed),
        TRAIN_TRIPLETS_CSV_PATH=str(p.triplet_csv('train')), VAL_TRIPLETS_CSV_PATH=str(p.triplet_csv('val')),
        TEST_TRIPLETS_CSV_PATH=str(p.triplet_csv('test')),
        CHECKPOINT_DIR=str(p.run), OUTPUT_DIR=str(p.results),
        MODEL=dict(M2CRL_WEIGHTS_PATH=read_weights_path(p, a), TEXT_ENCODER_MODEL=a.text_model,
                   **flags.get('MODEL', {})),
        DATA=dict(NUM_WORKERS=workers, SAMPLE_FPS=a.sample_fps),
        TRAIN=dict(SEED=a.seed, NUM_EPOCHS=a.epochs, WARMUP_EPOCHS=a.warmup, BATCH_SIZE=a.batch_size,
                   GRADIENT_ACCUMULATION_STEPS=a.accum, SUBSET_RATIO=a.subset_ratio, AMP_DTYPE=amp_dtype,
                   **flags.get('TRAIN', {})))


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


def reconcile_best(best, latest):
    """Finish (or discard) a best-model candidate that a stop left between the two checkpoint writes, so that the
    best model always belongs to the history the latest checkpoint describes (see checkpoint_utils.commit_best)."""
    best = Path(best)
    if not best.parent.exists() or not any(best.parent.glob(f'{best.stem}.pending-e*{best.suffix}')):
        return None
    from checkpoint_utils import commit_best
    committed = None
    if Path(latest).exists():
        import torch
        committed = torch.load(latest, map_location='cpu', weights_only=False).get('best_epoch')
    done = commit_best(best, committed)
    log(f'{best.parent.name}: ' + (f'completed the interrupted save of the best model (epoch {done})' if done else
                                   'discarded a best-model candidate of an epoch that was not committed'))
    return done


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


def training_finished(a, p):
    """Has this run's training already finished (its model would only be re-checked, never retrained)?"""
    if getattr(a, 'kind', 'proposed') != 'proposed':
        return p.run.exists() and any(p.run.glob('*/FINISHED.json'))
    return (p.run / 'latest_model.pth').exists() and completed_epochs(p) >= a.epochs


def stage_train(a, p):
    check_run_inputs(p, a.run_name)
    if 'train' in [f.strip() for f in str(getattr(a, 'force', None) or '').split(',')] and training_finished(a, p):
        # Finished training is never thrown away by a flag: retraining needs a new run (its own folder).
        raise StageError(f'run "{a.run_name}" has already finished training, and --force train does not retrain a '
                         f'model (it would only check it again). To train it again, use a new --run-name (or a new '
                         f'--root); to redo only its predictions and evaluation, use --force predict.')
    if getattr(a, 'kind', 'proposed') != 'proposed':
        return COMPARISON_METHODS[a.kind]['train'](a, p)
    if uses_flow(a):
        require_flow_weights_offline(a, p)
    amp = a.amp_dtype
    if amp == 'auto':
        amp = 'bf16' if native_bf16() else 'fp16'
    # 16 frames are decoded per sample; with enough CPUs, more loader processes keep the GPU busy.
    workers = a.train_workers if a.train_workers is not None else min(16, max(0, cpu_count() - 2))
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
    reconcile_best(p.run / 'best_model.pth', p.run / 'latest_model.pth')
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
def prediction_sets(a, p):
    """(name, manifest) of everything a run is evaluated on: the validation videos (where the threshold is
    chosen), the test videos, and - in the paper protocol - the test videos with unseen query wordings."""
    out = [('val', p.triplet_csv('val')), ('test', p.triplet_csv('test'))]
    if getattr(a, 'paraphrase_eval', False):
        for name in sorted(protocol.load_paraphrases()):
            out.append((f'test_para{name}', p.triplets / protocol.paraphrase_triplet_name(name)))
    return out


def stage_predict(a, p):
    check_run_inputs(p, a.run_name)
    p.results.mkdir(parents=True, exist_ok=True)
    sets = prediction_sets(a, p)
    for name, _ in sets:
        for f in (p.results / f'proposed_{name}.csv', p.results / f'proposed_{name}.csv.manifest.json'):
            f.unlink(missing_ok=True)
    if getattr(a, 'kind', 'proposed') != 'proposed':
        COMPARISON_METHODS[a.kind]['predict'](a, p, sets)
    else:
        ckpt = p.run / 'best_model.pth'
        if not ckpt.exists():
            raise StageError(f'{ckpt} not found; the train stage must finish first.')
        cfg = json.loads(p.run_cfg.read_text())
        for name, manifest in sets:
            run([sys.executable, 'predict.py', '--checkpoint', ckpt, '--triplets', manifest,
                 '--metadata', p.metadata, '--frames', p.frames, '--annotations', p.parsed,
                 '--output', p.results / f'proposed_{name}.csv', '--batch-size', a.pred_batch_size,
                 '--workers', cfg['DATA']['NUM_WORKERS']], a, p, f'predict_{name}')
    for name, manifest in sets:                         # labels on the same frames, independent of any model
        if not (p.results / f'proposed_{name}.csv').exists():
            raise StageError(f'no predictions were written for {name} ({p.results / f"proposed_{name}.csv"})')
        ref = p.results / f'{name}_reference.csv'
        ref.unlink(missing_ok=True)
        run([sys.executable, 'build_reference.py', '--triplets', manifest, '--metadata', p.metadata,
             '--frames', p.frames, '--annotations', p.parsed, '--output', ref, '--sample-fps', a.sample_fps],
            a, p, f'reference_{name}')


def metrics_file(p, name):
    return p.results / ('proposed_test_metrics.json' if name == 'test' else f'proposed_{name}_metrics.json')


def stage_evaluate(a, p):
    check_run_inputs(p, a.run_name)
    sets = prediction_sets(a, p)
    calib = p.results / 'proposed_calibration.json'
    for f in [calib] + [metrics_file(p, name) for name, _ in sets if name != 'val']:
        f.unlink(missing_ok=True)
    run([sys.executable, 'evaluate.py', '--predictions', p.results / 'proposed_val.csv', '--reference',
         p.results / 'val_reference.csv', '--split', 'validation', '--select-threshold', '--output', calib],
        a, p, 'evaluate_val')
    for name, _ in sets:                                # the validation threshold, unchanged, on every test set
        if name == 'val':
            continue
        run([sys.executable, 'evaluate.py', '--predictions', p.results / f'proposed_{name}.csv', '--reference',
             p.results / f'{name}_reference.csv', '--split', 'test', '--calibration', calib,
             '--output', metrics_file(p, name)], a, p, f'evaluate_{name}')


# --------------------------------------------------------------------------- comparison methods (paper)
# Each method is a run like ours: its own folder (<root>/runs/<run>), its predictions on exactly the same
# frames and manifests (proposed_<set>.csv in <root>/results/<run>), evaluated by the same evaluate.py with
# the threshold chosen on validation. A method that has to be trained resumes after an interruption and
# writes FINISHED.json when done; its own models are downloaded when its run starts (never with --offline).
PREFETCH = {
    'clip': 'from transformers import CLIPModel as M, CLIPProcessor as P',
    'xclip': 'from transformers import XCLIPModel as M, XCLIPProcessor as P',
    'cliptext': 'from transformers import CLIPTextModel as M, CLIPTokenizer as P',
}


def prefetch_hf(a, p, family, name):
    """Download (once) a Hugging Face model and its processor into <root>/hf_cache."""
    code = PREFETCH[family] + '; import sys; n = sys.argv[1]; M.from_pretrained(n); P.from_pretrained(n); print("ready:", n)'
    run([sys.executable, '-c', code, name], a, p, f'fetch_{family}', env=stage_env(a, p, online=not a.offline))


def finished_marker(folder):
    return Path(folder) / 'FINISHED.json'


def fresh_output(folder):
    """An output folder for a training script that refuses to overwrite: anything left by an attempt that
    was stopped before its first checkpoint is moved aside (never deleted)."""
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()) and not (folder / 'latest.pt').exists():
        aside = folder.with_name(f'{folder.name}.interrupted-{time.strftime("%Y%m%d-%H%M%S")}')
        os.replace(folder, aside)
        log(f'{folder.name}: an attempt stopped before its first epoch was moved aside ({aside.name})')


def work_order(a, p, manifest, name, labels=False, prompts=None):
    work = p.run / 'work'
    work.mkdir(parents=True, exist_ok=True)
    order = work / f'{name}.order.json'
    cmd = [sys.executable, 'comparison_models/frame_scores.py', 'work-order', '--triplets', manifest,
           '--metadata', p.metadata, '--frames', p.frames, '--annotations', p.parsed, '--output', order,
           '--sample-fps', a.sample_fps]
    label_file = work / f'{name}.labels.npz' if labels else None
    if labels:
        cmd += ['--labels', label_file]
    if prompts:
        write_json(work / f'{name}.prompts.json', prompts)
        cmd += ['--prompts', work / f'{name}.prompts.json']
    run(cmd, a, p, f'workorder_{name}')
    return order, label_file


def assemble_scores(a, p, scores, manifest, output, prompts=None, name='assemble'):
    Path(output).unlink(missing_ok=True)
    cmd = [sys.executable, 'comparison_models/frame_scores.py', 'assemble', '--scores', scores, '--triplets', manifest,
           '--metadata', p.metadata, '--annotations', p.parsed, '--output', output, '--sample-fps', a.sample_fps]
    if prompts:
        write_json(p.run / 'work' / f'{name}.prompts.json', prompts)
        cmd += ['--prompts', p.run / 'work' / f'{name}.prompts.json']
    run(cmd, a, p, name)


def score_workers(a):
    return a.train_workers if a.train_workers is not None else min(16, max(0, cpu_count() - 2))


# ----- CLIP (zero-shot)
def clip_train(a, p):
    prefetch_hf(a, p, 'clip', protocol.COMPARISONS['clip_zeroshot']['model'])
    log('CLIP zero-shot: nothing to train')


def clip_predict(a, p, sets):
    model = protocol.COMPARISONS['clip_zeroshot']['model']
    for name, manifest in sets:
        order, _ = work_order(a, p, manifest, name)
        scores = p.run / 'work' / f'{name}.scores.npz'
        scores.unlink(missing_ok=True)
        run([sys.executable, 'comparison_models/clip_scores.py', '--work-order', order, '--model-name', model,
             '--output', scores, '--workers', score_workers(a)], a, p, f'clip_{name}')
        assemble_scores(a, p, scores, manifest, p.results / f'proposed_{name}.csv', name=f'assemble_{name}')
        shutil.copy2(str(scores) + '.manifest.json', p.results / f'proposed_{name}.csv.manifest.json')


# ----- X-CLIP (fine-tuned): the repository's train_xclip.py and benchmark.py
def xclip_train(a, p):
    cfg = protocol.COMPARISONS['xclip_finetune']
    prefetch_hf(a, p, 'xclip', cfg['model'])
    out = p.run / 'xclip'
    if finished_marker(out).exists():
        log('X-CLIP fine-tuning already finished')
        return
    fresh_output(out)
    cmd = [sys.executable, 'train_xclip.py', '--train-triplets', p.triplet_csv('train'), '--val-triplets',
           p.triplet_csv('val'), '--metadata', p.metadata, '--frames', p.frames, '--annotations', p.parsed,
           '--output', out, '--model-name', cfg['model'], '--sample-fps', a.sample_fps, '--epochs', cfg['epochs'],
           '--batch-size', cfg['batch_size'], '--lr', cfg['lr'], '--seed', a.seed]
    if (out / 'latest.pt').exists():
        log('continuing the X-CLIP fine-tuning from its last finished epoch')
        cmd += ['--resume-from', out / 'latest.pt']
    run(cmd, a, p, 'train')
    if not (out / 'best.pt').exists():
        raise StageError('X-CLIP fine-tuning ended without a best checkpoint')
    write_json(finished_marker(out), dict(finished=time.ctime(), epochs=cfg['epochs']))


def xclip_predict(a, p, sets):
    cfg = protocol.COMPARISONS['xclip_finetune']
    for name, manifest in sets:
        run([sys.executable, 'benchmark.py', '--model', 'xclip', '--model-name', cfg['model'], '--checkpoint',
             p.run / 'xclip' / 'best.pt', '--triplets', manifest, '--metadata', p.metadata, '--frames', p.frames,
             '--annotations', p.parsed, '--output', p.results / f'proposed_{name}.csv', '--sample-fps', a.sample_fps],
            a, p, f'predict_{name}')


# ----- SurgVLP, HecVL, PeskaVLP (CAMMA's surgical video-language models), in their own environment
def peskavlp_prefix(p):
    if os.environ.get('LGEL_PESKAVLP_PREFIX'):
        return Path(os.environ['LGEL_PESKAVLP_PREFIX'])
    base = os.environ.get('LGEL_ENV_PREFIX') or os.environ.get('CONDA_PREFIX')
    return Path(base + '-peskavlp') if base else p.root / 'envs' / 'peskavlp'


def peskavlp_env(a, p, online=False):
    env = stage_env(a, p, online=online)
    env.pop('PYTHONPATH', None)                       # its own environment: nothing of this repository on the path
    return env


def safe_extract(archive, target):
    """Unpack a zip without letting a member escape the target folder."""
    target = Path(target)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            dest = (target / member).resolve()
            if target.resolve() not in dest.parents and dest != target.resolve():
                raise StageError(f'{archive}: unsafe member path {member!r}')
        zf.extractall(target)


def surgvlp_model(kind):
    """The SurgVLP-family model a comparison uses ('SurgVLP', 'HecVL' or 'PeskaVLP')."""
    return protocol.COMPARISONS[kind]['model']


def peskavlp_weights(a, p, model='PeskaVLP'):
    """The model's public checkpoint, downloaded once into <root>/comparisons/<model> (shared by its runs)."""
    if model not in protocol.SURGVLP['weights_urls']:
        raise StageError(f'unknown SurgVLP-family model {model!r}')
    folder = p.root / 'comparisons' / model.lower()
    download_path = folder / f'{model}.download'
    download(protocol.SURGVLP['weights_urls'][model], download_path, a, p, f'{model.lower()}_weights')
    if not zipfile.is_zipfile(download_path):
        return download_path                              # a plain checkpoint file
    unpacked = folder / 'unpacked'
    done = unpacked / '.complete'
    if not done.exists():
        shutil.rmtree(unpacked, ignore_errors=True)
        partial = folder / 'unpacked.partial'
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        safe_extract(download_path, partial)
        os.replace(partial, unpacked)
        done.write_text(time.ctime())
    found = sorted(f for f in unpacked.rglob('*') if f.suffix in ('.pth', '.pt', '.ckpt') and f.is_file()
                   and not f.name.startswith('._') and '__MACOSX' not in f.parts)      # (macOS archive extras)
    if len(found) != 1:
        raise StageError(f'expected one checkpoint in the {model} download, found {[f.name for f in found]}')
    return found[0]


def peskavlp_setup(a, p, model='PeskaVLP'):
    prefix = peskavlp_prefix(p)
    if a.offline and not (prefix / '.lgel_peskavlp_ready').exists():
        raise StageError(f'--offline: the SurgVLP environment ({prefix}) has not been created yet.')
    if not a.offline:
        run(['bash', 'comparison_models/peskavlp/setup_env.sh', prefix], a, p, 'peskavlp_setup',
            env=peskavlp_env(a, p, online=True))
    weights = peskavlp_weights(a, p, model)
    # One construction with the real weights (strict loading); it also fetches what the authors' model
    # constructor needs: Bio_ClinicalBERT (Hugging Face).
    run([prefix / 'bin' / 'python', 'comparison_models/peskavlp/peska_runner.py', 'check', '--weights', weights,
         '--model', model], a, p, f'{model.lower()}_check', env=peskavlp_env(a, p, online=not a.offline))
    return prefix, weights


def surgvlp_zeroshot_train(a, p):
    model = surgvlp_model(a.kind)
    peskavlp_setup(a, p, model)
    log(f'{model} zero-shot: nothing to train')


def peskavlp_finetune_train(a, p):
    cfg = protocol.COMPARISONS['peskavlp_finetune']
    prefix, weights = peskavlp_setup(a, p, cfg['model'])
    out = p.run / 'peskavlp'
    if finished_marker(out).exists():
        log('PeskaVLP fine-tuning already finished')
        return
    train_order, train_labels = work_order(a, p, p.triplet_csv('train'), 'train', labels=True)
    val_order, val_labels = work_order(a, p, p.triplet_csv('val'), 'val_labels', labels=True)
    out.mkdir(parents=True, exist_ok=True)
    run([prefix / 'bin' / 'python', 'comparison_models/peskavlp/peska_runner.py', 'finetune', '--train-order',
         train_order, '--train-labels', train_labels, '--val-order', val_order, '--val-labels', val_labels,
         '--weights', weights, '--model', cfg['model'], '--output', out, '--epochs', cfg['epochs'], '--lr', cfg['lr'],
         '--batch-size', cfg['batch_size'], '--seed', a.seed, '--workers', score_workers(a)], a, p, 'train',
        env=peskavlp_env(a, p))
    if not (out / 'best.pt').exists():
        raise StageError('PeskaVLP fine-tuning ended without a best checkpoint')
    write_json(finished_marker(out), dict(finished=time.ctime(), epochs=cfg['epochs']))


def peskavlp_predict(a, p, sets, checkpoint=None):
    model = surgvlp_model(a.kind)
    prefix, weights = peskavlp_prefix(p), peskavlp_weights(a, p, model)
    tag = model.lower()
    extra = []
    if checkpoint is None:
        # Also with the authors' own phase prompts (their zero-shot protocol), for the phase-recognition table.
        prompts = dict(zip(protocol.phase_queries(), protocol.PESKAVLP_PHASE_PROMPTS, strict=True))
        extra = [(f'authorprompts_{name}', manifest, prompts) for name, manifest in sets if name in ('val', 'test')]
    for name, manifest, prompts in [(n, m, None) for n, m in sets] + extra:
        order, _ = work_order(a, p, manifest, name, prompts=prompts)
        scores = p.run / 'work' / f'{name}.scores.npz'
        scores.unlink(missing_ok=True)
        cmd = [prefix / 'bin' / 'python', 'comparison_models/peskavlp/peska_runner.py', 'score', '--work-order', order,
               '--weights', weights, '--model', model, '--output', scores, '--workers', score_workers(a)]
        if checkpoint is not None:
            cmd += ['--checkpoint', checkpoint]
        run(cmd, a, p, f'{tag}_{name}', env=peskavlp_env(a, p))
        output = p.results / (f'{name}.csv' if prompts else f'proposed_{name}.csv')
        assemble_scores(a, p, scores, manifest, output, prompts=prompts, name=f'assemble_{name}')
        shutil.copy2(str(scores) + '.manifest.json', str(output) + '.manifest.json')


# ----- UniVTG (official model, M2CRL features)
def univtg_features_dir(a, p):
    weights = read_weights_path(p, a)
    key = file_sha(weights) if weights else 'random-init'
    return p.root / 'features' / f'm2crl_{key}'


def univtg_train(a, p):
    cfg = protocol.COMPARISONS['univtg']
    features = univtg_features_dir(a, p)
    weights = read_weights_path(p, a)
    p.run.mkdir(parents=True, exist_ok=True)
    run([sys.executable, 'comparison_models/univtg/adapter.py', 'features', '--weights', weights or '',
         '--metadata', p.metadata, '--frames', p.frames, '--videos', p.splits, '--output', features,
         '--text-model', a.text_model, '--sample-fps', a.sample_fps], a, p, 'univtg_features')
    manifests = [p.triplet_csv(s) for s in ('train', 'val', 'test')] + [m for n, m in prediction_sets(a, p)
                                                                          if n.startswith('test_para')]
    run([sys.executable, 'comparison_models/univtg/adapter.py', 'text', '--triplets', *manifests,
         '--output', p.run / 'text_features.pt', '--text-model', a.text_model], a, p, 'univtg_text')
    out = p.run / 'univtg'
    if finished_marker(out).exists():
        log('UniVTG training already finished')
        return
    run([sys.executable, 'comparison_models/univtg/adapter.py', 'train', '--train', p.triplet_csv('train'),
         '--val', p.triplet_csv('val'), '--metadata', p.metadata, '--annotations', p.parsed, '--features', features,
         '--text-features', p.run / 'text_features.pt', '--output', out, '--clip-frames', cfg['clip_length'],
         '--epochs', cfg['epochs'], '--eval-every', cfg['eval_every'], '--batch-size', cfg['batch_size'],
         '--lr', cfg['lr'], '--seed', a.seed, '--sample-fps', a.sample_fps], a, p, 'train')
    write_json(finished_marker(out), dict(finished=time.ctime(), epochs=cfg['epochs']))


def univtg_predict(a, p, sets):
    for name, manifest in sets:
        segments = p.results / f'native_segments_{name}.jsonl'
        segments.unlink(missing_ok=True)
        run([sys.executable, 'comparison_models/univtg/adapter.py', 'predict', '--checkpoint', p.run / 'univtg' / 'best.pt',
             '--triplets', manifest, '--metadata', p.metadata, '--annotations', p.parsed,
             '--features', univtg_features_dir(a, p), '--text-features', p.run / 'text_features.pt',
             '--output', p.results / f'proposed_{name}.csv', '--segments', segments], a, p, f'predict_{name}')


# ----- frame-wise closed-set models (no text input): one output per concept answers that concept's queries
def scored_predictions(a, p, sets, tag, command):
    """For each prediction set: a work order, the model's score matrix (command(order, scores) -> argv), and
    the standard prediction CSV assembled from it."""
    for name, manifest in sets:
        order, _ = work_order(a, p, manifest, name)
        scores = p.run / 'work' / f'{name}.scores.npz'
        scores.unlink(missing_ok=True)
        run(command(order, scores), a, p, f'{tag}_{name}')
        assemble_scores(a, p, scores, manifest, p.results / f'proposed_{name}.csv', name=f'assemble_{name}')
        shutil.copy2(str(scores) + '.manifest.json', p.results / f'proposed_{name}.csv.manifest.json')


# ----- our model without language (closed set): comparison_models/closedset/closedset.py
def closedset_train(a, p):
    spec = protocol.COMPARISONS[a.kind]
    out = p.run / 'closedset'
    if finished_marker(out).exists():
        log('closed-set training already finished')
        return
    amp = a.amp_dtype
    if amp == 'auto':
        amp = 'bf16' if native_bf16() else 'fp16'
    # Our model's own training configuration (same backbone weights, optimiser, schedule, epochs, crop, clip,
    # augmentation, seed), with the closed-set batch.
    cfg = train_config(a, p, score_workers(a), amp)
    cfg['TRAIN'].update(BATCH_SIZE=spec['batch_size'], GRADIENT_ACCUMULATION_STEPS=spec['accum'])
    config_file = p.run / 'closedset_config.json'
    if config_file.exists():
        saved = json.loads(config_file.read_text())
        if a.train_workers is None:                     # node-dependent choices are frozen at the first launch
            cfg['DATA']['NUM_WORKERS'] = saved['DATA']['NUM_WORKERS']
        if a.amp_dtype == 'auto':
            cfg['TRAIN']['AMP_DTYPE'] = saved['TRAIN']['AMP_DTYPE']
        def comparable(c):                              # the loader process count does not change the training
            return {k: ({kk: vv for kk, vv in v.items() if kk != 'NUM_WORKERS'} if k == 'DATA' else v)
                    for k, v in c.items()}
        if comparable(cfg) != comparable(saved) and (out / 'latest.pt').exists():
            raise StageError(f'Run "{a.run_name}" was started with different settings; restore them to resume it.')
    p.run.mkdir(parents=True, exist_ok=True)
    write_json(config_file, cfg)
    train_order, train_labels = work_order(a, p, p.triplet_csv('train'), 'train', labels=True)
    val_order, val_labels = work_order(a, p, p.triplet_csv('val'), 'val_labels', labels=True)
    run([sys.executable, 'comparison_models/closedset/closedset.py', 'train', '--config', config_file,
         '--train-order', train_order, '--train-labels', train_labels, '--val-order', val_order,
         '--val-labels', val_labels, '--output', out, '--parameter-counts', p.run / 'parameter_counts.json',
         '--repeats', spec['repeats'], '--workers', cfg['DATA']['NUM_WORKERS']], a, p, 'train',
        on_line=_epoch_progress(cfg['TRAIN']['NUM_EPOCHS']))
    if not (out / 'best.pt').exists():
        raise StageError('closed-set training ended without a best checkpoint')
    write_json(finished_marker(out), dict(finished=time.ctime(), epochs=cfg['TRAIN']['NUM_EPOCHS']))


def closedset_predict(a, p, sets):
    checkpoint = p.run / 'closedset' / 'best.pt'
    scored_predictions(a, p, sets, 'closedset', lambda order, scores: [
        sys.executable, 'comparison_models/closedset/closedset.py', 'predict', '--checkpoint', checkpoint,
        '--work-order', order, '--output', scores, '--workers', score_workers(a)])


# ----- TeCNO (re-implementation): comparison_models/tecno/tecno.py
TECNO = 'comparison_models/tecno/tecno.py'
RESNET50_IMAGENET = 'resnet50-0676ba61.pth'                # torchvision ResNet50_Weights.IMAGENET1K_V1


def tecno_train(a, p):
    cfg = protocol.COMPARISONS['tecno']
    cnn, features, tcn = p.run / 'tecno_cnn', p.run / 'tecno_features', p.run / 'tecno_tcn'
    if finished_marker(tcn).exists():
        log('TeCNO training already finished')
        return
    train_order, train_labels = work_order(a, p, p.triplet_csv('train'), 'train', labels=True)
    val_order, val_labels = work_order(a, p, p.triplet_csv('val'), 'val_labels', labels=True)
    common = ['--train-order', train_order, '--train-labels', train_labels, '--val-order', val_order,
              '--val-labels', val_labels, '--patience', cfg['patience'], '--seed', a.seed]
    if not finished_marker(cnn).exists():
        imagenet = p.root / 'torch_cache' / 'hub' / 'checkpoints' / RESNET50_IMAGENET
        if a.offline and not imagenet.exists() and not (cnn / 'latest.pt').exists():
            raise StageError(f'--offline: the ImageNet ResNet-50 weights TeCNO starts from ({imagenet}) have not been '
                             f'downloaded yet (torchvision downloads them on the first run with internet).')
        run([sys.executable, TECNO, 'cnn', *common, '--output', cnn, '--epochs', cfg['cnn_epochs'],
             '--min-epochs', cfg['cnn_min_epochs'], '--lr', cfg['cnn_lr'], '--batch-size', cfg['cnn_batch_size'],
             '--input-size', cfg['input_size'], '--workers', score_workers(a)], a, p, 'train_cnn')
        if not (cnn / 'best.pt').exists():
            raise StageError('TeCNO stage 1 ended without a best checkpoint')
        write_json(finished_marker(cnn), dict(finished=time.ctime()))
    test_order, _ = work_order(a, p, p.triplet_csv('test'), 'test')
    run([sys.executable, TECNO, 'extract', '--checkpoint', cnn / 'best.pt', '--work-order', train_order, val_order,
         test_order, '--output', features, '--workers', score_workers(a)], a, p, 'tecno_features')
    run([sys.executable, TECNO, 'tcn', *common, '--features', features, '--output', tcn, '--epochs', cfg['tcn_epochs'],
         '--min-epochs', cfg['tcn_min_epochs'], '--lr', cfg['tcn_lr'], '--parameter-counts',
         p.run / 'parameter_counts.json'], a, p, 'train_tcn')
    if not (tcn / 'best.pt').exists():
        raise StageError('TeCNO stage 2 ended without a best checkpoint')
    write_json(finished_marker(tcn), dict(finished=time.ctime()))


def tecno_predict(a, p, sets):
    scored_predictions(a, p, sets, 'tecno', lambda order, scores: [
        sys.executable, TECNO, 'predict', '--checkpoint', p.run / 'tecno_tcn' / 'best.pt', '--features',
        p.run / 'tecno_features', '--work-order', order, '--output', scores])


COMPARISON_METHODS = {
    'clip_zeroshot': dict(train=clip_train, predict=clip_predict),
    'xclip_finetune': dict(train=xclip_train, predict=xclip_predict),
    **{kind: dict(train=surgvlp_zeroshot_train, predict=lambda a, p, sets: peskavlp_predict(a, p, sets))
       for kind in protocol.SURGVLP_ZERO_SHOT},
    'peskavlp_finetune': dict(train=peskavlp_finetune_train,
                              predict=lambda a, p, sets: peskavlp_predict(a, p, sets, p.run / 'peskavlp' / 'best.pt')),
    'univtg': dict(train=univtg_train, predict=univtg_predict),
    'tecno': dict(train=tecno_train, predict=tecno_predict),
    'closedset': dict(train=closedset_train, predict=closedset_predict),
    'closedset_matched': dict(train=closedset_train, predict=closedset_predict),
}


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
    summary = dict(run=a.run_name, preset=a.preset, git_commit=commit,
                   latest_command_args={k: str(v) for k, v in vars(a).items() if 'url' not in k})
    kind = getattr(a, 'kind', 'proposed')
    summary['kind'] = kind
    if kind != 'proposed':
        summary['method'] = getattr(a, 'label', kind)
        lines += [f'Method: **{summary["method"]}** (comparison method; evaluated exactly like our model: the same '
                  f'frames and manifests, the threshold chosen on the validation videos).', '']
        for log_file in sorted(p.run.glob('*/training_metrics.jsonl')) if p.run.exists() else []:
            rows = []
            for line in log_file.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            summary.setdefault('training', []).extend(rows)
            if rows:
                keys = list(dict.fromkeys(k for r in rows for k in r))
                lines += [f'## Training ({log_file.parent.name})', '', '| ' + ' | '.join(keys) + ' |',
                          '|' + '---|' * len(keys)]
                lines += ['| ' + ' | '.join(f'{r[k]:.4g}' if isinstance(r.get(k), float) else str(r.get(k, ''))
                                            for k in keys) + ' |' for r in rows]
                lines.append('')
    if p.run_cfg.exists():                              # what this model was actually trained with
        trained = json.loads(p.run_cfg.read_text())
        summary['trained_with'] = dict(MODEL=trained.get('MODEL', {}), TRAIN=trained.get('TRAIN', {}))
        description = protocol.describe_flags(summary['trained_with'])
        tr = trained.get('TRAIN', {})
        lines += [f'Model: **{description}**'
                  + f'; trained for {tr.get("NUM_EPOCHS", "?")} epochs, batch {tr.get("BATCH_SIZE", "?")} x '
                    f'{tr.get("GRADIENT_ACCUMULATION_STEPS", "?")} accumulation, {tr.get("AMP_DTYPE", "?")}, '
                    f'seed {tr.get("SEED", "?")}.', '']
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
            lines.append(f'| {r.get("epoch")} | {_fmt(r.get("train_loss"))} | {_fmt(r.get("val_frame_nll"))} | '
                         f'{_fmt(r.get("auroc"))} | {_fmt(r.get("average_precision"))} | {_fmt(r.get("validation_f1"))} | '
                         f'{mins[0]} | {mins[1]} |')
        lines.append('')
        timed = [r for r in rows if isinstance(r.get('train_seconds'), (int, float)) and isinstance(r.get('val_seconds'), (int, float))]
        if timed and a.preset == 'pilot' and a.subset_ratio:
            tr = sorted(r['train_seconds'] for r in timed)[len(timed) // 2]
            va = sorted(r['val_seconds'] for r in timed)[len(timed) // 2]
            full = PRESETS['full']
            epoch_s = tr * full['subset_ratio'] / a.subset_ratio + va
            summary['full_run_estimate_hours'] = round(full['epochs'] * epoch_s / 3600, 1)
            summary['full_run_estimate_note'] = 'baseline model only; the full preset also trains the advanced model'
            predict_s = None
            try:
                predict_s = json.loads(p.marker('predict').read_text()).get('seconds')
            except (OSError, ValueError):
                pass
            predicted = ''
            if isinstance(predict_s, (int, float)):
                summary['prediction_minutes_per_model'] = round(predict_s / 60, 1)
                predicted = (f' Prediction (all validation and test videos) took **{_duration(predict_s)}** here; the '
                             f'full run predicts the same videos, so add about that much per model.')
            lines += ['## Time estimate for the `full` preset', '',
                      f'Pilot epoch (median): {tr / 60:.1f} min training on {a.subset_ratio:.0%} of the training windows + '
                      f'{va / 60:.1f} min validation. Scaling the training part to all windows gives about '
                      f'**{_duration(epoch_s)} per full epoch** and **{_duration(full["epochs"] * epoch_s)} for '
                      f'{full["epochs"]} epochs**, plus prediction.{predicted} Rough: it assumes the same GPU and file '
                      f'system, and that training time grows linearly with the number of training windows. Each job '
                      f'must fit at least one whole epoch, because training resumes from the last finished epoch.', '']
            if a.variant == 'baseline':
                lines += ['This is for the **baseline** model. The `full` preset then trains the **advanced** model for '
                      'the same number of epochs; it also computes optical flow at every step, so plan for **at least '
                      'twice this time in total** (the advanced run reports its own epoch times).', '']
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
            rows = [r for r in rows if all(k in r for k in ('epoch', 'train_loss', 'val_frame_nll', 'auroc', 'average_precision'))]
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
    unseen = [(name, metrics_file(p, name)) for name, _ in prediction_sets(a, p) if name.startswith('test_para')]
    unseen = [(name, f) for name, f in unseen if f.exists()]
    if unseen:
        lines += ['## Unseen query wordings (test videos, same threshold)', '',
                  '| query set | AP | AUROC | F1 | precision | recall |', '|---|---|---|---|---|---|']
        for name, f in unseen:
            m = json.loads(f.read_text())
            summary[name] = {k: v for k, v in m.items() if k not in ('per_video', 'per_query')}
            lines.append(f'| {name[len("test_"):]} | {_fmt(m["average_precision"])} | {_fmt(m["auroc"])} | '
                         f'{_fmt(m["f1"])} | {_fmt(m["precision"])} | {_fmt(m["recall"])} |')
        lines.append('')
    export_model_weights(p)
    write_json(p.results / 'summary.json', summary)
    (p.results / 'SUMMARY.md').write_text('\n'.join(lines))
    log(f'summary written: {p.results / "SUMMARY.md"}')


def write_comparison(a, plans):
    """One table with the test results of the baseline and the advanced model side by side."""
    keys = ('average_precision', 'auroc', 'f1', 'precision', 'recall', 'brier', 'ece')
    lines = [f'# Baseline vs advanced model: {a.run_name}', '',
             'Same data and splits; the advanced model adds the Mamba temporal head, the bi-level consistency '
             'loss, evidential uncertainty and confidence fusion. Test videos, threshold fixed from validation.', '',
             '| model | run | ' + ' | '.join(k.replace('_', ' ') for k in keys) + ' | epochs | batch x accumulation | precision |',
             '|---|---|' + '---|' * len(keys) + '---|---|---|']
    for sub, sp in plans:
        f = sp.results / 'proposed_test_metrics.json'
        m = json.loads(f.read_text()) if f.exists() else {}
        tr = json.loads(sp.run_cfg.read_text()).get('TRAIN', {}) if sp.run_cfg.exists() else {}
        lines.append(f'| {sub.variant} | {sub.run_name} | ' + ' | '.join(_fmt(m.get(k)) for k in keys) +
                     f' | {tr.get("NUM_EPOCHS", "?")} | {tr.get("BATCH_SIZE", "?")} x '
                     f'{tr.get("GRADIENT_ACCUMULATION_STEPS", "?")} | {tr.get("AMP_DTYPE", "?")} |')
    out = plans[0][1].root / 'results' / f'{a.run_name}_COMPARISON.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    log(f'comparison written: {out}')


# --------------------------------------------------------------------------- report bundle
BUNDLE_TOTAL = 80 * 1024 ** 2        # uncompressed payload of one report (about 15-20 MB as .tar.gz)
BUNDLE_FILE = 25 * 1024 ** 2         # largest single file taken whole
BUNDLE_TAIL = 5 * 1024 ** 2          # larger logs: only their last part
BUNDLE_TEXT = {'.json', '.jsonl', '.csv', '.md', '.txt', '.log', '.out', '.tex'}
BUNDLE_IMAGES = {'training_curves.png'}


def _is_image(f, root):
    """Pictures that go into a report: training curves, and the paper figures (PNG; their PDFs stay)."""
    if f.name in BUNDLE_IMAGES:
        return True
    try:
        rel = f.relative_to(root / 'paper')
    except ValueError:
        return False
    return f.suffix.lower() == '.png' and rel.parts[0] in ('figures', 'explanations')


def _bundle_candidates(ps):
    """Report files in priority order: results and metrics first, bulky predictions last.
    `ps`: the Paths of every run this job handled (the baseline and the advanced model)."""
    ps = ps if isinstance(ps, (list, tuple)) else [ps]
    p = ps[0]
    paper = p.root / 'paper'
    first = [paper / 'PAPER_RESULTS.md', paper / 'analysis.json'] if paper.exists() else []
    first += sorted((p.root / 'results').glob('*_COMPARISON.md')) if (p.root / 'results').exists() else []
    for q in ps:
        first += [q.results / n for n in ('SUMMARY.md', 'summary.json', 'proposed_calibration.json',
                                          'proposed_test_metrics.json', 'training_curves.png')]
    for q in ps:
        first += [q.run / 'training_metrics.jsonl', q.run / 'run_config.json', q.run / 'parameter_counts.json',
                  q.run_cfg, q.run_inputs]
    first += [p.audit, p.splits, p.splits.with_suffix('.provenance.json'),
              p.data / 'annotations_sanitized.json', p.metadata, p.inventory, *sorted(p.state.glob('*.json'))]
    logs = sorted(f for f in p.logs.rglob('*') if f.is_file()) if p.logs.exists() else []    # (logs/<run>/ too)
    logs.sort(key=lambda f: (f.name != 'main.log', not f.name.startswith('slurm-'), f.parent != p.logs, str(f)))
    figures = sorted(f for f in paper.rglob('*') if f.is_file()) if paper.exists() else []    # tables, figures
    rest = figures + [f for q in ps for f in (sorted(q.results.glob('*')) if q.results.exists() else [])]
    seen, out = set(), []
    for f in first + logs + rest:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def write_report_bundle(a, p, rc):
    """After every job (finished, failed or stopped by the scheduler) pack what is needed to see what
    happened - results, metrics, settings, data reports, logs - into one small file in <root>/outbox
    that can be sent back. Checkpoints, data and unknown files never go in; private links are
    redacted; the size is bounded; MANIFEST.txt lists anything left out or shortened.
    `p`: the Paths of the run, or a list of them when one job trains several models."""
    ps = p if isinstance(p, (list, tuple)) else [p]
    p = ps[0]
    try:
        import io
        import tarfile
        p.outbox.mkdir(parents=True, exist_ok=True)
        job = os.environ.get('SLURM_JOB_ID')
        status = 'OK' if rc == 0 else ('STOPPED' if rc in (129, 143) else 'FAILED')
        name = f'{a.run_name}-{time.strftime("%Y%m%d-%H%M%S")}{f"-job{job}" if job else ""}-{status}.tar.gz'
        included, shortened, omitted, total = [], [], [], 0
        payload = []
        for f in _bundle_candidates(ps):
            try:
                if f.is_symlink() or not f.is_file() or p.root not in f.resolve().parents:
                    continue
                rel = f.relative_to(p.root).as_posix()
                suffix = f.suffix.lower()
                image = _is_image(f, p.root)
                if suffix not in BUNDLE_TEXT and not image:
                    continue                                    # checkpoints, temporary files, data ...
                size = f.stat().st_size
                if image:
                    data = f.read_bytes() if size <= BUNDLE_FILE else None
                elif size > BUNDLE_FILE and suffix in ('.log', '.out', '.txt'):
                    with open(f, 'rb') as fh:
                        fh.seek(size - BUNDLE_TAIL)
                        data = fh.read()
                    shortened.append(f'{rel} (last {BUNDLE_TAIL // 1024 ** 2} MB of {size / 1024 ** 2:.0f} MB)')
                elif size > BUNDLE_FILE:
                    data = None
                else:
                    data = f.read_bytes()
                if data is None or total + len(data) > BUNDLE_TOTAL:
                    omitted.append(f'{rel} ({size / 1024 ** 2:.1f} MB)')
                    continue
                if not image:
                    data = redact(data.decode('utf-8', errors='replace')).encode('utf-8')
                total += len(data)
                payload.append((rel, data))
                included.append(rel)
            except OSError as e:
                omitted.append(f'{f} (unreadable: {e})')
        manifest = [f'report: {name}', f'outcome: {status} (exit code {rc})', f'run: {a.run_name}  preset: {a.preset}',
                    f'job: {job or "-"}  host: {platform.node()}  written: {time.ctime()}',
                    'checkpoints and data are never included; they stay under the run folder.',
                    '', f'included ({len(included)}):', *included,
                    '', f'shortened ({len(shortened)}):', *shortened,
                    '', f'left out because of the size limit ({len(omitted)}):', *omitted]
        payload.insert(0, ('MANIFEST.txt', ('\n'.join(manifest) + '\n').encode('utf-8')))
        tmp = p.outbox / f'.{name}.{os.getpid()}.tmp'
        with tarfile.open(tmp, 'w:gz') as tar:
            for rel, data in payload:
                info = tarfile.TarInfo(rel)
                info.size, info.mtime, info.mode = len(data), time.time(), 0o644
                tar.addfile(info, io.BytesIO(data))
        os.replace(tmp, p.outbox / name)
        latest = p.outbox / f'.LATEST.{os.getpid()}.tmp'
        latest.write_text(name + '\n')
        os.replace(latest, p.outbox / 'LATEST.txt')
        log(f'REPORT: everything needed to see what happened is in {p.outbox / name} '
            f'({(p.outbox / name).stat().st_size / 1024 ** 2:.1f} MB) - send this file to Soheil.')
    except Exception as e:                                  # noqa: BLE001 - must never hide the real outcome
        log(f'(could not write the report bundle: {e!r})')


def export_model_weights(p):
    """A copy of the best checkpoint without optimizer/scheduler state (about a third of the size), for
    sharing and inference; predict.py and inference.py accept it."""
    best, out = p.run / 'best_model.pth', p.results / 'model_weights.pth'
    if not best.exists() or (out.exists() and out.stat().st_mtime >= best.stat().st_mtime):
        return
    try:
        import torch
        ck = torch.load(best, map_location='cpu', weights_only=False)
        slim = {k: v for k, v in ck.items() if k not in ('optimizer', 'scheduler', 'scaler', 'rng')}
        tmp = out.with_name(f'{out.name}.{os.getpid()}.tmp')
        torch.save(slim, tmp)
        os.replace(tmp, out)
        log(f'best model weights (no optimizer state) exported to {out} ({out.stat().st_size / 1024 ** 2:.0f} MB)')
    except Exception as e:                                  # noqa: BLE001
        try:
            tmp.unlink()
        except (NameError, OSError):
            pass
        raise StageError(f'could not export the best model weights to {out} ({e!r}). The training checkpoints '
                         f'are intact; fix the cause (e.g. disk space) and run the same command again - it only '
                         f'redoes the summary.')


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


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='full')
    ap.add_argument('--root', default=os.environ.get('LGEL_ROOT'),
                    help='working directory for ALL data/checkpoints/results (put it on scratch; needs ~200 GB). '
                         'Required for the pilot/full presets (or env LGEL_ROOT).')
    ap.add_argument('--run-name', help='default: <preset>_seed<seed>; with the full preset <preset>_<model>_seed<seed> '
                                       '(full_baseline_seed42, full_advanced_seed42); with the paper preset the '
                                       'prefix of every run name (default paper: paper_baseline_s42, ...)')
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
    g.add_argument('--variant', choices=VARIANTS,
                   help='baseline, advanced (Mamba head + bi-level loss + uncertainty + confidence fusion) or both '
                        '(the baseline first, then the advanced model). Default: both for full, baseline otherwise')
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
    g.add_argument('--train-workers', type=int, help='DataLoader workers (default: min(16, cpus-2))')
    g.add_argument('--pred-batch-size', type=int, default=8)
    g.add_argument('--allow-cpu', action='store_true'); g.add_argument('--offline', action='store_true',
                   help='never touch the network (needs fetch stages done earlier)')
    g.add_argument('--single-job', action='store_true', default=os.environ.get('LGEL_SINGLE_JOB') == '1',
                   help='only needed on file systems without file locks: confirms that only one job at a time '
                        'uses this --root (or env LGEL_SINGLE_JOB=1)')
    g = ap.add_argument_group('stage control')
    g.add_argument('--stages', help='comma-separated subset, e.g. train,predict,evaluate')
    g.add_argument('--from-stage', choices=STAGES); g.add_argument('--to-stage', choices=STAGES)
    g.add_argument('--force', help='comma-separated stages to redo even if finished')
    g.add_argument('--dry-run', action='store_true'); g.add_argument('--list-stages', action='store_true')
    return ap


def main(argv=None):
    global _LOGFILE
    ap = build_parser()
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == '_check_model':     # internal helper
        _check_model(sys.argv[2])
        return 0
    if argv is None and len(sys.argv) > 1 and sys.argv[1] == '_check_advanced':  # internal helper
        _check_advanced(sys.argv[2], *([int(sys.argv[3])] if len(sys.argv) > 3 else []),
                        *(sys.argv[4:5] if len(sys.argv) > 4 else []))
        return 0
    a = ap.parse_args(argv)
    if a.list_stages:
        print('\n'.join(STAGES))
        return 0
    if a.root is None:
        if a.preset != 'smoke':
            ap.error('--root is required for the pilot/full/paper presets: give a folder on a large scratch filesystem '
                     '(~200 GB). The home directory is usually too small, and quotas are not visible to the '
                     'free-space check.')
        a.root = str(REPO / 'lgel_run')
    a = resolve_settings(a)
    subs = experiment_plans(a)
    # The advanced model's add-ons are checked (once per job, before any data is prepared) whenever a run of
    # this command uses one of them, and by the smoke test.
    a.check_advanced = a.preset == 'smoke' or any(sub.kind == 'proposed' and sub.flags for sub in subs)
    for sub in subs:
        sub.check_advanced = a.check_advanced
    plans = [(sub, Paths(a.root, sub.run_name)) for sub in subs]
    p = plans[0][1]                                     # the shared root, logs and state
    selected = select_stages(a)
    redo = []
    for f in filter(None, (a.force or '').split(',')):
        if f not in STAGES:
            raise SystemExit(f'--force: unknown stage {f!r}')
        redo += [s for s in STAGES[STAGES.index(f):] if s not in redo]
    paper = a.preset == 'paper'

    if a.dry_run:                                       # describe only: nothing is created or deleted
        if paper:
            print(f'paper protocol: {len(plans)} runs, in this order (root={p.root}):')
            for i, (sub, sp) in enumerate(plans, 1):
                state = 'finished' if model_finished(sp) else ('started' if sp.run_inputs.exists() else 'to do')
                extra = ' + unseen wordings' if sub.paraphrase_eval else ''
                print(f'  {i:2d}. {sub.run_name:<36} {sub.label} [{sub.kind}, seed {sub.seed}{extra}] - {state}')
            print('then: explanation maps (GPU, minutes) and paper_analysis.py (all tables and figures) -> '
                  f'{p.paper}')
        for i, (sub, sp) in enumerate(plans):
            print(f'preset={a.preset} run={sub.run_name} model={sub.variant} root={sp.root}')
            if i < len(plans) - 1 and not redo and model_finished(sp) and selected[-1] == STAGES[-1]:
                print('  already finished (trained, predicted and evaluated): only its summary would be redone')
                continue
            print('stages: ' + ' -> '.join(selected))
            for s in selected:
                if s == 'check_advanced' and not a.check_advanced:
                    what = 'skip (the advanced model is not trained by this command)'
                elif s in ALWAYS_RUN:
                    what = 'run'
                elif s in redo:
                    what = 'run (forced)'
                elif s == 'train':
                    what = 'run (resumes, or confirms that training is finished)'
                else:
                    what = 'skip (finished)' if sp.marker(s).exists() else 'run'
                print(f'  {s}: {what}')
        if redo:
            print('(--force would first discard the "finished" markers of: ' + ', '.join(redo) + ')')
        return 0

    for d in (p.root, p.state, p.logs, p.tmp):
        d.mkdir(parents=True, exist_ok=True)
    _LOGFILE = p.logs / 'main.log'
    # The scheduler stops a job (time limit, scancel) with SIGTERM: exit through the normal path so
    # that the report bundle is still written. (Exec'd children restore the default handler.)
    # A dropped terminal (SIGHUP) is handled the same way, unless it is deliberately ignored (nohup).
    previous = {}
    for s in STOP_SIGNALS:
        try:
            if s == getattr(signal, 'SIGHUP', None) and signal.getsignal(s) == signal.SIG_IGN:
                continue
            previous[s] = signal.signal(s, _on_sigterm)
        except ValueError:                                # not the main thread (tests)
            pass
    rc = 1
    try:
        failed = []
        for i, (sub, sp) in enumerate(plans):
            if len(plans) > 1:
                log(f'##### run {i + 1} of {len(plans)}: {sub.label} (run {sub.run_name})')
            stages = selected
            if i < len(plans) - 1 and not redo and model_finished(sp) and selected[-1] == STAGES[-1]:
                # Trained, predicted and evaluated: only its summary is redone (which also checks that the
                # data has not changed since, and exports its weights if that was interrupted). Its
                # training settings are not compared, so this command's settings can differ for the next model.
                log(f'{sub.run_name} has already finished (trained, predicted and evaluated): only its summary is redone')
                stages = ['summary']
            # data stages are redone (--force) once; run stages for every model
            rc_run = _drive(sub, sp, stages, redo if i == 0 else [s for s in redo if s in RUN_SCOPED])
            if rc_run == 0:
                continue
            failed.append(sub.run_name)
            if paper and getattr(sub, 'failed_stage', None) in RUN_OWN_FAILURES:
                # Only this run is affected (e.g. a comparison method's download failed): the protocol goes on
                # with the next run, and this one is retried when the same command is submitted again.
                log(f'!! run {sub.run_name} failed (stage {getattr(sub, "failed_stage", None)}); continuing with the next run of the '
                    f'protocol. It is retried, from where it stopped, when the same command is submitted again.')
                continue
            break
        rc = 1 if failed else 0
        if paper and selected[-1] == STAGES[-1] and data_prepared(p):
            rc = max(rc, paper_outputs(a, plans))
            if failed:
                log(f'!! {len(failed)} run(s) did not finish in this job: {failed}. Submit the same command again '
                    f'(finished runs are not repeated).')
        elif rc == 0 and len(plans) > 1 and all(model_finished(sp) for _, sp in plans):   # not after partial runs
            try:
                write_comparison(a, plans)
            except Exception as e:                              # noqa: BLE001
                rc = 1
                log(f'!! could not write the comparison table: {e!r}')
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        log(f'!! stopped (exit code {rc}); resubmit the same command to continue')
        raise
    finally:
        for s in previous:
            signal.signal(s, signal.SIG_IGN)                # a second stop signal must not cut the report short
        write_report_bundle(a, [sp for _, sp in plans], rc)
        for s, handler in previous.items():
            signal.signal(s, handler)
    return rc


# Failures that concern one run only (its own training, prediction, evaluation or summary, or a duplicate
# submission of it). Anything else - preflight, locks, the shared data, the advanced-model check - stops the job.
RUN_OWN_FAILURES = RUN_SCOPED | {'summary', 'duplicate'}


def data_prepared(p):
    return all(p.marker(s).exists() for s in STAGES[1:STAGES.index('train')] if s not in ALWAYS_RUN)


def paper_outputs(a, plans):
    """After the runs of a paper-protocol job: the explanation maps (GPU, minutes; once per checkpoint) and
    paper_analysis.py, which rewrites every table and figure from whatever has finished so far. Neither can
    damage a run; a failure is reported (exit code 1) and the step is simply redone by the next job."""
    p = plans[0][1]
    rc = 0
    by_name = {sub.run_name: (sub, sp) for sub, sp in plans}
    prefix = a.run_name if a.run_name_given else 'paper'
    for key in (f'baseline_s{protocol.MAIN_SEED}', f'advanced_s{protocol.MAIN_SEED}'):
        sub, sp = by_name.get(f'{prefix}_{key}', (None, None))
        if sp is None or not model_finished(sp) or not (sp.run / 'best_model.pth').exists():
            continue
        out = p.paper / 'explanations' / key
        try:
            checkpoint = sp.run / 'best_model.pth'
            done = out / 'done.json'
            if done.exists() and json.loads(done.read_text()).get('checkpoint_sha') == file_sha(checkpoint):
                continue
            run([sys.executable, 'paper_explain.py', '--checkpoint', checkpoint, '--triplets', sp.triplet_csv('test'),
                 '--metadata', sp.metadata, '--frames', sp.frames, '--annotations', sp.parsed, '--output', out,
                 '--checkpoint-sha', file_sha(checkpoint)], a, p, f'paper_explain_{key}')
        except Exception as e:                              # noqa: BLE001
            rc = 1
            log(f'!! explanation maps for {key} failed ({e}); the analysis continues without them')
    try:
        run([sys.executable, 'paper_analysis.py', '--root', p.root, '--prefix', prefix, '--output', p.paper,
             '--sample-fps', a.sample_fps], a, p, 'paper_analysis')
        log(f'PAPER: all tables and figures are in {p.paper} (start with PAPER_RESULTS.md)')
    except Exception as e:                                  # noqa: BLE001
        rc = 1
        log(f'!! paper_analysis.py failed ({e}); the runs themselves are intact - the next job redoes the analysis')
    return rc


STOP_SIGNALS = tuple(s for s in (getattr(signal, 'SIGTERM', None), getattr(signal, 'SIGHUP', None)) if s)


def _on_sigterm(signum, frame):
    for s in STOP_SIGNALS:                                  # one stop is enough; let the clean-up finish
        signal.signal(s, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def _drive(a, p, selected, redo):
    """Run the selected stages for one run. Returns 0 or 1; on failure `a.failed_stage` names where it
    failed (a stage, or 'preflight' / 'locks' / 'force' / 'data' / 'duplicate'), so that the paper protocol
    can tell a failure of this run alone (continue with the next run) from one that affects every run."""
    a.failed_stage = None
    try:
        return _drive_stages(a, p, selected, redo)
    finally:
        p.log_scope = None


def _drive_stages(a, p, selected, redo):
    log(f'preset={a.preset} run={a.run_name} root={p.root}')
    log('stages: ' + ' -> '.join(selected))
    why_not = locks_work(p.state)
    if why_not and not a.single_job:
        log(f'!! {why_not}, so the pipeline cannot stop two jobs from using this --root at the same time.\n'
            f'   Run only one job at a time on this --root (the normal way to use it) and add --single-job\n'
            f'   (or export LGEL_SINGLE_JOB=1) to the command, then submit it again.')
        a.failed_stage = 'locks'
        return 1
    FileLock.disabled = bool(why_not)
    if why_not:
        log(f'--single-job: {why_not}; job locking is off. Never run two jobs on this --root at the same time.')
    prep_lock, run_lock = FileLock(p.prep_lock), FileLock(p.run_lock)
    if redo:
        shared = [s for s in redo if s not in RUN_SCOPED and s not in ALWAYS_RUN]
        if shared:
            try:
                got = prep_lock.acquire(blocking=False)
            except OSError as e:
                log(f'!! could not lock {p.prep_lock} ({e}); resubmit the job.')
                a.failed_stage = 'locks'
                return 1
            if not got:
                log('!! another job is preparing data in this --root right now; --force of data stages has to wait '
                    'until it has finished.')
                a.failed_stage = 'force'
                return 1
            busy = busy_runs(p)
            if busy:
                log(f'!! run(s) {busy} are using the data of this --root right now; --force of data stages would '
                    f'change it under them. Wait until they have finished.')
                a.failed_stage = 'force'
                return 1
            log(f'--force: data stages {shared} will be redone. Runs trained on the previous data cannot be '
                f'continued or evaluated afterwards (they are refused); start new runs instead.')
        for s in redo:
            p.marker(s).unlink(missing_ok=True)
    try:
        duplicate = any(s in RUN_SCOPED for s in selected) and not run_lock.acquire(blocking=False)
    except OSError as e:
        log(f'!! could not lock {p.run_lock} ({e}); resubmit the job, or use --single-job if this keeps happening.')
        a.failed_stage = 'locks'
        return 1
    if duplicate:
        # Refuse a duplicate submission at once, before it waits for or touches anything.
        log(f'!! run "{a.run_name}" is already being processed by another job on this --root. Let that job '
            f'finish (then resubmit if needed), or use a different --run-name.')
        a.failed_stage = 'duplicate'
        return 1

    # Preflight runs first, before anything expensive, and catches the avoidable failures.
    t_all = time.time()
    try:
        stage_preflight(a, p, selected)
    except StageError as e:
        log(f'!! {e}')
        a.failed_stage = 'preflight'
        return 1
    funcs = dict(fetch_models=stage_fetch_models, fetch_weights=stage_fetch_weights,
                 check_advanced=stage_check_advanced, fetch_data=stage_fetch_data,
                 extract_frames=stage_extract_frames, sanitize_annotations=stage_sanitize_annotations,
                 parse_annotations=stage_parse_annotations, splits=stage_splits, triplets=stage_triplets,
                 audit=stage_audit, train=stage_train, predict=stage_predict, evaluate=stage_evaluate,
                 summary=stage_summary)
    for stage in selected:
        if stage == 'preflight':
            continue                                            # already run above
        if stage == 'check_advanced' and not getattr(a, 'check_advanced', False):
            if a.stages:
                log('== check_advanced: skipped (this command does not train the advanced model)')
            continue
        marker = p.marker(stage)
        sig = stage_signature(stage, a)
        if stage in RUN_SCOPED:
            # Wait while another job is (re)building the shared data. Training may only start on
            # finished data: otherwise a later job would rebuild the missing stages underneath it.
            try:
                prep_lock.acquire(blocking=True, waiting_for='preparing the shared data')
            except OSError as e:
                log(f'!! could not lock {p.prep_lock} ({e}); resubmit the job, or use --single-job if this keeps happening.')
                a.failed_stage = 'locks'
                return 1
            unfinished = [s for s in STAGES[1:STAGES.index('train')]
                          if s not in ALWAYS_RUN and not p.marker(s).exists()]
            prep_lock.release()
            if unfinished:
                log(f'!! the prepared data is incomplete: data stage(s) {unfinished} have not finished. Run the full '
                    f'command (without --stages / --from-stage) so the data is prepared first.')
                a.failed_stage = 'data'
                return 1
        elif stage not in ALWAYS_RUN and not marker.exists():
            # One job prepares the shared data; a second job started at the same time waits here and
            # then finds the stages finished.
            try:
                prep_lock.acquire(blocking=True, waiting_for='preparing the shared data')
            except OSError as e:
                log(f'!! could not lock {p.prep_lock} ({e}); resubmit the job, or use --single-job if this keeps happening.')
                a.failed_stage = 'locks'
                return 1
        if stage not in ALWAYS_RUN and marker.exists():
            old = json.loads(marker.read_text())
            if old.get('signature') != sig and stage != 'train':
                log(f'!! stage "{stage}" was already finished with different settings {old.get("signature")} '
                    f'(now {sig}). Use a fresh --root, or --force {stage} to redo it and every later stage.')
                a.failed_stage = stage
                return 1
            if stage != 'train':
                log(f'== {stage}: already done, skipping')
                continue
        log(f'== {stage}: starting')
        t0 = time.time()
        # The step logs of a run's own stages go to logs/<run>/ (the data stages' to logs/).
        p.log_scope = a.run_name if stage in RUN_SCOPED or stage == 'summary' else None
        try:
            funcs[stage](a, p)
        except StageError as e:
            log(f'!! stage {stage} FAILED: {e}')
            log('   Fix the cause and run the same command again: finished stages are skipped and '
                'training resumes from its last finished epoch.')
            a.failed_stage = stage
            return 1
        except Exception as e:                                  # noqa: BLE001
            import traceback
            log(f'!! stage {stage} crashed: {e!r}\n{traceback.format_exc()}')
            a.failed_stage = stage
            return 1
        finally:
            p.log_scope = None
        if stage not in ALWAYS_RUN:
            write_json(marker, dict(signature=sig, finished=time.ctime(), seconds=round(time.time() - t0)))
        log(f'== {stage}: finished in {(time.time() - t0) / 60:.1f} min')
    log(f'ALL DONE in {(time.time() - t_all) / 3600:.2f} h. Results: {p.results}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
