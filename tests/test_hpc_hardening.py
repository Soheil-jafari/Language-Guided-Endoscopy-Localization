"""Final cluster hardening: private links, stopping cleanly, bounded reports, submission guards."""
import argparse
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

import main as pipeline

REPO = Path(pipeline.__file__).resolve().parent
posix_only = pytest.mark.skipif(os.name == 'nt' or not hasattr(os, 'killpg'), reason='POSIX / Linux cluster only')
SECRET = 'https://files.invalid/s/9f8e7d6c5b4a?token=TOPSECRET123'


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setattr(pipeline, '_SECRETS', {})
    pipeline.register_secret(SECRET, 'LGEL_DATA_URL')
    return SECRET


# ------------------------------------------------------------------ 1. private links never leave the job
def test_links_and_url_paths_are_redacted(secret):
    assert pipeline.redact(f'curl -fsSL -C - -o x {secret}') == 'curl -fsSL -C - -o x <LGEL_DATA_URL>'
    out = pipeline.redact('From: https://drive.invalid/uc?id=ABC123 and https://host.invalid')
    assert 'ABC123' not in out and 'https://drive.invalid/...' in out and 'https://host.invalid' in out


def test_stage_logs_and_console_never_show_the_link(tmp_path, secret, capsys):
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(offline=False)
    pipeline._LOGFILE, saved = tmp_path / 'main.log', pipeline._LOGFILE
    try:
        pipeline.run([sys.executable, '-c', 'import sys; print("fetching", sys.argv[1])', secret], a, p, 'fetch')
    finally:
        pipeline._LOGFILE = saved
    texts = [capsys.readouterr().out, (p.logs / 'fetch.log').read_text(), (tmp_path / 'main.log').read_text()]
    assert all('TOPSECRET123' not in t and '9f8e7d6c5b4a' not in t for t in texts)
    assert all('<LGEL_DATA_URL>' in t for t in texts)


def test_report_never_contains_the_link_even_from_an_old_log(tmp_path, secret, monkeypatch):
    monkeypatch.setattr(pipeline.FileLock, 'disabled', False)
    root = tmp_path / 'root'
    (root / 'logs').mkdir(parents=True)
    (root / 'logs' / 'old.log').write_text(f'downloading {SECRET}\n')            # written by an older version
    a = argparse.Namespace(run_name='r', preset='smoke')
    p = pipeline.Paths(root, 'r')
    pipeline.write_report_bundle(a, p, 0)
    name = (p.outbox / 'LATEST.txt').read_text().strip()
    with tarfile.open(p.outbox / name) as tar:
        text = tar.extractfile('logs/old.log').read().decode()
    assert 'TOPSECRET123' not in text and '<LGEL_DATA_URL>' in text


# ------------------------------------------------------------------ 6. bounded, prioritised report
def test_report_is_bounded_prioritised_and_skips_binaries(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'BUNDLE_FILE', 2000)
    monkeypatch.setattr(pipeline, 'BUNDLE_TAIL', 500)
    monkeypatch.setattr(pipeline, 'BUNDLE_TOTAL', 6000)
    p = pipeline.Paths(tmp_path, 'full_seed42')
    p.results.mkdir(parents=True)
    p.logs.mkdir(parents=True)
    p.run.mkdir(parents=True)
    (p.results / 'SUMMARY.md').write_text('# summary\n')
    (p.results / 'training_curves.png').write_bytes(b'\x89PNG' + b'0' * 100)
    (p.results / 'model_weights.pth').write_bytes(b'w' * 100)
    (p.results / 'model_weights.pth.123.tmp').write_bytes(b'w' * 100)          # an interrupted export
    (p.results / 'proposed_test.csv').write_text('x' * 5000)                   # too big for one file
    (p.logs / 'main.log').write_text('start\n' + 'y' * 4000 + '\nTHE END\n')      # long log: tail kept
    for i in range(6):
        (p.logs / f'stage{i}.log').write_text('z' * 1500)                       # fill the total budget
    try:
        (p.logs / 'link.log').symlink_to(p.results / 'SUMMARY.md')       # never followed into a report
    except (OSError, NotImplementedError):                            # Windows without symlink rights
        pass
    a = argparse.Namespace(run_name='full_seed42', preset='full')
    pipeline.write_report_bundle(a, p, 1)
    name = (p.outbox / 'LATEST.txt').read_text().strip()
    assert name.endswith('-FAILED.tar.gz')
    with tarfile.open(p.outbox / name) as tar:
        names = tar.getnames()
        manifest = tar.extractfile('MANIFEST.txt').read().decode()
        main_log = tar.extractfile('logs/main.log').read().decode()
        payload = sum(m.size for m in tar.getmembers() if m.name != 'MANIFEST.txt')
    assert 'results/full_seed42/SUMMARY.md' in names and 'results/full_seed42/training_curves.png' in names
    assert not [n for n in names if '.pth' in n or n.endswith('.tmp')]
    assert 'logs/link.log' not in names
    assert 'results/full_seed42/proposed_test.csv' not in names and 'proposed_test.csv' in manifest
    assert main_log.rstrip().endswith('THE END') and 'start' not in main_log and 'logs/main.log (last' in manifest
    assert payload <= 6000 and 'left out because of the size limit' in manifest


# ------------------------------------------------------------------ 7. a failed export is a failure
def test_failed_weights_export_fails_the_summary_and_leaves_no_temp_file(tmp_path, monkeypatch):
    import torch
    p = pipeline.Paths(tmp_path, 'r')
    p.run.mkdir(parents=True)
    p.results.mkdir(parents=True)
    torch.save(dict(schema_version=2, config={}, model_state_dict={}, optimizer={}), p.run / 'best_model.pth')
    def disk_full(obj, path, *a, **k):
        Path(path).write_bytes(b'partial')
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(torch, 'save', disk_full)
    with pytest.raises(pipeline.StageError, match='could not export'):
        pipeline.export_model_weights(p)
    assert not list(p.results.glob('model_weights*'))


# ------------------------------------------------------------------ 4. a stop ends the running step first
def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                                    # a zombie is not alive
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except OSError:
        return True


@posix_only
def test_a_stop_ends_the_running_step_and_everything_it_started(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(offline=False)
    pidfile = tmp_path / 'pids'
    script = f'sleep 60 & echo $$ $! > {pidfile}; wait'
    previous = signal.signal(signal.SIGTERM, pipeline._on_sigterm)
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.time()
    try:
        with pytest.raises(SystemExit) as e:
            pipeline.run(['bash', '-c', script], a, p, 'step')
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert e.value.code == 143 and time.time() - started < 20
    shell, sleeper = map(int, pidfile.read_text().split())
    time.sleep(0.5)
    assert not _alive(shell) and not _alive(sleeper)       # the step's own children are gone too


@posix_only
def test_a_stop_terminates_running_extraction_workers(tmp_path, monkeypatch):
    p = pipeline.Paths(tmp_path, 'r')
    inv = {}
    for v in ('video01', 'video02', 'video03', 'video04'):
        (tmp_path / f'{v}.mp4').write_bytes(b'x')
        inv[v] = dict(video=str(tmp_path / f'{v}.mp4'), phase='x', tool='x')
    pipeline.write_json(p.inventory, dict(base=str(tmp_path), videos=inv))
    pidfile = tmp_path / 'worker_pids'
    monkeypatch.setattr(pipeline, '_extract_one', _slow_extract)
    monkeypatch.setenv('LGEL_TEST_PIDFILE', str(pidfile))
    a = argparse.Namespace(sample_fps=1.0, workers=2, cleanup_raw=False, data_dir=None, data_zip=None)
    previous = signal.signal(signal.SIGTERM, pipeline._on_sigterm)
    threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    started = time.time()
    try:
        with pytest.raises(SystemExit):
            pipeline.stage_extract_frames(a, p)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert time.time() - started < 20                       # not after the 60 s "videos" finished
    pids = [int(x) for x in pidfile.read_text().split()]
    time.sleep(0.5)
    assert pids and not any(_alive(x) for x in pids)


def _slow_extract(job):
    with open(os.environ['LGEL_TEST_PIDFILE'], 'a') as f:
        f.write(f'{os.getpid()}\n')
    time.sleep(60)


# ------------------------------------------------------------------ 8. options cannot be abbreviated
def test_options_cannot_be_abbreviated(tmp_path):
    with pytest.raises(SystemExit) as e:
        pipeline.main(['--preset', 'smoke', '--ro', str(tmp_path), '--dry-run'])
    assert e.value.code == 2


# ------------------------------------------------------------------ 2 + 8. submit.sh guards
def _fake_slurm(tmp_path, squeue='', squeue_rc=0, sbatch_out=None):
    stub = tmp_path / 'bin'
    stub.mkdir(parents=True, exist_ok=True)
    answer = f'echo "{sbatch_out}"' if sbatch_out else 'echo "$((1000 + $(wc -l < "$STUB_LOG")))"'
    (stub / 'sbatch').write_text(f'#!/bin/bash\necho "$@" >> "$STUB_LOG"\n{answer}\n')
    (stub / 'squeue').write_text(f'#!/bin/bash\nprintf "%s" "{squeue}"\nexit {squeue_rc}\n')
    for f in stub.iterdir():
        f.chmod(0o755)
    return stub


def _submit(tmp_path, *args, base=None, **slurm):
    stub = _fake_slurm(tmp_path, **slurm)
    base = base or tmp_path / 'base'
    log = tmp_path / 'sbatch.log'
    log.write_text('')
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', LGEL_BASE=str(base), STUB_LOG=str(log),
               USER='tester')
    env.update(slurm.get('env', {}) if isinstance(slurm.get('env'), dict) else {})
    out = subprocess.run(['bash', str(REPO / 'hpc' / 'submit.sh'), *args], env=env, capture_output=True, text=True)
    return out, [l for l in log.read_text().splitlines() if l]


def _links(base, data='https://d.invalid/d.zip', weights='https://w.invalid/w.pth'):
    base.mkdir(parents=True, exist_ok=True)
    (base / 'links.env').write_text(f"LGEL_DATA_URL='{data}'\nLGEL_WEIGHTS_URL='{weights}'\n")
    (base / 'links.env').chmod(0o600)


@posix_only
@pytest.mark.parametrize('args,message', [
    (['--repeat', '0'], 'between 1 and 100'),
    (['--repeat', 'two'], 'between 1 and 100'),
    (['--root', '/elsewhere'], 'set by this script'),
    (['--preset=pilot'], 'set by this script'),
    (['--data-url', 'https://x.invalid/y'], 'never on the command line'),
    (['--weights-url=https://x.invalid/y'], 'never on the command line'),
])
def test_submit_refuses_bad_or_conflicting_arguments(tmp_path, args, message):
    _links(tmp_path / 'base')
    out, calls = _submit(tmp_path, 'full', *args)
    assert out.returncode != 0 and message in out.stderr and not calls


@posix_only
def test_submit_fails_closed_when_the_queue_cannot_be_read(tmp_path):
    _links(tmp_path / 'base')
    out, calls = _submit(tmp_path, 'full', squeue='slurm_load_jobs error: Unable to contact controller', squeue_rc=1)
    assert out.returncode == 1 and 'could not ask Slurm' in out.stderr and not calls


@posix_only
def test_submit_refuses_empty_links_and_paths_with_commas(tmp_path):
    _links(tmp_path / 'base', weights='')
    out, calls = _submit(tmp_path, 'pilot')
    assert out.returncode == 1 and 'link is empty' in out.stderr and not calls
    out, calls = _submit(tmp_path, 'smoke', base=tmp_path / 'a,b')
    assert out.returncode == 1 and 'commas' in out.stderr and not calls


@posix_only
def test_submit_lock_blocks_a_concurrent_submission_but_not_a_dead_one(tmp_path):
    base = tmp_path / 'base'
    _links(base)
    lock = base / '.submit.lock'
    lock.mkdir()
    holder = subprocess.Popen(['sleep', '30'])
    try:
        (lock / 'owner').write_text(f'{os.uname().nodename} {holder.pid}\n')
        out, calls = _submit(tmp_path, 'smoke')
        assert out.returncode == 1 and 'another submit.sh is running' in out.stderr and not calls
    finally:
        holder.kill()
        holder.wait()
    out, calls = _submit(tmp_path, 'smoke')                            # its owner is gone: lock reclaimed
    assert out.returncode == 0, out.stderr
    assert len(calls) == 1 and not lock.exists()


@posix_only
def test_submit_validates_job_ids_and_passes_site_options(tmp_path):
    _links(tmp_path / 'base')
    out, calls = _submit(tmp_path, 'smoke', sbatch_out='sbatch: error: invalid account')
    assert out.returncode == 1 and 'unexpected answer from sbatch' in out.stderr
    stub = _fake_slurm(tmp_path / 'x')
    log = tmp_path / 'x.log'
    log.write_text('')
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', LGEL_BASE=str(tmp_path / 'base'),
               STUB_LOG=str(log), USER='tester', LGEL_SBATCH_OPTS='--account=proj1 --qos=normal')
    out = subprocess.run(['bash', str(REPO / 'hpc' / 'submit.sh'), 'smoke'], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '--account=proj1 --qos=normal' in log.read_text()


# ------------------------------------------------------------------ 3 + 4. the batch job: stop forwarding and fallback report
def _fake_repo(tmp_path, run_sh):
    repo = tmp_path / 'repo'
    (repo / 'hpc').mkdir(parents=True)
    for f in ('lgel.sbatch', 'fallback_report.sh'):
        (repo / 'hpc' / f).write_text((REPO / 'hpc' / f).read_text())
    (repo / 'run.sh').write_text(run_sh)
    return repo


def _job(tmp_path, repo, preset='smoke', args=(), module_fails=False, signal_after=None, links=True):
    stub = tmp_path / 'bin'
    stub.mkdir(exist_ok=True)
    (stub / 'module').write_text('#!/bin/bash\n' + ('[ "$2" = apps/miniconda3 ] && exit 1\n' if module_fails else '')
                                 + 'echo "module $*"\n')
    (stub / 'module').chmod(0o755)
    base = tmp_path / 'base'
    base.mkdir(exist_ok=True)
    if links:
        _links(base, data=SECRET)
    root = base / ('lgel_smoke' if preset == 'smoke' else 'lgel')
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PATH=f'{stub}:{os.environ["PATH"]}', LGEL_REPO=str(repo), LGEL_BASE=str(base),
               LGEL_PRESET=preset, LGEL_ROOT=str(root), SLURM_JOB_ID='555')
    env.pop('LGEL_WEIGHTS_URL', None)
    env.pop('LGEL_DATA_URL', None)
    logf = open(root / 'logs' / 'slurm-555.out', 'w')
    proc = subprocess.Popen(['bash', str(repo / 'hpc' / 'lgel.sbatch'), *args], env=env, stdout=logf,
                            stderr=subprocess.STDOUT, start_new_session=True)
    if signal_after is not None:
        time.sleep(signal_after)
        os.kill(proc.pid, signal.SIGUSR1)                     # Slurm's early warning, to the batch shell only
    rc = proc.wait(timeout=120)
    logf.close()
    return rc, root, (root / 'logs' / 'slurm-555.out').read_text()


FAKE_PIPELINE = '''#!/bin/bash
echo "ARGS: $*"
trap 'echo "pipeline: stopping"; mkdir -p "$LGEL_ROOT/outbox"; : > "$LGEL_ROOT/outbox/r-job555-STOPPED.tar.gz"; exit 143' TERM
sleep 60 & wait
'''


@posix_only
def test_early_warning_stops_the_pipeline_which_reports_itself(tmp_path):
    repo = _fake_repo(tmp_path, FAKE_PIPELINE)
    rc, root, log = _job(tmp_path, repo, signal_after=2)
    assert rc == 143
    assert 'the time limit is near (USR1)' in log and 'pipeline: stopping' in log
    reports = sorted(x.name for x in (root / 'outbox').glob('*.tar.gz'))
    assert reports == ['r-job555-STOPPED.tar.gz']           # no extra fallback report


@posix_only
def test_a_job_that_cannot_load_conda_still_leaves_a_report(tmp_path):
    repo = _fake_repo(tmp_path, FAKE_PIPELINE)
    rc, root, log = _job(tmp_path, repo, preset='pilot', module_fails=True)
    assert rc == 3 and "'module load apps/miniconda3' failed" in log
    name = (root / 'outbox' / 'LATEST.txt').read_text().strip()
    assert name.startswith('pilot-') and name.endswith('-job555-SETUP-FAILED.tar.gz')
    with tarfile.open(root / 'outbox' / name) as tar:
        tail = tar.extractfile('./slurm-555.out.tail.txt').read().decode()
    assert 'miniconda3' in tail


@posix_only
def test_a_missing_link_fails_in_seconds_with_a_report(tmp_path):
    repo = _fake_repo(tmp_path, FAKE_PIPELINE)
    rc, root, log = _job(tmp_path, repo, preset='full', links=False)
    assert rc == 2 and 'no dataset link' in log and 'ARGS:' not in log
    assert (root / 'outbox' / 'LATEST.txt').read_text().strip().endswith('-job555-SETUP-FAILED.tar.gz')


@posix_only
def test_smoke_respects_local_weights_and_redacts_links_in_fallback_reports(tmp_path):
    repo = _fake_repo(tmp_path, '#!/bin/bash\necho "ARGS: $*"\necho "using $LGEL_DATA_URL"\nexit 7\n')
    rc, root, log = _job(tmp_path, repo, args=('--weights-path', '/w/m2crl.pth'))
    assert rc == 7
    args_line = next(l for l in log.splitlines() if l.startswith('ARGS:'))
    assert '--synthetic 6' in args_line and '--random-init' not in args_line and '--single-job' in args_line
    name = (root / 'outbox' / 'LATEST.txt').read_text().strip()
    with tarfile.open(root / 'outbox' / name) as tar:
        tail = tar.extractfile('./slurm-555.out.tail.txt').read().decode()
    assert 'TOPSECRET123' not in tail and 'https://files.invalid/...' in tail


# ------------------------------------------------------------------ round 9: links off the process list, SIGHUP, redaction
def test_redaction_also_hides_credentials_and_query_strings_without_a_path():
    assert pipeline.redact('https://user:p4ss@files.invalid/data.zip') == 'https://files.invalid/...'
    assert pipeline.redact('https://cdn.invalid?X-Amz-Signature=S3CR3T') == 'https://cdn.invalid/...'
    assert pipeline.redact('{"url": "https://h.invalid/a?b=1"}') == '{"url": "https://h.invalid/..."}'


def test_download_commands_never_carry_the_link(tmp_path, monkeypatch):
    seen = []
    def fake_run(cmd, a, p, name, **kw):
        seen.append(([str(c) for c in cmd], dict(kw.get('env') or {})))
        files = [Path(c) for c in cmd if str(c).endswith('.txt') and Path(c).exists()]
        seen[-1] += (''.join(f.read_text() for f in files),)
        Path(cmd[cmd.index('-o') + 1] if '-o' in cmd else cmd[cmd.index('-O') + 1] if '-O' in cmd else cmd[-1]).write_bytes(b'PK\x03\x04data')
    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(pipeline, 'remote_size', lambda url, p: None)
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(offline=False)
    pipeline.download(SECRET, tmp_path / 'one.zip', a, p, 'x')                        # curl
    cmd, env, conf = seen[-1]
    assert SECRET not in ' '.join(cmd) and '-K' in cmd and SECRET in conf          # link only in the 0600 file
    assert '--speed-time' in cmd                                                   # a stalled transfer is retried
    real_which = pipeline.shutil.which
    monkeypatch.setattr(pipeline.shutil, 'which', lambda n: None if n == 'curl' else real_which(n) or '/usr/bin/' + n)
    pipeline.download(SECRET, tmp_path / 'two.zip', a, p, 'x')                        # wget
    cmd, env, conf = seen[-1]
    assert cmd[0] == 'wget' and SECRET not in ' '.join(cmd) and '-i' in cmd and SECRET in conf
    monkeypatch.setattr(pipeline.shutil, 'which', real_which)
    drive = 'https://drive.google.com/file/d/SECRETFILEID/view'
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0))
    pipeline.download(drive, tmp_path / 'three.pth', a, p, 'x')                       # Google Drive
    cmd, env, conf = seen[-1]
    assert drive not in ' '.join(cmd) and env.get('LGEL_GDOWN_URL') == drive
    assert not list(p.tmp.glob('.link-*'))                                          # temporary link files removed


@posix_only
def test_a_dropped_terminal_stops_the_step_like_sigterm(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(offline=False)
    pidfile = tmp_path / 'pids'
    previous = {s: signal.signal(s, pipeline._on_sigterm) for s in pipeline.STOP_SIGNALS}
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGHUP)).start()
    try:
        with pytest.raises(SystemExit) as e:
            pipeline.run(['bash', '-c', f'sleep 60 & echo $$ $! > {pidfile}; wait'], a, p, 'step')
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    assert e.value.code == 129
    shell, sleeper = map(int, pidfile.read_text().split())
    time.sleep(0.5)
    assert not _alive(shell) and not _alive(sleeper)


@posix_only
def test_fallback_report_tells_a_crash_from_a_setup_failure(tmp_path):
    root = tmp_path / 'root'
    (root / 'logs').mkdir(parents=True)
    start = root / 'logs' / '.job-9.started'
    start.write_text('')
    os.utime(start, (time.time() - 60, time.time() - 60))
    log = root / 'logs' / 'slurm-9.out'
    log.write_text('training ...\nKilled\n')
    script = str(REPO / 'hpc' / 'fallback_report.sh')
    subprocess.run(['bash', script, str(root), '9', '1', str(log), 'pilot', str(start)], check=True)
    assert (root / 'outbox' / 'LATEST.txt').read_text().strip().endswith('-job9-SETUP-FAILED.tar.gz')
    for f in (root / 'outbox').glob('*.tar.gz'):
        f.unlink()
    (root / 'logs' / 'main.log').write_text('== train: starting\n')                    # the pipeline did run
    subprocess.run(['bash', script, str(root), '9', '137', str(log), 'pilot', str(start)], check=True)
    assert (root / 'outbox' / 'LATEST.txt').read_text().strip().endswith('-job9-FAILED.tar.gz')


# ------------------------------------------------------------------ round 10: lost console, gdown versions, stalls, env stamp
class _DyingTerminal:
    """A terminal that works until it is hung up; afterwards every write fails like a closed tty."""
    def __init__(self):
        self.dead, self.text = threading.Event(), []

    def write(self, s):
        if self.dead.is_set():
            raise OSError(5, 'Input/output error')
        self.text.append(s)
        return len(s)

    def flush(self):
        if self.dead.is_set():
            raise OSError(5, 'Input/output error')


@posix_only
def test_a_lost_terminal_neither_stops_the_pipeline_nor_prevents_a_clean_stop(tmp_path, monkeypatch):
    term = _DyingTerminal()
    monkeypatch.setattr(sys, 'stdout', term)
    monkeypatch.setattr(pipeline, '_CONSOLE_LOST', False)
    monkeypatch.setattr(pipeline, '_LOGFILE', tmp_path / 'main.log')
    p = pipeline.Paths(tmp_path, 'r')
    a = argparse.Namespace(offline=False)
    pipeline.log('before the hangup')
    term.dead.set()
    pipeline.log('after the hangup')                        # must not raise
    pipeline.run([sys.executable, '-c', 'print("still working")'], a, p, 'quiet')
    assert 'still working' in (p.logs / 'quiet.log').read_text()
    # A stop arriving with the terminal gone still ends the step and everything it started.
    pidfile = tmp_path / 'pids'
    previous = {s: signal.signal(s, pipeline._on_sigterm) for s in pipeline.STOP_SIGNALS}
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGHUP)).start()
    try:
        with pytest.raises(SystemExit) as e:
            pipeline.run(['bash', '-c', f'sleep 60 & echo $$ $! > {pidfile}; wait'], a, p, 'step')
    finally:
        for s, h in previous.items():
            signal.signal(s, h)
    assert e.value.code == 129
    shell, sleeper = map(int, pidfile.read_text().split())
    time.sleep(0.5)
    assert not _alive(shell) and not _alive(sleeper)
    text = (tmp_path / 'main.log').read_text()
    assert 'before the hangup' in text and 'after the hangup' in text and 'stopping step' in text
    assert 'before the hangup' in ''.join(term.text)


HANGUP_CHILD = r'''
import argparse, os, signal, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import main as pipeline
root = Path(sys.argv[2])
p = pipeline.Paths(root, 'r')
p.logs.mkdir(parents=True, exist_ok=True)
pipeline._LOGFILE = p.logs / 'main.log'
for s in pipeline.STOP_SIGNALS:
    signal.signal(s, pipeline._on_sigterm)
code = 1
try:
    pipeline.run(['bash', '-c', 'sleep 60 & echo $$ $! > ' + str(root / 'pids') + '; wait'],
                 argparse.Namespace(offline=False), p, 'step')
except SystemExit as e:
    code = e.code
finally:
    pipeline.write_report_bundle(argparse.Namespace(run_name='r', preset='smoke'), p, code)
sys.exit(code)
'''


@posix_only
def test_a_real_terminal_hangup_stops_the_step_and_still_reports(tmp_path):
    import pty
    import select
    root = tmp_path / 'root'
    script = tmp_path / 'child.py'
    script.write_text(HANGUP_CHILD)
    pid, fd = pty.fork()                                    # the child gets the pty as its terminal
    if pid == 0:
        try:
            os.execv(sys.executable, [sys.executable, str(script), str(REPO), str(root)])
        finally:
            os._exit(99)
    pids = root / 'pids'
    deadline = time.time() + 60
    while time.time() < deadline and not (pids.exists() and pids.read_text().strip()):
        if select.select([fd], [], [], 0.1)[0]:
            try:
                os.read(fd, 65536)
            except OSError:
                break
    time.sleep(0.5)
    os.close(fd)                                            # the terminal goes away (a dropped SSH session)
    deadline, status = time.time() + 60, None
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail('the job did not stop after the hangup')
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 129
    shell, sleeper = map(int, pids.read_text().split())
    time.sleep(0.5)
    assert not _alive(shell) and not _alive(sleeper)
    assert 'stopping step' in (root / 'logs' / 'main.log').read_text()
    assert (root / 'outbox' / 'LATEST.txt').read_text().strip().endswith('-STOPPED.tar.gz')


@posix_only
def test_extraction_workers_take_the_default_action_for_every_stop_signal():
    previous = {s: signal.signal(s, pipeline._on_sigterm) for s in pipeline.STOP_SIGNALS}
    try:
        pipeline._worker_init()
        assert all(signal.getsignal(s) == signal.SIG_DFL for s in pipeline.STOP_SIGNALS)
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


FAKE_GDOWN = '''import json, os
__version__ = {version!r}
def download(url=None, output=None, quiet=False, {params}):
    kw = dict(quiet=quiet, {passed})
    with open(os.environ["FAKE_GDOWN_RECORD"], "w") as f:
        json.dump(dict(url=url, kw=kw), f)
    with open(output, "wb") as f:
        f.write(b"PK\\x03\\x04data")
    return output
'''


@pytest.mark.parametrize('version,params,expect', [
    ('6.4.1', 'resume=False, timeout=None, retries=0', dict(quiet=True, resume=True, timeout=[60, 600], retries=20)),
    ('5.2.0', 'fuzzy=False, resume=False', dict(quiet=True, fuzzy=True, resume=True)),
    ('4.7.3', 'fuzzy=False, resume=False', None),
])
def test_google_drive_downloads_pass_only_what_the_installed_gdown_accepts(tmp_path, monkeypatch, version, params,
                                                                          expect):
    import json
    names = [x.split('=')[0].strip() for x in params.split(',')]
    pkg = tmp_path / 'fake' / 'gdown'
    pkg.mkdir(parents=True)
    (pkg / '__init__.py').write_text(FAKE_GDOWN.format(version=version, params=params,
                                                       passed=', '.join(f'{n}={n}' for n in names)))
    seen, real_run = [], subprocess.run
    monkeypatch.setattr(pipeline, 'run', lambda cmd, a, p, name, **kw: seen.append((cmd, kw['env'])))
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0))
    drive = 'https://drive.google.com/file/d/SECRETFILEID/view?usp=sharing'
    pipeline._download(drive, tmp_path / 'checkpoint.pth', argparse.Namespace(offline=False),
                       pipeline.Paths(tmp_path, 'r'), 'x')
    cmd, env = seen[-1]
    env = dict(env, PYTHONPATH=str(tmp_path / 'fake'), FAKE_GDOWN_RECORD=str(tmp_path / 'record.json'))
    done = real_run([str(c) for c in cmd], env=env, capture_output=True, text=True)
    if expect is None:
        assert done.returncode != 0 and 'too old' in done.stderr and not (tmp_path / 'record.json').exists()
    else:
        assert done.returncode == 0, done.stderr
        record = json.loads((tmp_path / 'record.json').read_text())
        assert record == dict(url=drive, kw=expect)


def test_fallback_report_sed_delimiter_never_occurs_inside_the_expression():
    line = next(l for l in (REPO / 'hpc' / 'fallback_report.sh').read_text().splitlines() if l.startswith('redact()'))
    expr = line.split("sed -E '", 1)[1].split("'", 1)[0]
    assert expr.startswith('s') and expr.count(expr[1]) == 3   # s<d>pattern<d>replacement<d>g


@posix_only
def test_environment_stamp_is_one_fingerprint_of_requirements_and_setup_script():
    import hashlib
    def stamp(path):
        line = next(l for l in (REPO / path).read_text().splitlines() if 'hashlib' in l)
        return line[line.index("-c '"):].split(' 2>/dev/null')[0].split(' > ')[0]
    assert stamp('run.sh') == stamp('setup_env.sh')
    out = subprocess.run(['bash', '-c', f'REPO="{REPO}"; "{sys.executable}" {stamp("run.sh")}'],
                         capture_output=True, text=True, check=True).stdout.strip()
    expected = hashlib.sha256((REPO / 'requirements.txt').read_bytes() + (REPO / 'setup_env.sh').read_bytes())
    assert out == expected.hexdigest()


def test_heartbeat_reports_the_size_of_a_partial_google_drive_download(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_LOGFILE', tmp_path / 'main.log')
    (tmp_path / 'checkpoint.pthab12cd.part').write_bytes(b'x' * 1000)     # gdown's file until it is done
    pipeline.run([sys.executable, '-c', 'import time; time.sleep(1.5)'], argparse.Namespace(offline=False),
                 pipeline.Paths(tmp_path, 'r'), 'fetch', progress_path=tmp_path / 'checkpoint.pth', heartbeat=0.3)
    assert 'GB so far' in (tmp_path / 'main.log').read_text()


# ------------------------------------------------------------------ round 11: link parts, curl continuation, nohup, Drive
def test_parts_of_a_link_that_identify_the_file_are_hidden_too(monkeypatch):
    monkeypatch.setattr(pipeline, '_SECRETS', {})
    drive = 'https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz012345/view?usp=sharing'
    pipeline.register_secret(drive, 'LGEL_WEIGHTS_URL')
    pipeline.register_secret('https://h.invalid/scl/fi/k3j4h5g6f7/data.zip?rlkey=abcd1234efgh5678&dl=1', 'LGEL_DATA_URL')
    error = "Max retries exceeded with url: /uc?id=1AbCdEfGhIjKlMnOpQrStUvWxYz012345 (Caused by NameResolutionError)"
    assert '1AbCdEfGh' not in pipeline.redact(error) and '<LGEL_WEIGHTS_URL part>' in pipeline.redact(error)
    assert pipeline.redact(f'from {drive} to x') == 'from <LGEL_WEIGHTS_URL> to x'      # whole link first
    assert 'abcd1234efgh5678' not in pipeline.redact('?rlkey=abcd1234efgh5678')
    ordinary = 'downloading weights/checkpoint.pth and data.zip (usp=sharing, dl=1, export=download)'
    assert pipeline.redact(ordinary) == ordinary                                       # ordinary words stay


def _fake_curl(tmp_path, monkeypatch, outcomes, total):
    """pipeline.run stand-in: each call appends some bytes to the output file, then 'exits' with a code."""
    calls = []
    def fake_run(cmd, a, p, name, **kw):
        cmd = [str(c) for c in cmd]
        dest = Path(cmd[cmd.index('-o') + 1])
        calls.append(dict(resume='-C' in cmd, size_before=dest.stat().st_size if dest.exists() else 0))
        add, code = outcomes[len(calls) - 1]
        with open(dest, 'ab') as f:
            f.write(b'x' * add)
        if code:
            err = pipeline.StageError(f'curl exited with code {code}')
            err.rc = code
            raise err
    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(pipeline, 'remote_size', lambda url, p: total)
    monkeypatch.setattr(pipeline.time, 'sleep', lambda s: None)
    monkeypatch.setattr(pipeline.shutil, 'which', lambda n: '/usr/bin/curl' if n == 'curl' else None)
    return calls


def test_a_stalled_or_broken_download_continues_from_the_bytes_already_on_disk(tmp_path, monkeypatch):
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(b'x' * 100)                                      # from an earlier job
    calls = _fake_curl(tmp_path, monkeypatch, [(30, 28), (30, 56), (40, 0)], total=200)
    pipeline._download(SECRET, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
    assert [c['size_before'] for c in calls] == [100, 130, 160] and all(c['resume'] for c in calls)
    assert dest.stat().st_size == 200


def test_only_a_server_without_range_support_restarts_from_zero(tmp_path, monkeypatch):
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(b'x' * 100)
    calls = _fake_curl(tmp_path, monkeypatch, [(0, 33), (200, 0)], total=200)
    pipeline._download(SECRET, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
    assert [(c['resume'], c['size_before']) for c in calls] == [(True, 100), (False, 0)]
    assert dest.stat().st_size == 200


def test_a_download_that_keeps_failing_gives_up_but_keeps_the_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'DOWNLOAD_ATTEMPTS', 3)
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(b'x' * 100)
    calls = _fake_curl(tmp_path, monkeypatch, [(0, 22)] * 3, total=200)
    with pytest.raises(pipeline.StageError):
        pipeline._download(SECRET, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
    assert len(calls) == 3 and dest.stat().st_size == 100              # kept for the next job


@posix_only
@pytest.mark.skipif(not __import__('shutil').which('curl'), reason='needs curl')
@pytest.mark.parametrize('ranges', [True, False])
def test_real_curl_continues_after_stalls(tmp_path, monkeypatch, ranges):
    import http.server
    import re as _re
    blob = os.urandom(3_000_000)
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(blob)))
            self.end_headers()

        def do_GET(self):
            m = _re.match(r'bytes=(\d+)-', self.headers.get('Range') or '')
            start = int(m.group(1)) if (m and ranges) else 0
            seen.append((bool(m), start))
            if m and ranges:
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{len(blob) - 1}/{len(blob)}')
            else:
                self.send_response(200)
            self.send_header('Content-Length', str(len(blob) - start))
            self.end_headers()
            body = blob[start:]
            if len(seen) <= 2:                                       # the first two responses stall
                self.wfile.write(body[:700_000])
                self.wfile.flush()
                time.sleep(6)
                return
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pipeline, 'STALL_SECONDS', 2)
    monkeypatch.setattr(pipeline, 'DOWNLOAD_PAUSE', 0)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    p = pipeline.Paths(tmp_path, 'r')
    dest = tmp_path / 'dataset.zip'
    try:
        pipeline._download(f'http://127.0.0.1:{server.server_address[1]}/d.zip', dest,
                           argparse.Namespace(offline=False), p, 'fetch_data')
    finally:
        server.shutdown()
    assert dest.read_bytes() == blob
    if ranges:                                                       # each attempt continued, none restarted
        assert seen == [(False, 0), (True, 700_000), (True, 1_400_000)]
    else:                                                            # range refused: the whole file once more
        assert seen == [(False, 0), (True, 0), (False, 0)]


@posix_only
def test_workers_keep_ignoring_a_hangup_under_nohup():
    previous = {signal.SIGTERM: signal.signal(signal.SIGTERM, pipeline._on_sigterm),
                signal.SIGHUP: signal.signal(signal.SIGHUP, signal.SIG_IGN)}
    try:
        pipeline._worker_init()
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL and signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        for s, h in previous.items():
            signal.signal(s, h)


def test_google_drive_download_never_trusts_an_unverified_earlier_file(tmp_path, monkeypatch):
    dest = tmp_path / 'checkpoint.pth'
    dest.write_bytes(b'partial download from an earlier direct link')
    seen = []
    monkeypatch.setattr(pipeline, 'run', lambda cmd, a, p, name, **kw: seen.append(dest.exists()))
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0))
    pipeline._download('https://drive.google.com/file/d/X1234567890abc/view', dest, argparse.Namespace(offline=False),
                       pipeline.Paths(tmp_path, 'r'), 'x')
    assert seen == [False]                                           # gdown would have kept it as "finished"


def test_disk_check_counts_a_partial_google_drive_download(tmp_path):
    p = pipeline.Paths(tmp_path, 'r')
    p.raw.mkdir(parents=True)
    (p.raw / 'dataset.zipab12cd.part').write_bytes(b'z' * 4 * 1024 ** 2)
    a = argparse.Namespace(min_free_gb=50, synthetic=None, data_dir=None, data_zip=None)
    need, _ = pipeline.disk_needed_gb(a, p, list(pipeline.STAGES))
    assert need == pytest.approx(50 - 4 / 1024)


# ------------------------------------------------------------------ round 12: a broken link never costs the partial download
@pytest.mark.parametrize('headers,expected', [
    ('HTTP/1.1 403 Forbidden\r\nContent-Type: text/html\r\nContent-Length: 115\r\n\r\n', None),     # expired link
    ('HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n', None),                        # HEAD refused
    ('HTTP/1.1 302 Found\r\nLocation: x\r\nContent-Length: 0\r\n\r\nHTTP/1.1 200 OK\r\n\r\n', None),  # size unknown
    ('HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: 5120\r\n\r\n', None),
    ('HTTP/1.1 200 Connection established\r\n\r\nHTTP/2 302\r\ncontent-length: 0\r\nlocation: y\r\n\r\n'
     'HTTP/2 200\r\ncontent-type: application/zip\r\ncontent-length: 20000000\r\n\r\n', 20000000),
])
def test_the_file_size_is_only_taken_from_a_successful_final_response(tmp_path, monkeypatch, headers, expected):
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, headers, ''))
    assert pipeline.remote_size(SECRET, pipeline.Paths(tmp_path, 'r')) == expected


@posix_only
@pytest.mark.skipif(not __import__('shutil').which('curl'), reason='needs curl')
@pytest.mark.parametrize('mode', ['expired-403', 'expired-html', 'head-refused'])
def test_a_broken_link_never_costs_the_partial_download(tmp_path, monkeypatch, mode):
    import http.server
    import re as _re
    blob = os.urandom(2_000_000)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def page(self, code):
            body = b'<!DOCTYPE html><html><body>This link has expired</body></html>'
            self.send_response(code)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if self.command == 'GET':
                self.wfile.write(body)

        def do_HEAD(self):
            if mode == 'expired-403':
                return self.page(403)
            if mode == 'expired-html':
                return self.page(200)
            self.page(405)                                           # head-refused: GET still works

        def do_GET(self):
            if mode == 'expired-403':
                return self.page(403)
            if mode == 'expired-html':
                return self.page(200)
            start = int(_re.match(r'bytes=(\d+)-', self.headers['Range']).group(1)) if self.headers.get('Range') else 0
            self.send_response(206 if start else 200)
            if start:
                self.send_header('Content-Range', f'bytes {start}-{len(blob) - 1}/{len(blob)}')
            self.send_header('Content-Length', str(len(blob) - start))
            self.end_headers()
            self.wfile.write(blob[start:])

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pipeline, 'DOWNLOAD_ATTEMPTS', 2)
    monkeypatch.setattr(pipeline, 'DOWNLOAD_PAUSE', 0)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(blob[:1_000_000])                               # half of it, from an earlier job
    url = f'http://127.0.0.1:{server.server_address[1]}/d.zip'
    try:
        if mode == 'head-refused':
            pipeline.download(url, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
            assert dest.read_bytes() == blob                         # continued and completed
        else:
            with pytest.raises(pipeline.StageError):
                pipeline.download(url, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
            assert dest.read_bytes() == blob[:1_000_000]             # kept for a corrected link
            assert not dest.with_name('dataset.zip.restart').exists()
    finally:
        server.shutdown()


# ------------------------------------------------------------------ round 13: the remaining ways a broken link could cost it
def test_a_failed_head_request_says_nothing_about_the_size(tmp_path, monkeypatch):
    tunnel = 'HTTP/1.1 200 Connection established\r\nContent-Length: 0\r\n\r\n'
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 35, tunnel, ''))
    assert pipeline.remote_size(SECRET, pipeline.Paths(tmp_path, 'r')) is None


def test_a_link_that_now_returns_something_smaller_keeps_the_partial_download(tmp_path, monkeypatch):
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(b'x' * 100)
    calls = _fake_curl(tmp_path, monkeypatch, [], total=25)           # e.g. a 25-byte JSON error message
    with pytest.raises(pipeline.StageError, match='larger than what the link returns now'):
        pipeline._download(SECRET, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
    assert calls == [] and dest.stat().st_size == 100


@posix_only
@pytest.mark.skipif(not __import__('shutil').which('curl'), reason='needs curl')
@pytest.mark.parametrize('mode', ['static-error-page', 'complete-without-size'])
def test_when_curl_adds_nothing_the_first_bytes_decide(tmp_path, monkeypatch, mode):
    import http.server
    import re as _re
    blob = os.urandom(1_500_000)
    page = b'<!DOCTYPE html><html><body>Link expired</body></html>'
    served = page if mode == 'static-error-page' else blob

    class H(http.server.BaseHTTPRequestHandler):                     # a static server with byte ranges
        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html' if served is page else 'application/octet-stream')
            self.end_headers()                                       # no Content-Length: size unknown

        def do_GET(self):
            m = _re.match(r'bytes=(\d+)-(\d*)', self.headers.get('Range') or '')
            if not m:
                self.send_response(200)
                self.send_header('Content-Length', str(len(served)))
                self.end_headers()
                return self.wfile.write(served)
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else len(served) - 1, len(served) - 1)
            if start >= len(served):
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{len(served)}')
                self.send_header('Content-Length', '0')
                return self.end_headers()
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(served)}')
            self.send_header('Content-Length', str(end - start + 1))
            self.end_headers()
            self.wfile.write(served[start:end + 1])

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(pipeline, 'DOWNLOAD_PAUSE', 0)
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    monkeypatch.setenv('no_proxy', '127.0.0.1')
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(blob if mode == 'complete-without-size' else blob[:1_000_000])
    url = f'http://127.0.0.1:{server.server_address[1]}/d.zip'
    a, p = argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r')
    try:
        if mode == 'static-error-page':
            with pytest.raises(pipeline.StageError, match='no longer returns the file'):
                pipeline.download(url, dest, a, p, 'x')
            assert dest.read_bytes() == blob[:1_000_000]             # the partial download is kept
        else:
            pipeline.download(url, dest, a, p, 'x')                  # finished earlier, marker lost: accepted
            assert dest.read_bytes() == blob and dest.with_name('dataset.zip.complete').exists()
    finally:
        server.shutdown()


# ------------------------------------------------------------------ round 14 (ChatGPT's final review)
def test_a_new_copy_that_is_not_the_announced_size_never_replaces_the_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'DOWNLOAD_ATTEMPTS', 3)
    dest = tmp_path / 'dataset.zip'
    dest.write_bytes(b'p' * 500)
    calls = _fake_curl(tmp_path, monkeypatch, [(0, 33), (100, 0), (100, 0)], total=1000)
    with pytest.raises(pipeline.StageError, match='partial download is kept'):
        pipeline._download(SECRET, dest, argparse.Namespace(offline=False), pipeline.Paths(tmp_path, 'r'), 'x')
    assert len(calls) == 3 and dest.read_bytes() == b'p' * 500
    assert not dest.with_name('dataset.zip.restart').exists()


@posix_only
def test_stopping_a_step_also_ends_what_survives_its_first_process(tmp_path):
    pidfile = tmp_path / 'pid'
    proc = subprocess.Popen(['bash', '-c', f'(trap "" TERM; exec sleep 60) & echo $! > {pidfile}'],
                            start_new_session=True)
    proc.wait(timeout=10)                                   # the first process has already ended
    survivor = int(pidfile.read_text())
    assert _alive(survivor)
    started = time.time()
    pipeline.stop_process_group(proc, grace=1)               # TERM is ignored, so KILL must follow
    time.sleep(0.3)
    assert not _alive(survivor) and time.time() - started < 10


@posix_only
def test_submit_never_echoes_a_rejected_link(tmp_path):
    _links(tmp_path / 'base')
    out, calls = _submit(tmp_path, 'full', '--weights-url=https://private.invalid/SECRET12345')
    assert out.returncode == 1 and 'SECRET12345' not in out.stderr + out.stdout and not calls


@posix_only
def test_submit_takes_over_a_dead_lock_only_one_at_a_time(tmp_path):
    base = tmp_path / 'base'
    _links(base)
    lock = base / '.submit.lock'
    lock.mkdir()
    dead = subprocess.Popen(['true'])
    dead.wait()
    (lock / 'owner').write_text(f'{os.uname().nodename} {dead.pid}\n')
    (base / '.submit.lock.reclaim').mkdir()                  # another submit.sh is taking it over right now
    out, calls = _submit(tmp_path, 'smoke')
    assert out.returncode == 1 and 'another submit.sh is running' in out.stderr and not calls
    (base / '.submit.lock.reclaim').rmdir()
    out, calls = _submit(tmp_path, 'smoke')
    assert out.returncode == 0 and len(calls) == 1, out.stderr
    assert not lock.exists() and not (base / '.submit.lock.reclaim').exists()
