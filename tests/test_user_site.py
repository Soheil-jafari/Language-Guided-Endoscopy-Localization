"""The project environment never uses or changes the user's personal Python packages (~/.local).

On the project cluster, the supervisor's own packages in ~/.local were visible inside the new conda
environment: pip counted some of them as already installed (the environment then broke when they
changed) and uninstalled others from ~/.local while upgrading them (which broke the supervisor's own work).
"""
import argparse
import base64
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import main as pipeline

REPO = Path(__file__).resolve().parent.parent
posix_only = pytest.mark.skipif(os.name == 'nt', reason='shell scripts for Linux/HPC')
INSTALLER_ISOLATION = ('export PYTHONNOUSERSITE=1', 'unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_ROOT',
                       'export PIP_USER=0')


def _code(script):
    return '\n'.join(l for l in (REPO / script).read_text().splitlines() if not l.lstrip().startswith('#'))


@pytest.mark.parametrize('script,needed,first_uses', [
    ('setup_env.sh', INSTALLER_ISOLATION, ('conda info', 'conda create', 'pip install', '"$PY"')),
    ('install_mamba.sh', INSTALLER_ISOLATION, ('"$PY"',)),
    ('hpc/lgel.sbatch', ('export PYTHONNOUSERSITE=1', 'unset PYTHONPATH PYTHONHOME'), ('run.sh"',)),
    ('run.sh', ('export PYTHONNOUSERSITE=1', 'unset PYTHONPATH PYTHONHOME'), ('conda info', '$CONDA_PREFIX/bin/python')),
])
def test_every_script_isolates_python_and_pip_before_they_first_run(script, needed, first_uses):
    code = _code(script)
    for line in needed:
        assert line in code, (script, line)
        for use in first_uses:
            assert code.index(line) < code.index(use), (script, line, use)


def _wheel(directory):
    """A minimal local wheel, so that pip can be exercised without a network."""
    name = 'lgeldummy-1.0'
    files = {'lgeldummy/__init__.py': b'X = 1\n',
             f'{name}.dist-info/METADATA': b'Metadata-Version: 2.1\nName: lgeldummy\nVersion: 1.0\n',
             f'{name}.dist-info/WHEEL': b'Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
    record = ''.join(f'{k},sha256={base64.urlsafe_b64encode(hashlib.sha256(v).digest()).rstrip(b"=").decode()},{len(v)}\n'
                     for k, v in files.items()) + f'{name}.dist-info/RECORD,,\n'
    with zipfile.ZipFile(directory / 'lgeldummy-1.0-py3-none-any.whl', 'w') as z:
        for k, v in files.items():
            z.writestr(k, v)
        z.writestr(f'{name}.dist-info/RECORD', record)


def test_pip_user_0_overrides_a_pip_configuration_that_asks_for_user_installs(tmp_path):
    """The mechanism the installers rely on: PIP_USER=0 wins over `user = true` in a pip configuration file."""
    _wheel(tmp_path)
    (tmp_path / 'pip.conf').write_text('[install]\nuser = true\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
    env.update(PIP_CONFIG_FILE=str(tmp_path / 'pip.conf'), PIP_DISABLE_PIP_VERSION_CHECK='1')
    cmd = [sys.executable, '-m', 'pip', 'install', '--dry-run', '--no-index', '--find-links', str(tmp_path), 'lgeldummy']
    asked = subprocess.run(cmd, env=env, capture_output=True, text=True)
    overridden = subprocess.run(cmd, env=dict(env, PIP_USER='0'), capture_output=True, text=True)
    if sys.prefix == getattr(sys, 'base_prefix', sys.prefix):   # not a virtualenv: a user install would be allowed
        assert 'Would install' in overridden.stdout
        return
    assert asked.returncode != 0 and '--user' in asked.stderr                  # the configuration asks for --user
    assert overridden.returncode == 0 and 'Would install lgeldummy-1.0' in overridden.stdout


def test_stage_subprocesses_get_the_repository_on_their_path_and_no_empty_entry(tmp_path, monkeypatch):
    p = pipeline.Paths(tmp_path, 'r')
    monkeypatch.delenv('PYTHONPATH', raising=False)
    assert pipeline.stage_env(argparse.Namespace(offline=False), p)['PYTHONPATH'] == str(pipeline.REPO)
    monkeypatch.setenv('PYTHONPATH', '/extra')
    assert pipeline.stage_env(argparse.Namespace(offline=False), p)['PYTHONPATH'] == str(pipeline.REPO) + os.pathsep + '/extra'


# ------------------------------------------------------------------ run.sh with a conda stand-in
def _fake_conda(tmp_path):
    """A conda stand-in. Its python fails the core import check while <tmp>/BROKEN exists, and the Mamba
    import check while <tmp>/MAMBA_BROKEN exists; everything else runs the real interpreter."""
    env = tmp_path / 'env'
    (env / 'bin').mkdir(parents=True)
    py = env / 'bin' / 'python'
    py.write_text(f'''#!/bin/bash
if [ "$1" = -c ] && [[ "$2" == *"import torch"* ]]; then [ -e "{tmp_path}/BROKEN" ] && exit 1; exit 0; fi
if [ "$1" = -c ] && [[ "$2" == *"import mamba_ssm"* ]]; then [ -e "{tmp_path}/MAMBA_BROKEN" ] && exit 1; exit 0; fi
exec "{sys.executable}" "$@"
''')
    py.chmod(0o755)
    base = tmp_path / 'condabase'
    (base / 'etc' / 'profile.d').mkdir(parents=True)
    (base / 'etc' / 'profile.d' / 'conda.sh').write_text('''
conda() {
  case "$1" in
    activate) export CONDA_PREFIX="$2"; export PATH="$2/bin:$PATH" ;;
    deactivate) unset CONDA_PREFIX ;;
  esac
}
''')
    stub = tmp_path / 'bin'
    stub.mkdir()
    (stub / 'conda').write_text(f'#!/bin/bash\n[ "$1" = info ] && echo "{base}"\n')
    (stub / 'conda').chmod(0o755)
    return env, stub


def _fake_repo(tmp_path, setup_fails=False):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'run.sh').write_text((REPO / 'run.sh').read_text())
    (repo / 'requirements.txt').write_text('torch==2.5.1\n')
    # install_mamba.sh stand-in: records the call and repairs Mamba
    (repo / 'install_mamba.sh').write_text(f'echo "MAMBA RETRY" >> "{tmp_path}/calls"; rm -f "{tmp_path}/MAMBA_BROKEN"\n')
    # setup_env.sh stand-in: records the call and its environment, and repairs (or fails)
    (repo / 'setup_env.sh').write_text(
        f'echo "SETUP RAN nousersite=$PYTHONNOUSERSITE pythonpath=${{PYTHONPATH-unset}}" >> "{tmp_path}/calls"\n'
        + ('exit 1\n' if setup_fails else f'rm -f "{tmp_path}/BROKEN"\n'))
    (repo / 'main.py').write_text('import os\n'
                                  'print("MAIN RAN nousersite=%s pythonpath=%s" % (os.environ.get("PYTHONNOUSERSITE"), '
                                  'os.environ.get("PYTHONPATH", "unset")))\n')
    return repo


def _stamp(repo):
    h = hashlib.sha256()
    for f in ('requirements.txt', 'setup_env.sh', 'install_mamba.sh'):
        h.update((repo / f).read_bytes())
    return h.hexdigest()


def _run_sh(tmp_path, broken=False, mamba_broken=False, offline=False, setup_fails=False, stale_stamp=False):
    for flag, on in (('BROKEN', broken), ('MAMBA_BROKEN', mamba_broken)):
        if on:
            (tmp_path / flag).write_text('')
    env_dir, stub = _fake_conda(tmp_path)
    repo = _fake_repo(tmp_path, setup_fails)
    (env_dir / '.lgel_env_ready').write_text(('old' if stale_stamp else _stamp(repo)) + '\n')
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONNOUSERSITE', 'CONDA_PREFIX')}
    env.update(PATH=f'{stub}:{os.environ["PATH"]}', LGEL_ENV_PREFIX=str(env_dir),
               PYTHONPATH='/users/someone/.local/lib/python3.12/site-packages')      # e.g. set in a login profile
    out = subprocess.run(['bash', str(repo / 'run.sh'), *(['--offline'] if offline else [])],
                         env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / 'calls').read_text() if (tmp_path / 'calls').exists() else ''
    return out, calls


@posix_only
def test_run_sh_starts_main_with_only_the_environments_packages(tmp_path):
    out, calls = _run_sh(tmp_path)
    assert out.returncode == 0, out.stderr
    assert 'MAIN RAN nousersite=1 pythonpath=unset' in out.stdout           # inherited PYTHONPATH cleared
    assert not calls


@posix_only
@pytest.mark.parametrize('stale_stamp', [False, True])
def test_run_sh_repairs_an_environment_that_no_longer_imports_or_is_out_of_date(tmp_path, stale_stamp):
    out, calls = _run_sh(tmp_path, broken=not stale_stamp, stale_stamp=stale_stamp)
    assert out.returncode == 0, out.stderr
    reason = 'is incomplete or out of date' if stale_stamp else 'can no longer import all of its packages'
    assert reason in out.stdout
    assert calls == 'SETUP RAN nousersite=1 pythonpath=unset\n'               # setup runs isolated as well
    assert 'MAIN RAN nousersite=1' in out.stdout


@posix_only
def test_run_sh_stops_when_the_repair_fails(tmp_path):
    out, calls = _run_sh(tmp_path, broken=True, setup_fails=True)
    assert out.returncode != 0 and 'SETUP RAN' in calls and 'MAIN RAN' not in out.stdout


@posix_only
def test_offline_jobs_never_install_they_say_what_to_do(tmp_path):
    out, calls = _run_sh(tmp_path, broken=True, offline=True)
    assert out.returncode != 0 and 'can no longer import' in out.stderr + out.stdout and not calls


@posix_only
@pytest.mark.parametrize('offline', [False, True])
def test_mamba_that_stopped_importing_is_installed_again_without_a_missing_marker(tmp_path, offline):
    out, calls = _run_sh(tmp_path, mamba_broken=True, offline=offline)
    assert out.returncode == 0, out.stderr
    assert 'MAIN RAN' in out.stdout                                            # never fatal: the baseline needs no Mamba
    assert calls == ('' if offline else 'MAMBA RETRY\n')
