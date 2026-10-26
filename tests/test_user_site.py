"""The project environment never uses or changes the user's personal Python packages (~/.local).

On the project cluster, the supervisor's own packages in ~/.local were visible inside the new conda
environment: pip counted some of them as already installed (the environment then broke when they
changed) and uninstalled others from ~/.local while upgrading them (which broke the supervisor's own work).
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
posix_only = pytest.mark.skipif(os.name == 'nt', reason='shell scripts for Linux/HPC')


@pytest.mark.parametrize('script,first_use', [
    ('setup_env.sh', 'pip install'),
    ('install_mamba.sh', '"$PY"'),
    ('hpc/lgel.sbatch', 'run.sh'),
    ('run.sh', 'conda info'),
])
def test_every_script_hides_the_users_packages_before_python_or_pip_runs(script, first_use):
    text = (REPO / script).read_text()
    code = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('#'))
    assert 'export PYTHONNOUSERSITE=1' in code
    assert code.index('export PYTHONNOUSERSITE=1') < code.index(first_use)


def _fake_conda(tmp_path, broken):
    """A conda stand-in: an environment whose python fails the import check while `broken` exists."""
    env = tmp_path / 'env'
    (env / 'bin').mkdir(parents=True)
    py = env / 'bin' / 'python'
    py.write_text(f'''#!/bin/bash
if [ "$1" = -c ] && [[ "$2" == *"import torch"* ]]; then [ -e "{broken}" ] && exit 1; exit 0; fi
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


def _fake_repo(tmp_path, broken):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'run.sh').write_text((REPO / 'run.sh').read_text())
    (repo / 'requirements.txt').write_text('torch==2.5.1\n')
    (repo / 'install_mamba.sh').write_text('exit 0\n')
    # setup_env.sh stand-in: records the call and repairs the environment
    (repo / 'setup_env.sh').write_text(f'echo "SETUP RAN nousersite=$PYTHONNOUSERSITE" >> "{tmp_path}/calls"; rm -f "{broken}"\n')
    (repo / 'main.py').write_text('import os, sys, site\n'
                                  'print("MAIN RAN nousersite=%s user_site=%s" % (os.environ.get("PYTHONNOUSERSITE"), '
                                  'site.ENABLE_USER_SITE))\n')
    return repo


def _stamp(repo):
    h = hashlib.sha256()
    for f in ('requirements.txt', 'setup_env.sh', 'install_mamba.sh'):
        h.update((repo / f).read_bytes())
    return h.hexdigest()


@posix_only
@pytest.mark.parametrize('broken_env,offline', [(False, False), (True, False), (True, True)])
def test_run_sh_uses_only_the_environment_and_repairs_one_that_no_longer_imports(tmp_path, broken_env, offline):
    broken = tmp_path / 'BROKEN'
    if broken_env:
        broken.write_text('')
    env_dir, stub = _fake_conda(tmp_path, broken)
    repo = _fake_repo(tmp_path, broken)
    (env_dir / '.lgel_env_ready').write_text(_stamp(repo) + '\n')       # set up, and up to date
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONNOUSERSITE', 'CONDA_PREFIX')}
    env.update(PATH=f'{stub}:{os.environ["PATH"]}', LGEL_ENV_PREFIX=str(env_dir))
    args = ['--offline'] if offline else []
    out = subprocess.run(['bash', str(repo / 'run.sh'), *args], env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / 'calls').read_text() if (tmp_path / 'calls').exists() else ''
    if offline:                                   # an --offline job never installs: it says what to do instead
        assert out.returncode != 0 and 'can no longer import' in out.stderr + out.stdout and not calls
        return
    assert out.returncode == 0, out.stderr
    assert 'MAIN RAN nousersite=1 user_site=False' in out.stdout
    if broken_env:
        assert 'can no longer import all of its packages' in out.stdout
        assert calls == 'SETUP RAN nousersite=1\n'
    else:
        assert not calls
