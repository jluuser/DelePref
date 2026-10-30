"""Execute the real batch script with an argv-capture stub, without running a model."""
import os
from pathlib import Path
import subprocess
import pytest


def launcher_path():
    source=Path(__file__).resolve().parents[2]/'slurm/cache_plm.sbatch'
    if source.exists():return source
    # Frozen computational snapshots contain code, while the Slurm controller lives at the experiment root.
    from common import AUTHORIZED_ROOT
    return AUTHORIZED_ROOT/'slurm/cache_plm.sbatch'


@pytest.mark.parametrize('mode,prefix', [('pilot',''),('full',''),('pilot','confirmation'),('full','confirmation')])
def test_cache_launcher_actual_bash(mode,prefix,tmp_path):
    capture=tmp_path/'argv.bin';stub=tmp_path/'python-stub'
    stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$TEST_CAPTURE"\n')
    stub.chmod(0o755)
    env=dict(os.environ,PLM_PYTHON=str(stub),CACHE_MODE=mode,SLURM_ARRAY_TASK_ID='0',TEST_CAPTURE=str(capture))
    env.pop('CACHE_INDEX_PREFIX',None)
    if prefix:env['CACHE_INDEX_PREFIX']=prefix
    script=launcher_path()
    result=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    argv=capture.read_bytes().decode().split('\0')[:-1]
    assert argv[0].endswith('/code/cache_plm.py')
    assert '--execute' in argv
    assert ('--pilot' in argv)==(mode=='pilot')
    assert ('--index-prefix' in argv)==bool(prefix)
    if prefix:assert argv[argv.index('--index-prefix')+1]==prefix
    assert '' not in argv


def test_launcher_lookup_inside_computational_snapshot(monkeypatch,tmp_path):
    import sys
    from common import AUTHORIZED_ROOT
    monkeypatch.setattr(sys.modules[__name__],'__file__',str(tmp_path/'snapshot/code/tests/test_slurm_launch.py'))
    assert launcher_path()==AUTHORIZED_ROOT/'slurm/cache_plm.sbatch'
