"""Archive new experiment outputs and scripts, with per-member hashes; omit weights/cache DBs."""
import hashlib
import json
from pathlib import Path
import zipfile
from recomb_expanded_external_prepare_20261017 import ROOT


def main():
    out=ROOT/'reports/recomb_task_aligned_final_20261017';archive=out/'full_outputs_verified.zip'
    dirs=['external/recomb_expanded_20261017','reports/recomb_task_budget_20261017',
          'reports/recomb_task_structure_20261017','reports/recomb_task_structure_20261017_v2',
          'reports/recomb_task_structures_20261017','reports/recomb_task_structures_20261017_v2',
          'reports/recomb_functional_annotations_20261017','reports/recomb_legacy_evidence_20261017',
          'reports/recomb_cached_diagnostics_20261017','reports/recomb_task_aligned_final_20261017',
          'provenance/recomb_functional_annotations_20261017']
    files=[]
    for directory in dirs:
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or p.is_symlink() or any(part.startswith('tmp_') for part in p.parts):continue
            if p.suffix not in ('.json','.jsonl','.csv','.tsv','.fasta','.log','.npz','.md','.png','.pdf'):continue
            if p==archive or p.name=='archive_manifest.json':continue
            p.resolve().relative_to(ROOT.resolve());files.append(p)
    files+=list((ROOT/'configs').glob('recomb*20261017*.json'))
    files+=list((ROOT/'analysis').glob('recomb*20261017*.py'))
    files+=list((ROOT/'slurm').glob('recomb*20261017*.sbatch'))
    files+=[ROOT/'reports/recomb_task_aligned_execution_20261017.json']
    files=sorted(set(files));manifest=[]
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            b=p.read_bytes();name=str(p.relative_to(ROOT));z.writestr(name,b)
            manifest.append(dict(file=name,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
        z.writestr('member_manifest.json',json.dumps(manifest,indent=2).encode())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    result=dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,
        members=len(files)+1,models_or_embedding_caches_included=False,
        old641_archive='Retained separately in existing caches; references and full hashes preserved',member_manifest_inside_archive=True)
    (out/'archive_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
