"""Read-only SCISOR provenance audit; inspect pickle opcodes without unpickling."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickletools
import zipfile

from delepref_v8_baseline import ROOT,write

SOURCE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
PROTOCOL=SOURCE/'results/experiments/student_v3_step004000_v3_25_binder50_fair_contact_comparison/protocol.json'


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()


def inventory():
    protocol=json.loads(PROTOCOL.read_text());inherited=protocol['methods']['scisor']
    checkpoint=Path(inherited['checkpoint']);p0=Path(inherited['p0'])
    assets={}
    for name,path in [('checkpoint',checkpoint),('p0',p0)]:
        digest=sha(path)
        if digest!=inherited[name+'_sha256']:raise ValueError('inherited SCISOR '+name+' identity changed')
        assets[name]=dict(path=str(path),sha256=digest,size=path.stat().st_size)
    globals_=set();backbones=set();metadata_names=[]
    if zipfile.is_zipfile(checkpoint):
        with zipfile.ZipFile(checkpoint) as archive:
            metadata_names=[n for n in archive.namelist() if n.endswith('/data.pkl') or n=='data.pkl']
            if len(metadata_names)!=1 or archive.getinfo(metadata_names[0]).file_size>8*1024*1024:
                raise ValueError('unexpected checkpoint metadata container')
            for op,arg,_ in pickletools.genops(archive.read(metadata_names[0])):
                if op.name=='GLOBAL':globals_.add(str(arg))
                if isinstance(arg,str) and ('esm2_' in arg or arg.startswith('facebook/')):backbones.add(arg)
    code={str(p):sha(p) for p in sorted((SOURCE/'SCISOR').glob('*.py'))}
    code[str(SOURCE/'scripts/external_scisor.py')]=sha(SOURCE/'scripts/external_scisor.py')
    code[str(SOURCE/'scripts/run_delepref_25_binder50_fair_contact_comparison.py')]=sha(SOURCE/'scripts/run_delepref_25_binder50_fair_contact_comparison.py')
    versions={}
    for package in ('torch','transformers','pytorch-lightning','lightning','flash-attn'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    report=dict(status='SCISOR_SOURCE_AND_WEIGHT_IDENTITIES_VERIFIED_RUNTIME_PENDING',
        inherited_protocol=dict(path=str(PROTOCOL),sha256=sha(PROTOCOL)),assets=assets,code_sha256=code,
        checkpoint_metadata_inspection='pickletools opcodes only; no pickle execution or model loading',
        pickle_globals=sorted(globals_),backbone_strings=sorted(backbones),environment_package_versions=versions,
        old_default_path_missing=str(SOURCE/'models/scisor/SCISOR_U90_S.ckpt'),
        original_native_generation=dict(candidate_count=1,exact_budget_via_explicit_count=True,
            unlimited_spans=True,temperature=1.,alpha_rate='1/1.1',explicit_preserved_indices=True),
        comparison='native single-plan SCISOR reported separately from K8 same-Scorer methods; no equal-K/compute claim',
        upstream_revision_and_license='not yet independently verified for this server-local source copy',
        external_fixed_variant_scoring='not established by native shrink_sequence interface',
        next='verify referenced local backbone/tokenizer and import/runtime compatibility, then freeze generation before Slurm inference',
        runtime_model_loaded=False,GPU_calls=0,new_structure_predictions=0,paper_changes=False)
    out=ROOT/'reports/delepref_v8_scisor_inventory.json';write(out,report)
    print(json.dumps(dict(status=report['status'],report=str(out),checkpoint_sha256=assets['checkpoint']['sha256'],
        backbone_strings=report['backbone_strings'],runtime_model_loaded=False)),flush=True)


if __name__=='__main__':inventory()
