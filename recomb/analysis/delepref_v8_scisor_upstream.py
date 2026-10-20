"""Fetch small official provenance files; never download model weights."""
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url,limit=2*1024*1024):
    request=urllib.request.Request(url,headers={'User-Agent':'DelePref-baseline-provenance'})
    with urllib.request.urlopen(request,timeout=40) as response:
        raw=response.read(limit+1)
    if len(raw)>limit:raise ValueError('public metadata/source larger than declared limit')
    return raw


def main():
    root=Path.cwd();out=root/'provenance/scisor_upstream_20261016';out.mkdir(exist_ok=False)
    commit=json.loads(fetch('https://api.github.com/repos/baronet2/SCISOR/commits/main'))
    revision=commit['sha']
    names=['LICENSE.md','shrink_sequences.py','SCISOR/shortening_scud.py','SCISOR/shortening_diffusion.py',
        'SCISOR/continuous_time_diffusion.py','SCISOR/faesm.py','SCISOR/esm.py','SCISOR/alignments.py','SCISOR/utils.py']
    files={}
    for name in names:
        url='https://raw.githubusercontent.com/baronet2/SCISOR/'+revision+'/'+name
        raw=fetch(url);path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        files[name]=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    hf=json.loads(fetch('https://huggingface.co/api/models/SCISOR/SCISOR?blobs=true'))
    checkpoint=next(x for x in hf['siblings'] if x['rfilename']=='SCISOR_U90_S.ckpt')
    expected=json.loads((root/'reports/delepref_v8_scisor_inventory.json').read_text())['assets']['checkpoint']
    if checkpoint['lfs']['sha256']!=expected['sha256'] or checkpoint['lfs']['size']!=expected['size']:
        raise ValueError('cached SCISOR checkpoint does not match official HF object')
    report=dict(status='OFFICIAL_SCISOR_WEIGHT_OBJECT_AND_SOURCE_REVISION_RECORDED',
        github=dict(repository='https://github.com/baronet2/SCISOR',revision=revision,commit_time=commit['commit']['committer']['date'],
            license_url='https://github.com/baronet2/SCISOR/blob/'+revision+'/LICENSE.md'),
        HF=dict(repository='https://huggingface.co/SCISOR/SCISOR',revision=hf['sha'],file=checkpoint,
            license=hf.get('cardData',{}).get('license'),cached_checkpoint_matches_official_LFS=True),
        files=files,model_weights_downloaded=False,source_copy_diff_not_yet_reviewed=True,paper_changes=False)
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (root/'reports/delepref_v8_scisor_upstream.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],github_revision=revision,HF_revision=hf['sha'],weight_matches=True)),flush=True)


if __name__=='__main__':main()
