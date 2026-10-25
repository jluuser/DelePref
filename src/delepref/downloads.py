"""Checksum-verified assets. No third-party large model is downloaded implicitly."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import urllib.request
import urllib.parse
import zipfile


def catalog():
    return json.loads((Path(__file__).resolve().parent/'assets/catalog.json').read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def extract_case(path,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True);base=destination.resolve()
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            target=(base/member.filename).resolve()
            if not target.is_relative_to(base) or stat.S_ISLNK(member.external_attr>>16):
                raise ValueError('Unsafe archive path/link')
        z.extractall(base)


def download(name,out,*,source_dir=None,revision=None):
    c=catalog();entry=c['assets'].get(name)
    if entry is None:raise ValueError('Unknown asset: '+name)
    if entry['status']=='recipe_only':raise ValueError(entry['note'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True);target=out/entry['filename']
    if target.exists():
        if sha(target)!=entry['sha256']:raise ValueError('Existing asset has a different hash; choose another output directory')
    elif source_dir is not None and 'repository_path' in entry:
        source=Path(source_dir)/entry['repository_path']
        if sha(source)!=entry['sha256']:raise ValueError('Local asset source checksum mismatch')
        shutil.copyfile(source,target)
    else:
        revision=revision or c['default_revision']
        url=entry.get('url') or 'https://raw.githubusercontent.com/{}/{}/{}'.format(
            c['repository'],urllib.parse.quote(revision,safe='/'),entry['repository_path'])
        request=urllib.request.Request(url,headers={'User-Agent':'DelePref-asset-downloader/1.1'})
        # Only an explicitly supplied token is used, and only for this GitHub resource.
        token=os.environ.get('DELEPREF_GITHUB_TOKEN')
        if token and urllib.parse.urlparse(url).hostname=='raw.githubusercontent.com':request.add_header('Authorization','Bearer '+token)
        partial=target.with_suffix(target.suffix+'.partial')
        try:
            with urllib.request.urlopen(request,timeout=90) as response,partial.open('wb') as f:
                for block in iter(lambda:response.read(8*1024*1024),b''):f.write(block)
            if partial.stat().st_size!=entry['bytes'] or sha(partial)!=entry['sha256']:
                raise ValueError('Downloaded asset identity mismatch; partial file retained for diagnosis')
            partial.replace(target)
        except Exception as exc:
            raise RuntimeError('Asset download failed. For an authorized private checkout use --source-dir, or explicitly supply a GitHub token; no URL with credentials is logged.') from exc
    if target.stat().st_size!=entry['bytes'] or sha(target)!=entry['sha256']:raise ValueError('Asset size/checksum mismatch')
    if name=='sec6_case':extract_case(target,out/'sec6_case')
    return target


def assets_command(args):
    c=catalog()
    if args.action=='list':
        print(json.dumps(c,indent=2));return
    names=[k for k,v in c['assets'].items() if v['status']=='included'] if args.asset=='all' else [args.asset]
    for name in names:
        path=download(name,args.out,source_dir=args.source_dir,revision=args.revision)
        print(json.dumps({'asset':name,'path':str(path),'verified':True}))
