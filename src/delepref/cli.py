"""Command-line inference, cached real-encoder demos and asset utilities."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from .inference import encode_sequences,file_sha,load_heads,load_profile,plan_from_embeddings,validate_sequence
from .paths import packaged_asset


def read_fasta(path):
    records=[];name=None;parts=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line:continue
        if line.startswith('>'):
            if name is not None:records.append((name,validate_sequence(''.join(parts).upper())))
            name=line[1:].split()[0] if line[1:].split() else None;parts=[]
            if not name:raise ValueError('Empty FASTA identifier')
        else:
            if name is None:raise ValueError('FASTA sequence before its identifier')
            parts.append(line)
    if name is not None:records.append((name,validate_sequence(''.join(parts).upper())))
    if not records or len({p for p,s in records})!=len(records):raise ValueError('Empty FASTA or duplicate identifiers')
    return records


def write_results(output,results):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    with (output/'shortened.fasta').open('w',encoding='ascii') as f:
        for result in results:
            f.write('>{}\n{}\n'.format(result['protein_id'],result['selected']['guided']['edited_sequence']))


def demo(args):
    runtime=load_heads(args.weights,args.model_config);results=[]
    cases=['sec6','af1862'] if args.case=='all' else [args.case]
    for name in cases:
        base=packaged_asset('examples',name)
        provenance=json.loads((base/'provenance.json').read_text())
        if not provenance['real_ESMC_representation'] or file_sha(base/'embedding.safetensors')!=provenance['embedding_sha256']:
            raise ValueError('Authentic demo embedding identity mismatch')
        pid,sequence=read_fasta(base/'input.fasta')[0]
        h=load_file(str(base/'embedding.safetensors'))['residue_embeddings']
        profile=None if args.without_homology else load_profile(base/'profile.json',pid,sequence)
        result=plan_from_embeddings(runtime,pid,sequence,h,budget=.2,seed=13,profile=profile)
        result.update(embedding_source='frozen_real_ESMC_cache',new_encoder_calls=0,demo_case=name)
        results.append(result)
        print(json.dumps({'case':name,'WT_length':len(sequence),'output_length':len(result['selected']['guided']['edited_sequence']),
              'kind':result['selected']['guided']['generation_kind'],'fallback_exact':result['fallback_exact']}))
    write_results(args.out,results)


def shrink(args):
    records=read_fasta(args.fasta);runtime=load_heads(args.weights,args.model_config)
    if args.embedding:
        if len(records)!=1:raise ValueError('An explicit embedding file currently requires exactly one WT')
        h={records[0][0]:load_file(str(args.embedding))['residue_embeddings']}
        source='explicit_cached_embedding'
    else:
        print('DELEPREF_ESMC_LOAD_START',flush=True)
        h=encode_sequences(records,args.encoder_snapshot,device='cuda',microbatch_size=args.microbatch_size,
            progress=lambda event:print('DELEPREF_'+event,flush=True))
        source='fresh_ESMC'
    results=[]
    for pid,sequence in records:
        profile=load_profile(args.profile,pid,sequence)
        r=plan_from_embeddings(runtime,pid,sequence,h[pid],budget=args.budget,seed=args.seed,profile=profile)
        r['embedding_source']=source;results.append(r)
    write_results(args.out,results)


def main(argv=None):
    parser=argparse.ArgumentParser(prog='delepref',description='Homology-guided exact-budget protein shrinking')
    sub=parser.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo',help='CPU inference with real frozen ESMC embeddings and trained heads')
    d.add_argument('--case',choices=['sec6','af1862','all'],default='sec6')
    d.add_argument('--without-homology',action='store_true');d.add_argument('--out',type=Path,default=Path('delepref-demo-output'))
    d.add_argument('--weights',type=Path);d.add_argument('--model-config',type=Path);d.set_defaults(func=demo)
    s=sub.add_parser('shrink',help='Encode each fresh WT once, or read an explicitly supplied embedding')
    s.add_argument('--fasta',type=Path,required=True);s.add_argument('--budget',type=float,default=.2)
    s.add_argument('--seed',type=int,default=13);s.add_argument('--profile',type=Path)
    s.add_argument('--embedding',type=Path);s.add_argument('--encoder-snapshot',type=Path)
    s.add_argument('--microbatch-size',type=int,default=32);s.add_argument('--weights',type=Path)
    s.add_argument('--model-config',type=Path);s.add_argument('--out',type=Path,required=True);s.set_defaults(func=shrink)
    r=sub.add_parser('retrieve',help='MMseqs2 search and WT-coordinate projection')
    r.add_argument('--fasta',type=Path,required=True);r.add_argument('--database',type=Path,required=True)
    r.add_argument('--out',type=Path,required=True);r.add_argument('--mmseqs',default='mmseqs')
    r.add_argument('--threads',type=int,default=4);r.add_argument('--reviewed-targets',type=Path)
    def retrieve_command(a):
        from .retrieval import search_profiles
        search_profiles(read_fasta(a.fasta),a.database,a.out,executable=a.mmseqs,threads=a.threads,reviewed_targets=a.reviewed_targets)
    r.set_defaults(func=retrieve_command)
    a=sub.add_parser('assets',help='List or download versioned benchmark/reproduction assets')
    a.add_argument('action',choices=['list','download']);a.add_argument('--asset',default='paper_tables')
    a.add_argument('--out',type=Path,default=Path('delepref-data'));a.add_argument('--source-dir',type=Path)
    a.add_argument('--revision')
    def assets_command(a):
        from .downloads import assets_command as execute
        execute(a)
    a.set_defaults(func=assets_command)
    p=sub.add_parser('reproduce',help='Recreate tables and figures from frozen measurements, without folding')
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sec6',action='store_true')
    def reproduce_command(a):
        from .reproduction import reproduce
        reproduce(a.data_dir,a.out,sec6=a.sec6)
    p.set_defaults(func=reproduce_command)
    args=parser.parse_args(argv);torch.set_num_threads(4)
    args.func(args)


if __name__=='__main__':main()
