"""Conservative source/identity filtering of staged MSAs for reviewed WT constructs."""
import concurrent.futures
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
from recomb_confirm_v2_diagnostics import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external/recomb_confirm_v2_WT_MSA_staging'
OUT=ROOT/'external/recomb_confirm_v2_WT_MSA_reviewed'
PREFIX=128
# Taxonomic source groups are reviewed for reference use on already-approved WT,
# not a blanket approval of new design targets. Concerning function text is held.
SOURCE_GROUPS=('Mammalia','Viridiplantae','Archaea')
MODEL_SOURCES=('Saccharomyces cerevisiae','Schizosaccharomyces pombe',
    'Neurospora crassa','Escherichia coli (strain K12)',
    'Bacillus subtilis (strain 168)','Thermus thermophilus','Thermus aquaticus',
    'Thermotoga maritima','Aquifex aeolicus','Hydrogenobacter thermophilus',
    'Geobacillus stearothermophilus','Geobacillus thermodenitrificans',
    'Chloroflexus aurantiacus','Synechococcus elongatus','Synechocystis sp.',
    'Streptomyces coelicolor','Lactococcus lactis','Lactiplantibacillus plantarum')
CONCERN=re.compile(r'toxin|virulen|pathogen|venom|hemolysin|haemolysin|'
    r'antibiotic resistance|antimicrobial resistance|beta.lactamase|shiga|ricin',re.I)


def records(path):
    result=[];head=None;sequence=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('>'):
            if head is not None:result.append((head,''.join(sequence)))
            head=line[1:];sequence=[]
        elif line.strip():sequence.append(line.strip())
    if head is not None:result.append((head,''.join(sequence)))
    return result


def source_class(entry):
    organism=entry.get('organism',{});name=organism.get('scientificName','')
    lineage=organism.get('lineage',[])
    text=json.dumps({k:entry.get(k) for k in ('proteinDescription','keywords','comments')},ensure_ascii=False)
    if CONCERN.search(text):return 'HOLD_FUNCTION_REVIEW'
    if any(g in lineage for g in SOURCE_GROUPS) or any(name.startswith(n) for n in MODEL_SOURCES):
        return 'SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE'
    return 'HOLD_SOURCE_NOT_POSITIVELY_REVIEWED'


def row_identity(header,aligned,canonical):
    fragment=aligned.replace('-','').upper()
    if not fragment or any(a not in 'ACDEFGHIKLMNPQRSTVWY' for a in fragment):return False
    fields=header.split('\t')
    if len(fields)>=10:
        try:
            a,b,n=int(fields[-3]),int(fields[-2]),int(fields[-1])
            if n==len(canonical) and canonical[a:b+1]==fragment:return True
        except ValueError:pass
    return fragment in canonical


def fetch_batch(accessions):
    key=hashlib.sha256('|'.join(accessions).encode()).hexdigest()
    cached=OUT/'metadata_batches'/(key+'.json')
    if cached.exists():return read(cached)
    query=' OR '.join('accession:'+a for a in accessions)
    url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode({'query':'('+query+')','format':'json'})
    error=None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'RECOMB-source-identity-audit/2.0'})
            with urllib.request.urlopen(req,timeout=80) as handle:raw=handle.read()
            value=json.loads(raw);results={}
            for e in value.get('results',[]):
                canonical=e.get('sequence',{}).get('value','')
                accession=e['primaryAccession'];status=source_class(e)
                matches=[]
                for row in ROWS_BY_ACCESSION[accession]:
                    if row_identity(row['header'],row['aligned'],canonical):matches.append(row['row_key'])
                results[accession]=dict(status=status,organism=e.get('organism'),
                    proteinDescription=e.get('proteinDescription'),keywords=e.get('keywords',[]),
                    canonical_sequence_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                    canonical_length=len(canonical),verified_row_keys=matches)
            record=dict(status='SUCCEEDED',requested=accessions,results=results,source_url=url,
                source_response_sha256=hashlib.sha256(raw).hexdigest(),raw_sequence_saved=False)
            write(cached,record);return record
        except Exception as e:
            error=repr(e)
            if attempt<2:time.sleep(2)
    record=dict(status='FAILED',requested=accessions,error=error,results={})
    write(cached,record);return record


ROWS_BY_ACCESSION=defaultdict(list)


def main():
    if (OUT/'completion.json').exists():raise FileExistsError('MSA review already completed')
    manifest=read(BASE/'request_manifest.json');completion=read(BASE/'completion.json')
    queries={r['sequence_sha256']:r for r in manifest['queries']}
    staged={};audits=[]
    for f in completion['files']:
        path=BASE/'a3m'/(f['sequence_sha256']+'.a3m')
        if sha(path)!=f['sha256']:raise ValueError('Staged MSA changed')
        rows=records(path);h=f['sequence_sha256'];query=queries[h]['sequence']
        if rows[0][1]!=query:raise ValueError('MSA query differs from reviewed WT/partner')
        staged[h]=rows
        for rank,(header,aligned) in enumerate(rows[1:PREFIX+1],1):
            row_key=h+':'+str(rank)
            token=header.split()[0]
            match=re.fullmatch(r'UniRef100_([A-Z0-9]+)',token)
            accession=match[1] if match and not match[1].startswith('UPI') else None
            record=dict(row_key=row_key,query_sha256=h,rank=rank,header=header,aligned=aligned,
                        accession=accession)
            if accession:ROWS_BY_ACCESSION[accession].append(record)
    accessions=sorted(ROWS_BY_ACCESSION)
    write(OUT/'review_policy.json',dict(prefix_rows=PREFIX,source_groups=SOURCE_GROUPS,
        model_sources=MODEL_SOURCES,function_concern_pattern=CONCERN.pattern,
        unknown_source_action='omit_reference_row_not_classify_as_dangerous',
        exact_query_and_donor_fragment_identity_required=True,new_design_target_approval=False,
        staged_manifest_sha256=sha(BASE/'request_manifest.json'),source_sha256=sha(Path(__file__))))
    batches=[accessions[i:i+60] for i in range(0,len(accessions),60)]
    entries={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for i,record in enumerate(executor.map(fetch_batch,batches),1):
            entries.update(record['results'])
            print('MSA_SOURCE_METADATA_BATCH '+str(i)+'/'+str(len(batches)),flush=True)
    files=[]
    for h,rows in staged.items():
        retained=[rows[0]];counts=Counter()
        for rank,(header,aligned) in enumerate(rows[1:PREFIX+1],1):
            token=header.split()[0];match=re.fullmatch(r'UniRef100_([A-Z0-9]+)',token)
            acc=match[1] if match and not match[1].startswith('UPI') else None
            e=entries.get(acc);row_key=h+':'+str(rank)
            if e is None:status='HOLD_METADATA_UNAVAILABLE'
            elif e['status']!='SOURCE_GROUP_ACCEPTED_FOR_REVIEWED_WT_REFERENCE':status=e['status']
            elif row_key not in e['verified_row_keys']:status='HOLD_DONOR_SEQUENCE_IDENTITY'
            else:status='RETAIN'
            counts[status]+=1
            audits.append(dict(query_sha256=h,rank=rank,accession=acc,status=status,
                organism=e.get('organism',{}).get('scientificName') if e else None))
            if status=='RETAIN':retained.append((header,aligned))
        dest=OUT/'a3m'/(h+'.a3m');dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(''.join('>'+head+'\n'+seq+'\n' for head,seq in retained),encoding='utf-8')
        files.append(dict(query_sha256=h,path=str(dest),sha256=sha(dest),rows=len(retained),
            independent_depth_not_established=True,source_disposition=dict(counts),
            roles=queries[h]['uses']))
    write(OUT/'row_source_and_identity_audit.json',audits)
    write(OUT/'completion.json',dict(status='SOURCE_FILTERED_WT_MSA_READY_FOR_CPU_PARSER_PREFLIGHT',
        unique_reviewed_queries=len(staged),donor_accessions_requested=len(accessions),
        rows_retained=sum(r['rows']-1 for r in files),query_only=sum(r['rows']==1 for r in files),
        not_unfiltered_standard_MSA=True,source_filter_is_a_disclosed_protocol_condition=True,
        source_disposition=dict(Counter(r['status'] for r in audits)),files=files,new_structure_predictions=0))
    print('MSA_SOURCE_FILTER_COMPLETE',flush=True)


if __name__=='__main__':main()
